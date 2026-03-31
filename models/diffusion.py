"""
Unified Transition Diffusion Model
====================================
Core diffusion model that handles:
  1. Training with independent timesteps (UWM style)
  2. Single transition sampling (conditional / unconditional)
  3. Chain planning with constrained denoising (CompDiffuser style)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple

from .noise_schedule import NoiseSchedule
from .denoiser import build_denoiser


class UnifiedTransitionDiffusion(nn.Module):
    """
    Unified Transition Diffusion Model.

    Learns the joint distribution p(initial, final) using independent
    diffusion timesteps for each part, following UWM.

    Supports:
      - Forward generation: p(final | initial)   [t_init=0, denoise t_final]
      - Backward generation: p(initial | final)  [denoise t_init, t_final=0]
      - Joint generation: p(initial, final)       [denoise both]
      - Chain planning with equality constraints  [CompDiffuser style]
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.config_dim = config['data']['config_dim']

        # Build denoiser
        self.denoiser = build_denoiser(config)

        # Build noise schedule
        diff_cfg = config['diffusion']
        self.noise_schedule = NoiseSchedule(
            num_timesteps=diff_cfg['num_timesteps'],
            beta_start=diff_cfg['beta_start'],
            beta_end=diff_cfg['beta_end'],
            schedule_type=diff_cfg['beta_schedule'],
        )
        self.num_timesteps = diff_cfg['num_timesteps']
        self.use_ddim = diff_cfg.get('use_ddim', True)
        self.ddim_steps = diff_cfg.get('ddim_steps', 50)

        # Training config
        train_cfg = config.get('training', {})
        self.w_initial = train_cfg.get('w_initial', 1.0)
        self.w_final = train_cfg.get('w_final', 1.0)

        # Conditioning probabilities (stgl style)
        # Per side independently: first drop with p_drop, then remaining split into overlap/inpainting
        self.tr_1side_drop_prob = train_cfg.get('tr_1side_drop_prob', 0.15)
        self.tr_ovlp_prob = train_cfg.get('tr_ovlp_prob', 0.5)
        self.tr_inpat_prob = train_cfg.get('tr_inpat_prob', 0.5)
        self.tr_no_ovlp_none = train_cfg.get('tr_no_ovlp_none', False)
        self.non_repla_inpat_prob = train_cfg.get('non_repla_inpat_prob', 0.5)

    def to(self, device):
        """Override to also move noise schedule."""
        super().to(device)
        self.noise_schedule.to(device)
        return self

    # ================================================================
    # Training
    # ================================================================

    def compute_loss(
        self,
        initial: torch.Tensor,   # (B, D) clean initial configs
        final: torch.Tensor,     # (B, D) clean final configs
    ) -> dict:
        """
        Compute training loss following stgl's create_train_tj_cond 3-step approach.

        Flow:
          1. q_sample both sides with base t → x_noisy (denoising targets)
          2. Two-step decision per side (drop / ovlp / inpat):
             - inpat: replace x_noisy with clean, set t=0, loss_weight=0
             - drop: zero out overlap conditioning (unconditional)
             - ovlp: separately q_sample clean state → overlap conditioning
          3. Assemble tj_cond dict, call denoiser
        """
        B = initial.shape[0]
        device = initial.device

        # ---- Step 1: Sample base timesteps & q_sample ----
        t_init = torch.randint(0, self.num_timesteps, (B,), device=device)
        t_final = torch.randint(0, self.num_timesteps, (B,), device=device)

        noise_init = torch.randn_like(initial)
        noise_final = torch.randn_like(final)

        initial_noisy = self.noise_schedule.q_sample(initial, t_init, noise_init)
        final_noisy = self.noise_schedule.q_sample(final, t_final, noise_final)

        # ---- Step 2: Two-step stgl decision per side ----
        p_drop = self.tr_1side_drop_prob

        init_is_drop = torch.rand(B, device=device) < p_drop
        final_is_drop = torch.rand(B, device=device) < p_drop
        
        init_cd_use_ovlp = torch.rand(B, device=device) < self.tr_ovlp_prob
        final_cd_use_ovlp = torch.rand(B, device=device) < self.tr_ovlp_prob
        
        init_cd_use_inpat = ~init_cd_use_ovlp
        final_cd_use_inpat = ~final_cd_use_ovlp
        
        ## set those to be dropout to False, so no condition at all for 0.15 * bs
        init_cd_use_ovlp[init_is_drop] = False
        init_cd_use_inpat[init_is_drop] = False

        final_cd_use_ovlp[final_is_drop] = False
        final_cd_use_inpat[final_is_drop] = False

        # Optional stgl heuristic: if one side is overlap-conditioned while the
        # opposite side is fully dropped, drop that overlap and randomly convert
        # part of it to inpainting. Default is disabled to preserve current behavior.
        if self.tr_no_ovlp_none:
            tmp_init_none = torch.logical_and(init_cd_use_ovlp, final_is_drop)
            tmp_init_to_inpat = torch.logical_and(
                tmp_init_none,
                torch.rand(B, device=device) < self.non_repla_inpat_prob,
            )
            init_cd_use_ovlp[tmp_init_none] = False
            init_cd_use_inpat[tmp_init_to_inpat] = True

            tmp_final_none = torch.logical_and(init_is_drop, final_cd_use_ovlp)
            tmp_final_to_inpat = torch.logical_and(
                tmp_final_none,
                torch.rand(B, device=device) < self.non_repla_inpat_prob,
            )
            final_cd_use_ovlp[tmp_final_none] = False
            final_cd_use_inpat[tmp_final_to_inpat] = True

        # ---- Step 3: Inpainting — replace x_noisy with clean, set t=0, loss_weight=0 ----
        loss_weight_init = torch.ones(B, device=device)
        loss_weight_final = torch.ones(B, device=device)

        if init_cd_use_inpat.any():
            initial_noisy[init_cd_use_inpat] = initial[init_cd_use_inpat]
            t_init[init_cd_use_inpat] = 0
            loss_weight_init[init_cd_use_inpat] = 0.0
        if final_cd_use_inpat.any():
            final_noisy[final_cd_use_inpat] = final[final_cd_use_inpat]
            t_final[final_cd_use_inpat] = 0
            loss_weight_final[final_cd_use_inpat] = 0.0

        # ---- Step 4: Build overlap conditioning tensors (SEPARATE from x_noisy) ----
        # For overlap samples: q_sample clean state with perturbed t
        init_ovlp_t = torch.clamp(
            t_init - torch.randint(0, 2, (B,), device=device),
            min=0, max=self.num_timesteps - 1,
        )
        final_ovlp_t = torch.clamp(
            t_final - torch.randint(0, 2, (B,), device=device),
            min=0, max=self.num_timesteps - 1,
        )
        init_ovlp_state = self.noise_schedule.q_sample(initial.detach(), init_ovlp_t)
        final_ovlp_state = self.noise_schedule.q_sample(final.detach(), final_ovlp_t)

        tj_cond = {
            'init_ovlp_state': init_ovlp_state,
            'final_ovlp_state': final_ovlp_state,
            'init_ovlp_t': init_ovlp_t,
            'final_ovlp_t': final_ovlp_t,
            'init_cd_use_ovlp': init_cd_use_ovlp,
            'final_cd_use_ovlp': final_cd_use_ovlp,
            'init_cd_use_inpat': init_cd_use_inpat,
            'final_cd_use_inpat': final_cd_use_inpat,
        }

        # ---- Step 5: Predict noise ----
        eps_init_pred, eps_final_pred = self.denoiser(
            initial_noisy, final_noisy, t_init, t_final, tj_cond=tj_cond
        )

        # ---- Step 6: Compute weighted losses ----
        loss_init_per_sample = ((eps_init_pred - noise_init) ** 2).mean(dim=-1)
        loss_final_per_sample = ((eps_final_pred - noise_final) ** 2).mean(dim=-1)

        loss_init = (loss_init_per_sample * loss_weight_init).sum() / loss_weight_init.sum().clamp_min(1.0)
        loss_final = (loss_final_per_sample * loss_weight_final).sum() / loss_weight_final.sum().clamp_min(1.0)

        total_loss = self.w_initial * loss_init + self.w_final * loss_final

        return {
            'loss': total_loss,
            'loss_init': loss_init.item(),
            'loss_final': loss_final.item(),
        }

    # ================================================================
    # Single Transition Sampling
    # ================================================================

    @torch.no_grad()
    def sample_transition(
        self,
        num_samples: int,
        device: torch.device,
        condition_initial: Optional[torch.Tensor] = None,  # (B, 22) or None
        condition_final: Optional[torch.Tensor] = None,    # (B, 22) or None
        guidance_scale: float = 1.0,
        use_ddim: bool = True,
        ddim_steps: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample transitions with flexible conditioning.

        Modes (following UWM):
          - condition_initial given, condition_final None:
              Forward dynamics p(final | initial)
          - condition_initial None, condition_final given:
              Backward dynamics p(initial | final)
          - Both None:
              Joint generation p(initial, final)
          - Both given:
              Infilling / refinement

        Args:
            num_samples: number of samples to generate
            device: torch device
            condition_initial: if provided, fix initial config (t_init=0)
            condition_final: if provided, fix final config (t_final=0)
            guidance_scale: CFG scale (>1 for stronger conditioning)
            use_ddim: whether to use DDIM sampling
            ddim_steps: number of DDIM steps

        Returns:
            initial_samples: (B, 22) generated/conditioned initial configs
            final_samples: (B, 22) generated/conditioned final configs
        """
        B = num_samples

        # Initialize with noise
        initial_t = torch.randn(B, self.config_dim, device=device)
        final_t = torch.randn(B, self.config_dim, device=device)

        # Determine which parts to denoise
        denoise_init = condition_initial is None
        denoise_final = condition_final is None

        # If conditioned, replace noise with clean data
        if not denoise_init:
            initial_t = condition_initial.clone()
        if not denoise_final:
            final_t = condition_final.clone()

        # Setup timestep schedule
        if use_ddim:
            timesteps = torch.linspace(self.num_timesteps - 1, 0, ddim_steps + 1, dtype=torch.long, device=device)
        else:
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1, device=device)

        for i in range(len(timesteps) - 1):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]

            # Build timestep tensors
            # For conditioned parts: t=0 (clean), for denoised parts: t=t_current
            t_init_tensor = torch.full((B,), t_current.item(), dtype=torch.long, device=device) if denoise_init else torch.zeros(B, dtype=torch.long, device=device)
            t_final_tensor = torch.full((B,), t_current.item(), dtype=torch.long, device=device) if denoise_final else torch.zeros(B, dtype=torch.long, device=device)

            # Build inference tj_cond: conditioned sides use inpaint tokens
            zero_state = torch.zeros(B, self.config_dim, device=device)
            zero_t = torch.zeros(B, dtype=torch.long, device=device)
            tj_cond = {
                'init_ovlp_state': zero_state,
                'final_ovlp_state': zero_state,
                'init_ovlp_t': zero_t,
                'final_ovlp_t': zero_t,
                'init_cd_use_ovlp': torch.zeros(B, dtype=torch.bool, device=device),
                'final_cd_use_ovlp': torch.zeros(B, dtype=torch.bool, device=device),
                'init_cd_use_inpat': torch.full((B,), not denoise_init, dtype=torch.bool, device=device),
                'final_cd_use_inpat': torch.full((B,), not denoise_final, dtype=torch.bool, device=device),
            }

            # Predict noise
            eps_init_pred, eps_final_pred = self.denoiser(
                initial_t, final_t, t_init_tensor, t_final_tensor, tj_cond=tj_cond
            )

            # Classifier-free guidance (stgl-style: same inputs, zero condition features)
            # Source (stgl) duplicates the batch with force_dropout=True, half_fd=True:
            #   first half keeps overlap features, second half zeros them out.
            # Here we achieve the same by running the denoiser twice:
            #   conditional (with tj_cond) vs unconditional (tj_cond=None → all zeros).
            # IMPORTANT: unconditional branch uses the SAME inputs & timesteps,
            #   including clean data at t=0 for conditioned sides. Only the
            #   condition features (overlap/inpaint tokens) are zeroed.
            if guidance_scale != 1.0 and (not denoise_init or not denoise_final):
                eps_init_uncond, eps_final_uncond = self.denoiser(
                    initial_t, final_t, t_init_tensor, t_final_tensor, tj_cond=None
                )

                # Apply guidance only to denoised sides
                if denoise_init:
                    eps_init_pred = eps_init_uncond + guidance_scale * (eps_init_pred - eps_init_uncond)
                if denoise_final:
                    eps_final_pred = eps_final_uncond + guidance_scale * (eps_final_pred - eps_final_uncond)

            # Denoise step
            if denoise_init:
                if use_ddim:
                    t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                    t_nxt = torch.full((B,), t_next.item(), dtype=torch.long, device=device)
                    initial_t = self.noise_schedule.ddim_step(initial_t, t_cur, t_nxt, eps_init_pred)
                else:
                    t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                    initial_t = self.noise_schedule.p_sample_ddpm(initial_t, t_cur, eps_init_pred)

            if denoise_final:
                if use_ddim:
                    t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                    t_nxt = torch.full((B,), t_next.item(), dtype=torch.long, device=device)
                    final_t = self.noise_schedule.ddim_step(final_t, t_cur, t_nxt, eps_final_pred)
                else:
                    t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                    final_t = self.noise_schedule.p_sample_ddpm(final_t, t_cur, eps_final_pred)

        return initial_t, final_t

    # ================================================================
    # Chain Planning with Constrained Denoising (CompDiffuser style)
    # ================================================================

    @torch.no_grad()
    def plan_chain(
        self,
        start_config: torch.Tensor,         # (22,) start state A
        goal_config: torch.Tensor,           # (22,) goal state D
        num_transitions: int = 3,            # K transitions in the chain
        num_samples: int = 8,                # parallel samples for selection
        mode: str = "parallel",              # "parallel" or "autoregressive"
        use_ddim: bool = True,
        ddim_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Chain planning: generate a sequence of transitions
        (A, c1) -> (c1, c2) -> ... -> (c_{K-1}, D)
        with equality constraints at connection points.

        Based on CompDiffuser's constrained denoising approach.

        Args:
            start_config: (22,) initial state
            goal_config: (22,) goal state
            num_transitions: K transitions in the chain
            num_samples: number of parallel candidate chains
            mode: "parallel" or "autoregressive"
            use_ddim: use DDIM sampling
            ddim_steps: DDIM steps

        Returns:
            best_chain: list of K+1 configs [A, c1, c2, ..., D]
            all_chains: (num_samples, K+1, 22) all candidate chains
        """
        device = start_config.device
        K = num_transitions
        B = num_samples

        # Expand start and goal for batch
        start = start_config.unsqueeze(0).expand(B, -1)  # (B, 22)
        goal = goal_config.unsqueeze(0).expand(B, -1)    # (B, 22)

        # Initialize K transitions: each is (initial, final) pair
        # transitions[k] = (initial_k, final_k), both (B, 22)
        initials = [torch.randn(B, self.config_dim, device=device) for _ in range(K)]
        finals = [torch.randn(B, self.config_dim, device=device) for _ in range(K)]

        # Hard constraints: first initial = start, last final = goal
        initials[0] = start.clone()
        finals[K - 1] = goal.clone()

        # Setup timestep schedule
        if use_ddim:
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, ddim_steps + 1, dtype=torch.long, device=device
            )
        else:
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1, device=device)

        if mode == "parallel":
            initials, finals = self._chain_parallel(
                initials, finals, start, goal, K, B, timesteps, use_ddim, guidance_scale, device
            )
        elif mode == "autoregressive":
            initials, finals = self._chain_autoregressive(
                initials, finals, start, goal, K, B, timesteps, use_ddim, guidance_scale, device
            )
        else:
            raise ValueError(f"Unknown chain mode: {mode}")

        # Build chains: [A, c1, c2, ..., D]
        # For each transition k, the "waypoint" is finals[k] (= initials[k+1])
        all_chains = torch.zeros(B, K + 1, self.config_dim, device=device)
        all_chains[:, 0, :] = initials[0]  # A
        for k in range(K - 1):
            # Average the overlap: finals[k] and initials[k+1]
            all_chains[:, k + 1, :] = (finals[k] + initials[k + 1]) / 2
        all_chains[:, K, :] = finals[K - 1]  # D

        # Select best chain based on overlap consistency
        overlap_errors = []
        for k in range(K - 1):
            err = torch.norm(finals[k] - initials[k + 1], dim=-1)  # (B,)
            overlap_errors.append(err)
        if overlap_errors:
            total_error = torch.stack(overlap_errors, dim=-1).sum(dim=-1)  # (B,)
            best_idx = total_error.argmin().item()
        else:
            best_idx = 0

        best_chain = [all_chains[best_idx, i, :] for i in range(K + 1)]

        return best_chain, all_chains

    def _chain_parallel(
        self, initials, finals, start, goal, K, B, timesteps, use_ddim, guidance_scale, device
    ):
        """
        Parallel constrained denoising for chain planning.
        All transitions denoised simultaneously, with constraint enforcement after each step.
        """
        for i in range(len(timesteps) - 1):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]

            # --- Step 1: Denoise all K transitions independently ---
            new_initials = []
            new_finals = []

            for k in range(K):
                # Determine which parts are fixed
                is_first = (k == 0)
                is_last = (k == K - 1)

                # Timesteps: fixed parts get t=0, free parts get t_current
                t_init_k = torch.zeros(B, dtype=torch.long, device=device) if is_first else \
                    torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                t_final_k = torch.zeros(B, dtype=torch.long, device=device) if is_last else \
                    torch.full((B,), t_current.item(), dtype=torch.long, device=device)

                # Build per-transition tj_cond (inpaint for fixed sides)
                zero_state = torch.zeros(B, self.config_dim, device=device)
                zero_t = torch.zeros(B, dtype=torch.long, device=device)
                tj_cond_k = {
                    'init_ovlp_state': zero_state,
                    'final_ovlp_state': zero_state,
                    'init_ovlp_t': zero_t,
                    'final_ovlp_t': zero_t,
                    'init_cd_use_ovlp': torch.zeros(B, dtype=torch.bool, device=device),
                    'final_cd_use_ovlp': torch.zeros(B, dtype=torch.bool, device=device),
                    'init_cd_use_inpat': torch.full((B,), is_first, dtype=torch.bool, device=device),
                    'final_cd_use_inpat': torch.full((B,), is_last, dtype=torch.bool, device=device),
                }

                # Predict noise
                eps_init_pred, eps_final_pred = self.denoiser(
                    initials[k], finals[k], t_init_k, t_final_k, tj_cond=tj_cond_k
                )

                # Denoise free parts
                if not is_first:
                    if use_ddim:
                        t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                        t_nxt = torch.full((B,), t_next.item(), dtype=torch.long, device=device)
                        new_init = self.noise_schedule.ddim_step(initials[k], t_cur, t_nxt, eps_init_pred)
                    else:
                        t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                        new_init = self.noise_schedule.p_sample_ddpm(initials[k], t_cur, eps_init_pred)
                else:
                    new_init = start.clone()

                if not is_last:
                    if use_ddim:
                        t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                        t_nxt = torch.full((B,), t_next.item(), dtype=torch.long, device=device)
                        new_final = self.noise_schedule.ddim_step(finals[k], t_cur, t_nxt, eps_final_pred)
                    else:
                        t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                        new_final = self.noise_schedule.p_sample_ddpm(finals[k], t_cur, eps_final_pred)
                else:
                    new_final = goal.clone()

                new_initials.append(new_init)
                new_finals.append(new_final)

            # --- Step 2: Enforce equality constraints at connection points ---
            # finals[k] should equal initials[k+1] for k = 0, ..., K-2
            for k in range(K - 1):
                avg = (new_finals[k] + new_initials[k + 1]) / 2.0
                new_finals[k] = avg
                new_initials[k + 1] = avg

            # Re-enforce hard constraints
            new_initials[0] = start.clone()
            new_finals[K - 1] = goal.clone()

            initials = new_initials
            finals = new_finals

        return initials, finals

    def _chain_autoregressive(
        self, initials, finals, start, goal, K, B, timesteps, use_ddim, guidance_scale, device
    ):
        """
        Autoregressive constrained denoising for chain planning.
        Within each denoising step, transitions are denoised sequentially,
        so each transition benefits from the already-denoised previous one.
        """
        for i in range(len(timesteps) - 1):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]

            for k in range(K):
                is_first = (k == 0)
                is_last = (k == K - 1)

                # Timesteps
                t_init_k = torch.zeros(B, dtype=torch.long, device=device) if is_first else \
                    torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                t_final_k = torch.zeros(B, dtype=torch.long, device=device) if is_last else \
                    torch.full((B,), t_current.item(), dtype=torch.long, device=device)

                # Build per-transition tj_cond (inpaint for fixed sides)
                zero_state = torch.zeros(B, self.config_dim, device=device)
                zero_t = torch.zeros(B, dtype=torch.long, device=device)
                tj_cond_k = {
                    'init_ovlp_state': zero_state,
                    'final_ovlp_state': zero_state,
                    'init_ovlp_t': zero_t,
                    'final_ovlp_t': zero_t,
                    'init_cd_use_ovlp': torch.zeros(B, dtype=torch.bool, device=device),
                    'final_cd_use_ovlp': torch.zeros(B, dtype=torch.bool, device=device),
                    'init_cd_use_inpat': torch.full((B,), is_first, dtype=torch.bool, device=device),
                    'final_cd_use_inpat': torch.full((B,), is_last, dtype=torch.bool, device=device),
                }

                # Predict noise
                eps_init_pred, eps_final_pred = self.denoiser(
                    initials[k], finals[k], t_init_k, t_final_k, tj_cond=tj_cond_k
                )

                # Denoise
                if not is_first:
                    if use_ddim:
                        t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                        t_nxt = torch.full((B,), t_next.item(), dtype=torch.long, device=device)
                        initials[k] = self.noise_schedule.ddim_step(initials[k], t_cur, t_nxt, eps_init_pred)
                    else:
                        t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                        initials[k] = self.noise_schedule.p_sample_ddpm(initials[k], t_cur, eps_init_pred)

                if not is_last:
                    if use_ddim:
                        t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                        t_nxt = torch.full((B,), t_next.item(), dtype=torch.long, device=device)
                        finals[k] = self.noise_schedule.ddim_step(finals[k], t_cur, t_nxt, eps_final_pred)
                    else:
                        t_cur = torch.full((B,), t_current.item(), dtype=torch.long, device=device)
                        finals[k] = self.noise_schedule.p_sample_ddpm(finals[k], t_cur, eps_final_pred)

                # Enforce constraint: finals[k] = initials[k+1] (for AR, propagate forward)
                if not is_last:
                    # Set next transition's initial to this transition's final
                    initials[k + 1] = finals[k].clone()

            # Re-enforce hard constraints
            initials[0] = start.clone()
            finals[K - 1] = goal.clone()

        return initials, finals

    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
