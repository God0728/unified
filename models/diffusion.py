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
    # Helpers: CFG prediction + denoising step
    # ================================================================

    def _ddim_timesteps(self, ddim_steps: int, device: torch.device) -> torch.Tensor:
        """DDIM timestep schedule following stgl's ddim_set_timesteps."""
        step_ratio = self.num_timesteps // ddim_steps
        ts = (np.arange(0, ddim_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        return torch.from_numpy(ts).long().to(device)

    def _cfg_predict(
        self,
        initial: torch.Tensor,
        final: torch.Tensor,
        t_init: torch.Tensor,
        t_final: torch.Tensor,
        tj_cond: dict,
        guidance_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict noise with classifier-free guidance, following stgl's pattern.

        stgl duplicates the batch: 1st half = conditional, 2nd half = force_dropout.
        Here we use two forward passes (conditional + unconditional) then combine:
            eps = eps_uncond + w * (eps_cond - eps_uncond)
        """
        eps_init_cond, eps_final_cond = self.denoiser(
            initial, final, t_init, t_final, tj_cond=tj_cond
        )
        if guidance_scale == 1.0:
            return eps_init_cond, eps_final_cond

        # Unconditional: same inputs & timesteps, all condition features zeroed
        eps_init_uncond, eps_final_uncond = self.denoiser(
            initial, final, t_init, t_final, tj_cond=None
        )
        eps_init = eps_init_uncond + guidance_scale * (eps_init_cond - eps_init_uncond)
        eps_final = eps_final_uncond + guidance_scale * (eps_final_cond - eps_final_uncond)
        return eps_init, eps_final

    def _step(
        self,
        x_t: torch.Tensor,
        t_current: int,
        t_next: int,
        eps_pred: torch.Tensor,
        use_ddim: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """One denoising step: DDIM or DDPM."""
        B = x_t.shape[0]
        t_cur = torch.full((B,), t_current, dtype=torch.long, device=device)
        if use_ddim:
            t_nxt = torch.full((B,), t_next, dtype=torch.long, device=device)
            return self.noise_schedule.ddim_step(x_t, t_cur, t_nxt, eps_pred)
        else:
            return self.noise_schedule.p_sample_ddpm(x_t, t_cur, eps_pred)

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

        # Setup timestep schedule (following stgl)
        if use_ddim:
            time_schedule = self._ddim_timesteps(ddim_steps, device)
            step_size = self.num_timesteps // ddim_steps
        else:
            time_schedule = torch.arange(self.num_timesteps - 1, -1, -1, device=device)
            step_size = 1

        for idx in range(len(time_schedule)):
            t_current = time_schedule[idx].item()
            t_next = t_current - step_size

            t_init_tensor = (torch.full((B,), t_current, dtype=torch.long, device=device)
                             if denoise_init
                             else torch.zeros(B, dtype=torch.long, device=device))
            t_final_tensor = (torch.full((B,), t_current, dtype=torch.long, device=device)
                              if denoise_final
                              else torch.zeros(B, dtype=torch.long, device=device))

            # Conditioned sides → inpaint token; denoised sides → no condition
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

            # Predict noise with CFG (stgl: cond vs uncond → combine)
            eps_init_pred, eps_final_pred = self._cfg_predict(
                initial_t, final_t, t_init_tensor, t_final_tensor,
                tj_cond, guidance_scale,
            )

            # Denoise step
            if denoise_init:
                initial_t = self._step(initial_t, t_current, t_next, eps_init_pred, use_ddim, device)
            if denoise_final:
                final_t = self._step(final_t, t_current, t_next, eps_final_pred, use_ddim, device)

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

        # Setup timestep schedule (following stgl)
        if use_ddim:
            time_schedule = self._ddim_timesteps(ddim_steps, device)
            step_size = self.num_timesteps // ddim_steps
        else:
            time_schedule = torch.arange(self.num_timesteps - 1, -1, -1, device=device)
            step_size = 1

        if mode == "parallel":
            initials, finals = self._chain_parallel(
                initials, finals, start, goal, K, B,
                time_schedule, step_size, use_ddim, guidance_scale, device
            )
        elif mode == "autoregressive":
            initials, finals = self._chain_autoregressive(
                initials, finals, start, goal, K, B,
                time_schedule, step_size, use_ddim, guidance_scale, device
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
        self, initials, finals, start, goal, K, B,
        time_schedule, step_size, use_ddim, guidance_scale, device
    ):
        """
        Parallel constrained denoising (stgl comp_pred_p_loop_n_same_t style).

        At each denoising step:
          1. Build overlap conditioning from OLD (same noise level) neighbors
          2. Denoise all K transitions independently (with CFG)
          3. Store denoised results in a separate list
          4. Average connection points for constraint enforcement (GSC style)
          5. Update
        """
        for idx in range(len(time_schedule)):
            t_current = time_schedule[idx].item()
            t_next = t_current - step_size

            t_cur_tensor = torch.full((B,), t_current, dtype=torch.long, device=device)

            # Store denoised results separately (stgl same_t pattern)
            new_initials = [None] * K
            new_finals = [None] * K

            for k in range(K):
                is_first = (k == 0)
                is_last = (k == K - 1)

                # --- Build tj_cond with overlap from OLD neighbors ---
                # Init side
                if is_first:
                    # Inpaint start: clean data at t=0
                    cur_initial = start.clone()
                    t_init_k = torch.zeros(B, dtype=torch.long, device=device)
                    init_cd_use_inpat = True
                    init_cd_use_ovlp = False
                    init_ovlp_state = torch.zeros(B, self.config_dim, device=device)
                    init_ovlp_t = torch.zeros(B, dtype=torch.long, device=device)
                else:
                    # Overlap from previous transition's final (same noise level)
                    cur_initial = initials[k]
                    t_init_k = t_cur_tensor.clone()
                    init_cd_use_inpat = False
                    init_cd_use_ovlp = True
                    init_ovlp_state = finals[k - 1].clone()   # still at t_current
                    init_ovlp_t = t_cur_tensor.clone()

                # Final side
                if is_last:
                    # Inpaint goal: clean data at t=0
                    cur_final = goal.clone()
                    t_final_k = torch.zeros(B, dtype=torch.long, device=device)
                    final_cd_use_inpat = True
                    final_cd_use_ovlp = False
                    final_ovlp_state = torch.zeros(B, self.config_dim, device=device)
                    final_ovlp_t = torch.zeros(B, dtype=torch.long, device=device)
                else:
                    # Overlap from next transition's initial (same noise level)
                    cur_final = finals[k]
                    t_final_k = t_cur_tensor.clone()
                    final_cd_use_inpat = False
                    final_cd_use_ovlp = True
                    final_ovlp_state = initials[k + 1].clone()  # still at t_current
                    final_ovlp_t = t_cur_tensor.clone()

                tj_cond_k = {
                    'init_ovlp_state': init_ovlp_state,
                    'final_ovlp_state': final_ovlp_state,
                    'init_ovlp_t': init_ovlp_t,
                    'final_ovlp_t': final_ovlp_t,
                    'init_cd_use_ovlp': torch.full((B,), init_cd_use_ovlp, dtype=torch.bool, device=device),
                    'final_cd_use_ovlp': torch.full((B,), final_cd_use_ovlp, dtype=torch.bool, device=device),
                    'init_cd_use_inpat': torch.full((B,), init_cd_use_inpat, dtype=torch.bool, device=device),
                    'final_cd_use_inpat': torch.full((B,), final_cd_use_inpat, dtype=torch.bool, device=device),
                }

                # --- Predict noise with CFG ---
                eps_init_pred, eps_final_pred = self._cfg_predict(
                    cur_initial, cur_final, t_init_k, t_final_k,
                    tj_cond_k, guidance_scale,
                )

                # --- Denoise free parts ---
                if is_first:
                    new_initials[k] = start.clone()
                else:
                    new_initials[k] = self._step(
                        cur_initial, t_current, t_next, eps_init_pred, use_ddim, device
                    )

                if is_last:
                    new_finals[k] = goal.clone()
                else:
                    new_finals[k] = self._step(
                        cur_final, t_current, t_next, eps_final_pred, use_ddim, device
                    )

            # --- Enforce equality constraints: average connection points (GSC) ---
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
        self, initials, finals, start, goal, K, B,
        time_schedule, step_size, use_ddim, guidance_scale, device
    ):
        """
        Autoregressive constrained denoising (stgl comp_pred_p_loop_n style).

        Within each denoising step, transitions are processed sequentially:
          - Already-denoised transition k provides overlap conditioning to transition k+1.
          - init_ovlp_state = finals[k-1] (just denoised, ~t_next level)
          - final_ovlp_state = initials[k+1] (not yet denoised, still t_current)

        The overlap conditioning guides the denoiser — the actual data (initials[k],
        finals[k]) stays at its own noise level and is NOT replaced by the neighbor.
        """
        for idx in range(len(time_schedule)):
            t_current = time_schedule[idx].item()
            t_next = t_current - step_size

            t_cur_tensor = torch.full((B,), t_current, dtype=torch.long, device=device)

            for k in range(K):
                is_first = (k == 0)
                is_last = (k == K - 1)

                # --- Build tj_cond with overlap from neighbors ---

                # Init side
                if is_first:
                    # Inpaint start: clean data at t=0
                    cur_initial = start.clone()
                    t_init_k = torch.zeros(B, dtype=torch.long, device=device)
                    init_cd_use_inpat = True
                    init_cd_use_ovlp = False
                    init_ovlp_state = torch.zeros(B, self.config_dim, device=device)
                    init_ovlp_t = torch.zeros(B, dtype=torch.long, device=device)
                else:
                    # Overlap from previous transition's final (already denoised in
                    # this iteration → approximately at t_next noise level).
                    # Following stgl: report ovlp_t = t_current - 1 (approximate).
                    cur_initial = initials[k]
                    t_init_k = t_cur_tensor.clone()
                    init_cd_use_inpat = False
                    init_cd_use_ovlp = True
                    init_ovlp_state = finals[k - 1].clone()   # already denoised
                    init_ovlp_t = torch.clamp(t_cur_tensor - 1, min=0)

                # Final side
                if is_last:
                    # Inpaint goal: clean data at t=0
                    cur_final = goal.clone()
                    t_final_k = torch.zeros(B, dtype=torch.long, device=device)
                    final_cd_use_inpat = True
                    final_cd_use_ovlp = False
                    final_ovlp_state = torch.zeros(B, self.config_dim, device=device)
                    final_ovlp_t = torch.zeros(B, dtype=torch.long, device=device)
                else:
                    # Overlap from next transition's initial (not yet denoised
                    # in this iteration → still at t_current noise level).
                    cur_final = finals[k]
                    t_final_k = t_cur_tensor.clone()
                    final_cd_use_inpat = False
                    final_cd_use_ovlp = True
                    final_ovlp_state = initials[k + 1].clone()  # still noisy
                    final_ovlp_t = t_cur_tensor.clone()

                tj_cond_k = {
                    'init_ovlp_state': init_ovlp_state,
                    'final_ovlp_state': final_ovlp_state,
                    'init_ovlp_t': init_ovlp_t,
                    'final_ovlp_t': final_ovlp_t,
                    'init_cd_use_ovlp': torch.full((B,), init_cd_use_ovlp, dtype=torch.bool, device=device),
                    'final_cd_use_ovlp': torch.full((B,), final_cd_use_ovlp, dtype=torch.bool, device=device),
                    'init_cd_use_inpat': torch.full((B,), init_cd_use_inpat, dtype=torch.bool, device=device),
                    'final_cd_use_inpat': torch.full((B,), final_cd_use_inpat, dtype=torch.bool, device=device),
                }

                # --- Predict noise with CFG ---
                eps_init_pred, eps_final_pred = self._cfg_predict(
                    cur_initial, cur_final, t_init_k, t_final_k,
                    tj_cond_k, guidance_scale,
                )

                # --- Denoise and update in-place (AR pattern) ---
                if not is_first:
                    initials[k] = self._step(
                        cur_initial, t_current, t_next, eps_init_pred, use_ddim, device
                    )

                if not is_last:
                    finals[k] = self._step(
                        cur_final, t_current, t_next, eps_final_pred, use_ddim, device
                    )

                # AR propagation: next transition's initial gets this final
                # as overlap conditioning (NOT data replacement — the overlap
                # is used via init_ovlp_state in the next iteration of k).

            # Re-enforce hard constraints
            initials[0] = start.clone()
            finals[K - 1] = goal.clone()

        return initials, finals

    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
