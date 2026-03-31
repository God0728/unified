"""
Unified Transition Denoiser Network
====================================
Based on UWM (arXiv:2504.02792):
  - Takes noisy initial and final configs as input
  - Conditioned on TWO independent diffusion timesteps (t_init, t_final)
  - Predicts noise for both initial and final simultaneously

Architecture: Transformer with AdaLN (Adaptive Layer Norm) for timestep conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ============================================================
# Timestep Embedding
# ============================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimestepEmbedder(nn.Module):
    """Embeds a single diffusion timestep into a vector."""

    def __init__(self, time_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# ============================================================
# Overlap State Encoder (stgl-style conditioning)
# ============================================================

class OverlapStateEncoder(nn.Module):
    """
    Encode an overlap conditioning state + its timestep into a feature vector.
    Analogous to stgl's Traj_Time_Encoder but for single states (B, config_dim).
    """

    def __init__(self, config_dim: int, time_embed_dim: int, feat_dim: int):
        super().__init__()
        self.state_proj = nn.Linear(config_dim, feat_dim)
        self.time_emb = TimestepEmbedder(time_embed_dim, feat_dim)
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, config_dim) overlap state (q_sampled from clean)
            t: (B,) timestep used for q_sample
        Returns:
            feat: (B, feat_dim)
        """
        s_feat = self.state_proj(state)   # (B, feat_dim)
        t_feat = self.time_emb(t)          # (B, feat_dim)
        return self.fuse(torch.cat([s_feat, t_feat], dim=-1))  # (B, feat_dim)


# ============================================================
# Adaptive Layer Norm (AdaLN) - conditioned on combined timestep embedding
# ============================================================

class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim)
        scale_shift = self.linear(cond)  # (B, 2*hidden_dim)
        scale, shift = scale_shift.chunk(2, dim=-1)  # each (B, hidden_dim)

        if x.dim() == 3:
            # x: (B, L, hidden_dim)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        return self.norm(x) * (1 + scale) + shift


# ============================================================
# Transformer Block with AdaLN
# ============================================================

class TransformerBlock(nn.Module):
    """Transformer block with AdaLN conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # AdaLN for both sub-layers
        self.adaln1 = AdaLayerNorm(hidden_dim, cond_dim)
        self.adaln2 = AdaLayerNorm(hidden_dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Self-attention with AdaLN
        x_norm = self.adaln1(x, cond)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feed-forward with AdaLN
        x_norm = self.adaln2(x, cond)
        x = x + self.ff(x_norm)

        return x


# ============================================================
# Unified Transition Denoiser (Transformer)
# ============================================================

class UnifiedTransitionDenoiser(nn.Module):
    """
    Unified denoiser for transition (initial, final) pairs.

    Following UWM, uses independent timesteps for initial and final:
      s_θ(initial_{t_init}, final_{t_final}, t_init, t_final)

    The model predicts noise for both initial and final simultaneously.

    Input tokens:
      - Token 0: initial config (noisy)
      - Token 1: final config (noisy)

    Conditioning:
      - t_init: diffusion timestep for initial
      - t_final: diffusion timestep for final
      - Combined via AdaLN
    """

    def __init__(
        self,
        config_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        time_embed_dim: int = 128,
        dropout: float = 0.1,
        ovlp_feat_dim: int = 64,
    ):
        super().__init__()
        self.config_dim = config_dim
        self.hidden_dim = hidden_dim
        self.ovlp_feat_dim = ovlp_feat_dim

        # --- Timestep embedders (independent for init and final) ---
        self.time_embed_init = TimestepEmbedder(time_embed_dim, hidden_dim)
        self.time_embed_final = TimestepEmbedder(time_embed_dim, hidden_dim)

        # --- Overlap state encoders (stgl-style) ---
        self.init_ovlp_encoder = OverlapStateEncoder(config_dim, time_embed_dim, ovlp_feat_dim)
        self.final_ovlp_encoder = OverlapStateEncoder(config_dim, time_embed_dim, ovlp_feat_dim)

        # --- Inpaint tokens (same dim as ovlp_feat for mutually exclusive slot) ---
        self.init_inpat_token = nn.Parameter(torch.randn(1, ovlp_feat_dim))
        self.final_inpat_token = nn.Parameter(torch.randn(1, ovlp_feat_dim))

        # Combined conditioning: [t_init_emb, t_final_emb, init_side_feat, final_side_feat]
        # Per-side feature occupies a single slot: ovlp_feat / inpat_token / zeros (mutually exclusive)
        cond_input_dim = hidden_dim * 2 + ovlp_feat_dim * 2
        self.cond_dim = hidden_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Input projection ---
        # Each config (22D) is projected to hidden_dim
        self.input_proj_init = nn.Linear(config_dim, hidden_dim)
        self.input_proj_final = nn.Linear(config_dim, hidden_dim)

        # Learnable token type embeddings (initial vs final)
        self.token_type_embed = nn.Embedding(2, hidden_dim)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, self.cond_dim, dropout)
            for _ in range(num_layers)
        ])

        # --- Output projection ---
        # Project back to config_dim for each token
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj_init = nn.Linear(hidden_dim, config_dim)
        self.output_proj_final = nn.Linear(hidden_dim, config_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        # Zero-init output projections for stable training
        nn.init.zeros_(self.output_proj_init.weight)
        nn.init.zeros_(self.output_proj_init.bias)
        nn.init.zeros_(self.output_proj_final.weight)
        nn.init.zeros_(self.output_proj_final.bias)

    def forward(
        self,
        initial_noisy: torch.Tensor,   # (B, D) noisy initial config
        final_noisy: torch.Tensor,      # (B, D) noisy final config
        t_init: torch.Tensor,           # (B,) timestep for initial
        t_final: torch.Tensor,          # (B,) timestep for final
        tj_cond: Optional[Dict] = None, # stgl-style overlap/inpaint conditioning
    ) -> tuple:
        """
        Forward pass.

        Args:
            tj_cond: dict with keys:
                - init_ovlp_state: (B, D) q_sampled overlap state for init side
                - final_ovlp_state: (B, D) q_sampled overlap state for final side
                - init_ovlp_t / final_ovlp_t: (B,) timestep for overlap q_sample
                - init_cd_use_ovlp / final_cd_use_ovlp: (B,) bool mask
                - init_cd_use_inpat / final_cd_use_inpat: (B,) bool mask

        Returns:
            eps_init_pred: (B, D) predicted noise for initial
            eps_final_pred: (B, D) predicted noise for final
        """
        B = initial_noisy.shape[0]
        device = initial_noisy.device

        # --- Timestep embeddings ---
        t_init_emb = self.time_embed_init(t_init)    # (B, hidden_dim)
        t_final_emb = self.time_embed_final(t_final)  # (B, hidden_dim)

        # --- Per-side conditioning (mutually exclusive: ovlp / inpat / drop) ---
        # Each side occupies ONE feature slot. Exactly one of {ovlp, inpat, drop} is active.
        if tj_cond is not None:
            init_ovlp_raw = self.init_ovlp_encoder(
                tj_cond['init_ovlp_state'], tj_cond['init_ovlp_t']
            )
            final_ovlp_raw = self.final_ovlp_encoder(
                tj_cond['final_ovlp_state'], tj_cond['final_ovlp_t']
            )
            init_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)
            final_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)

            m = tj_cond['init_cd_use_ovlp']
            if m.any():
                init_side_feat[m] = init_ovlp_raw[m]
            m = tj_cond['init_cd_use_inpat']
            if m.any():
                init_side_feat[m] = self.init_inpat_token.expand(B, -1)[m]

            m = tj_cond['final_cd_use_ovlp']
            if m.any():
                final_side_feat[m] = final_ovlp_raw[m]
            m = tj_cond['final_cd_use_inpat']
            if m.any():
                final_side_feat[m] = self.final_inpat_token.expand(B, -1)[m]
        else:
            init_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)
            final_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)

        # Combined conditioning
        cond = self.cond_proj(torch.cat([
            t_init_emb, t_final_emb,
            init_side_feat, final_side_feat,
        ], dim=-1))  # (B, hidden_dim)

        # --- Input tokens ---
        tok_init = self.input_proj_init(initial_noisy)   # (B, hidden_dim)
        tok_final = self.input_proj_final(final_noisy)    # (B, hidden_dim)

        # Add token type embeddings
        tok_init = tok_init + self.token_type_embed(
            torch.zeros(B, dtype=torch.long, device=initial_noisy.device)
        )
        tok_final = tok_final + self.token_type_embed(
            torch.ones(B, dtype=torch.long, device=initial_noisy.device)
        )

        # Stack into sequence: (B, 2, hidden_dim)
        x = torch.stack([tok_init, tok_final], dim=1)

        # --- Transformer blocks ---
        for block in self.blocks:
            x = block(x, cond)

        # --- Output ---
        x = self.output_norm(x)
        eps_init_pred = self.output_proj_init(x[:, 0, :])   # (B, 22)
        eps_final_pred = self.output_proj_final(x[:, 1, :])  # (B, 22)

        return eps_init_pred, eps_final_pred


# ============================================================
# MLP-based Unified Denoiser (simpler alternative)
# ============================================================

class UnifiedTransitionDenoiserMLP(nn.Module):
    """
    MLP-based unified denoiser with per-layer condition injection.

    Following stgl's architecture: at each hidden layer, the global condition
    vector is projected and ADDED to the hidden features. This is analogous to
    ResidualTemporalBlock_dd's `out = block(x) + time_mlp(cond)` pattern, where
    the condition influences every layer of the network — not just the input.

    Per-side conditioning is mutually exclusive: for each side (init/final),
    exactly one of {overlap, inpaint, drop} is active, occupying a single
    feature slot of size ovlp_feat_dim.

    Structure:
      1. Timestep embedding: t_init, t_final → emb vectors
      2. Per-side conditioning: mutually exclusive ovlp/inpat/drop → side_feat
      3. Global condition: [t_init_emb, t_final_emb, init_side_feat, final_side_feat]
      4. MLP input: [initial_noisy, final_noisy] (data only)
      5. Per-layer: h = act(linear(h) + cond_proj(cond))   ← condition injected every layer
      6. Two output heads: separate Linear for eps_init and eps_final
    """

    def __init__(
        self,
        config_dim: int,
        hidden_dims: list = [512, 512, 512, 512],
        time_embed_dim: int = 128,
        dropout: float = 0.1,
        act_fn: str = 'silu',       # 'relu' / 'prelu' / 'silu'
        use_dropout: bool = True,
        ovlp_feat_dim: int = 64,
    ):
        super().__init__()
        self.config_dim = config_dim
        self.ovlp_feat_dim = ovlp_feat_dim

        # ---- Timestep embedders (independent for init and final) ----
        time_hidden = hidden_dims[0]
        self.time_embed_init = TimestepEmbedder(time_embed_dim, time_hidden)
        self.time_embed_final = TimestepEmbedder(time_embed_dim, time_hidden)

        # ---- Overlap state encoders (stgl-style) ----
        self.init_ovlp_encoder = OverlapStateEncoder(config_dim, time_embed_dim, ovlp_feat_dim)
        self.final_ovlp_encoder = OverlapStateEncoder(config_dim, time_embed_dim, ovlp_feat_dim)

        # ---- Inpaint tokens (same dim as ovlp_feat for mutually exclusive slot) ----
        self.init_inpat_token = nn.Parameter(torch.randn(1, ovlp_feat_dim))
        self.final_inpat_token = nn.Parameter(torch.randn(1, ovlp_feat_dim))

        # ---- Condition dim: [t_init_emb, t_final_emb, init_side_feat, final_side_feat] ----
        cond_dim = time_hidden * 2 + ovlp_feat_dim * 2

        # ---- MLP input: just [initial_noisy, final_noisy] ----
        input_dim = config_dim * 2

        # ---- Build per-layer MLP with condition injection ----
        act_cls = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'silu': nn.SiLU}[act_fn]
        layer_dims = [input_dim] + hidden_dims
        num_layers = len(layer_dims) - 1

        self.linears = nn.ModuleList()
        self.cond_projs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            self.linears.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            # Per-layer condition projection (analogous to ResBlock's time_mlp)
            self.cond_projs.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, layer_dims[i + 1]),
            ))
            if i < num_layers - 1:
                self.acts.append(act_cls())
            else:
                self.acts.append(nn.Identity())  # no activation on last hidden layer
            if use_dropout and i < num_layers - 2:
                self.dropouts.append(nn.Dropout(p=dropout))
            else:
                self.dropouts.append(nn.Identity())

        # ---- Two output heads ----
        self.output_init = nn.Linear(hidden_dims[-1], config_dim)
        self.output_final = nn.Linear(hidden_dims[-1], config_dim)

    def forward(
        self,
        initial_noisy: torch.Tensor,   # (B, D)
        final_noisy: torch.Tensor,      # (B, D)
        t_init: torch.Tensor,           # (B,)
        t_final: torch.Tensor,          # (B,)
        tj_cond: Optional[Dict] = None, # stgl-style overlap/inpaint conditioning
    ) -> tuple:
        """
        Per-layer conditioned forward pass.

        Returns:
            eps_init_pred (B, D): predicted noise for initial
            eps_final_pred (B, D): predicted noise for final
        """
        B = initial_noisy.shape[0]
        device = initial_noisy.device

        # Timestep embeddings
        t_init_emb = self.time_embed_init(t_init)
        t_final_emb = self.time_embed_final(t_final)

        # Per-side conditioning (mutually exclusive: ovlp / inpat / drop)
        if tj_cond is not None:
            init_ovlp_raw = self.init_ovlp_encoder(
                tj_cond['init_ovlp_state'], tj_cond['init_ovlp_t']
            )
            final_ovlp_raw = self.final_ovlp_encoder(
                tj_cond['final_ovlp_state'], tj_cond['final_ovlp_t']
            )
            init_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)
            final_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)

            m = tj_cond['init_cd_use_ovlp']
            if m.any():
                init_side_feat[m] = init_ovlp_raw[m]
            m = tj_cond['init_cd_use_inpat']
            if m.any():
                init_side_feat[m] = self.init_inpat_token.expand(B, -1)[m]

            m = tj_cond['final_cd_use_ovlp']
            if m.any():
                final_side_feat[m] = final_ovlp_raw[m]
            m = tj_cond['final_cd_use_inpat']
            if m.any():
                final_side_feat[m] = self.final_inpat_token.expand(B, -1)[m]
        else:
            init_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)
            final_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)

        # Global condition vector
        cond = torch.cat([t_init_emb, t_final_emb, init_side_feat, final_side_feat], dim=-1)

        # MLP input: just the noisy data
        h = torch.cat([initial_noisy, final_noisy], dim=-1)

        # Per-layer forward with condition injection at every layer
        for linear, cond_proj, act, dp in zip(
            self.linears, self.cond_projs, self.acts, self.dropouts
        ):
            h = dp(act(linear(h) + cond_proj(cond)))

        # Predict noise for both parts
        eps_init_pred = self.output_init(h)    # (B, D)
        eps_final_pred = self.output_final(h)  # (B, D)

        return eps_init_pred, eps_final_pred


def build_denoiser(config: dict) -> nn.Module:
    """Build denoiser from config dict."""
    model_cfg = config['model']
    data_cfg = config['data']

    ovlp_feat_dim = model_cfg.get('ovlp_feat_dim', 64)

    if model_cfg['architecture'] == 'transformer':
        return UnifiedTransitionDenoiser(
            config_dim=data_cfg['config_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            num_heads=model_cfg['num_heads'],
            num_layers=model_cfg['num_layers'],
            time_embed_dim=model_cfg['time_embed_dim'],
            dropout=model_cfg['dropout'],
            ovlp_feat_dim=ovlp_feat_dim,
        )
    elif model_cfg['architecture'] == 'mlp':
        return UnifiedTransitionDenoiserMLP(
            config_dim=data_cfg['config_dim'],
            hidden_dims=model_cfg['mlp_hidden_dims'],
            time_embed_dim=model_cfg['time_embed_dim'],
            dropout=model_cfg.get('dropout', 0.1),
            act_fn=model_cfg.get('act_fn', 'silu'),
            use_dropout=model_cfg.get('use_dropout', True),
            ovlp_feat_dim=ovlp_feat_dim,
        )
    else:
        raise ValueError(f"Unknown architecture: {model_cfg['architecture']}")
