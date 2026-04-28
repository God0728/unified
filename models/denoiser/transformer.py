"""
Transformer-based Unified Transition Denoiser
===============================================
Uses AdaLN (Adaptive Layer Norm) for timestep + overlap conditioning.
Input: two tokens (initial, final config), each projected to hidden_dim.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .embeddings import TimestepEmbedder
from .conditioning import OverlapStateEncoder, AdaLayerNorm


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

        # Placeholder change_logits for interface compatibility with MLP denoiser
        change_logits = torch.zeros(B, 4, device=device)

        return eps_init_pred, eps_final_pred, change_logits
