"""
Conditioning Modules
=====================
Overlap state encoder (stgl-style) and Adaptive Layer Norm (AdaLN).
"""

import torch
import torch.nn as nn

from .embeddings import TimestepEmbedder


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
