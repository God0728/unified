"""Action heads for Mini VLA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ActionHeadConfig:
    """Configuration for the continuous action head."""

    embed_dim: int = 256
    hidden_dim: int = 512
    action_dim: int = 7
    chunk_size: int = 8
    dropout: float = 0.1


class ContinuousActionHead(nn.Module):
    """Predict a continuous robot action chunk from a pooled token."""

    def __init__(self, config: ActionHeadConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.chunk_size * config.action_dim),
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2:
            raise ValueError(f"pooled must have shape [B,D], got {tuple(pooled.shape)}")
        out = self.net(pooled)
        return out.view(-1, self.config.chunk_size, self.config.action_dim)
