"""Vision encoders for Mini VLA.

The default encoder is intentionally small and dependency-light. It converts
single-view or multi-view RGB images into a sequence of visual tokens that can
be fused with language and robot-state tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class VisionConfig:
    """Configuration for the patch-based vision encoder."""

    image_size: int = 128
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 256
    max_views: int = 4
    dropout: float = 0.0

    @property
    def patches_per_view(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        side = self.image_size // self.patch_size
        return side * side


class PatchVisionEncoder(nn.Module):
    """A compact ViT-style patch embedding module.

    Parameters
    ----------
    config:
        Vision encoder configuration.

    Input shapes
    ------------
    - Single view: ``[batch, channels, height, width]``.
    - Multi view: ``[batch, views, channels, height, width]``.

    Output shape
    ------------
    ``[batch, views * patches_per_view, embed_dim]``.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.patches_per_view, config.embed_dim)
        )
        self.view_embed = nn.Parameter(torch.zeros(1, config.max_views, 1, config.embed_dim))
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.view_embed, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode="fan_out", nonlinearity="relu")
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(
                "images must have shape [B,C,H,W] or [B,V,C,H,W], "
                f"got {tuple(images.shape)}"
            )

        batch, views, channels, height, width = images.shape
        if views > self.config.max_views:
            raise ValueError(f"got {views} views, but max_views={self.config.max_views}")
        if channels != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} image channels, got {channels}"
            )
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                f"expected image size {self.config.image_size}x{self.config.image_size}, "
                f"got {height}x{width}"
            )

        flat = images.reshape(batch * views, channels, height, width)
        tokens = self.patch_embed(flat).flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.config.patches_per_view:
            raise RuntimeError("unexpected number of visual patches")

        tokens = tokens + self.pos_embed
        tokens = tokens.reshape(batch, views, self.config.patches_per_view, self.config.embed_dim)
        tokens = tokens + self.view_embed[:, :views]
        tokens = tokens.reshape(batch, views * self.config.patches_per_view, self.config.embed_dim)
        return self.dropout(self.norm(tokens))
