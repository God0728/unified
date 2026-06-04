"""Multimodal fusion backbone for Mini VLA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FusionConfig:
    """Configuration for the multimodal Transformer encoder."""

    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    use_act_token: bool = True


class FusionTransformer(nn.Module):
    """Fuse language, vision and state tokens with self-attention."""

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.act_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.type_embed = nn.Embedding(5, config.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.embed_dim)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.act_token, std=0.02)
        nn.init.normal_(self.type_embed.weight, std=0.02)

    def forward(
        self,
        text_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
        state_tokens: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full fused tokens and the pooled policy token.

        Type ids are assigned as follows: ``0=CLS``, ``1=text``, ``2=vision``,
        ``3=state`` and ``4=ACT``. The action head reads the ACT token when
        enabled; otherwise it reads the CLS token.
        """

        if text_tokens.ndim != 3 or vision_tokens.ndim != 3:
            raise ValueError("text_tokens and vision_tokens must be rank-3 tensors")
        batch = text_tokens.shape[0]
        if vision_tokens.shape[0] != batch:
            raise ValueError("text and vision batch sizes must match")

        device = text_tokens.device
        pieces = [self.cls_token.expand(batch, -1, -1), text_tokens, vision_tokens]
        type_ids = [
            torch.zeros(batch, 1, dtype=torch.long, device=device),
            torch.ones(batch, text_tokens.shape[1], dtype=torch.long, device=device),
            torch.full((batch, vision_tokens.shape[1]), 2, dtype=torch.long, device=device),
        ]

        if state_tokens is not None:
            if state_tokens.ndim != 3 or state_tokens.shape[0] != batch:
                raise ValueError("state_tokens must have shape [B,N,D]")
            pieces.append(state_tokens)
            type_ids.append(torch.full((batch, state_tokens.shape[1]), 3, dtype=torch.long, device=device))

        if self.config.use_act_token:
            pieces.append(self.act_token.expand(batch, -1, -1))
            type_ids.append(torch.full((batch, 1), 4, dtype=torch.long, device=device))

        tokens = torch.cat(pieces, dim=1)
        types = torch.cat(type_ids, dim=1)
        tokens = tokens + self.type_embed(types)

        if text_attention_mask is None:
            text_attention_mask = torch.ones(
                batch, text_tokens.shape[1], dtype=torch.bool, device=device
            )
        else:
            text_attention_mask = text_attention_mask.bool().to(device)

        valid_parts = [
            torch.ones(batch, 1, dtype=torch.bool, device=device),
            text_attention_mask,
            torch.ones(batch, vision_tokens.shape[1], dtype=torch.bool, device=device),
        ]
        if state_tokens is not None:
            valid_parts.append(torch.ones(batch, state_tokens.shape[1], dtype=torch.bool, device=device))
        if self.config.use_act_token:
            valid_parts.append(torch.ones(batch, 1, dtype=torch.bool, device=device))
        valid_mask = torch.cat(valid_parts, dim=1)

        fused = self.encoder(tokens, src_key_padding_mask=~valid_mask)
        fused = self.norm(fused)
        pooled_index = -1 if self.config.use_act_token else 0
        pooled = fused[:, pooled_index]
        return fused, pooled
