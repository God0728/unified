"""Text encoder modules for Mini VLA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TextConfig:
    """Configuration for the simple instruction encoder."""

    vocab_size: int = 4096
    max_length: int = 64
    embed_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    pad_token_id: int = 0


class SimpleTextEncoder(nn.Module):
    """A compact Transformer encoder for instruction tokens.

    This class deliberately keeps tokenization outside the model. Dataset
    adapters provide ``input_ids`` and ``attention_mask`` so the same model can
    later consume tokens from a stronger tokenizer/backbone.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=config.pad_token_id,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_length, config.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        with torch.no_grad():
            self.token_embed.weight[self.config.pad_token_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,T], got {tuple(input_ids.shape)}")
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_length:
            raise ValueError(f"sequence length {seq_len} exceeds max_length={self.config.max_length}")

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        attention_mask = attention_mask.bool()

        x = self.token_embed(input_ids) + self.pos_embed[:, :seq_len]
        x = self.dropout(x)
        key_padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)
