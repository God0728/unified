"""A tiny deterministic tokenizer for Mini VLA experiments.

The tokenizer is intentionally simple: it lowercases text, extracts word-like
pieces and maps them into a fixed-size vocabulary with a stable hash. This keeps
smoke tests and synthetic pretraining independent from external model downloads.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import torch

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[^\s]")


@dataclass
class SimpleTokenizerConfig:
    vocab_size: int = 4096
    max_length: int = 64
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3


class SimpleTokenizer:
    """Deterministic hash tokenizer with BPE-free behavior."""

    def __init__(self, config: SimpleTokenizerConfig):
        if config.vocab_size < 128:
            raise ValueError("vocab_size should be at least 128")
        self.config = config

    def _hash_token(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        return 4 + (value % (self.config.vocab_size - 4))

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [self.config.bos_token_id]
        tokens.extend(self._hash_token(tok) for tok in self.tokenize(text))
        tokens.append(self.config.eos_token_id)
        tokens = tokens[: self.config.max_length]
        attention = [1] * len(tokens)
        while len(tokens) < self.config.max_length:
            tokens.append(self.config.pad_token_id)
            attention.append(0)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(attention, dtype=torch.bool)

    def batch_encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(text) for text in texts]
        input_ids = torch.stack([x[0] for x in encoded], dim=0)
        masks = torch.stack([x[1] for x in encoded], dim=0)
        return input_ids, masks
