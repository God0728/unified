"""Batch collation and dataset factories for Mini VLA."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .synthetic import SyntheticVLAConfig, SyntheticVLADataset
from .tokenizer import SimpleTokenizer, SimpleTokenizerConfig


class MiniVLACollator:
    """Collate dataset examples into the model's unified batch schema."""

    def __init__(self, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = torch.stack([ex["images"] for ex in examples], dim=0)
        states = torch.stack([ex["state"] for ex in examples], dim=0)
        actions = torch.stack([ex["actions"] for ex in examples], dim=0)
        instructions = [str(ex["instruction"]) for ex in examples]
        input_ids, attention_mask = self.tokenizer.batch_encode(instructions)
        return {
            "images": images,
            "state": states,
            "actions": actions,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "instructions": instructions,
            "metadata": [ex.get("metadata", {}) for ex in examples],
        }


def build_tokenizer(config: dict[str, Any]) -> SimpleTokenizer:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    tokenizer_cfg = data_cfg.get("tokenizer", {})
    return SimpleTokenizer(
        SimpleTokenizerConfig(
            vocab_size=int(model_cfg.get("vocab_size", tokenizer_cfg.get("vocab_size", 4096))),
            max_length=int(model_cfg.get("max_text_len", tokenizer_cfg.get("max_length", 64))),
        )
    )


def build_dataset(config: dict[str, Any], split: str = "train") -> Dataset:
    data_cfg = config.get("data", {})
    dataset_type = str(data_cfg.get("type", "synthetic")).lower()
    if dataset_type != "synthetic":
        raise ValueError(
            f"Unsupported pretrain dataset type '{dataset_type}'. "
            "For real robot data use the finetune adapters."
        )

    model_cfg = config.get("model", {})
    split_cfg = data_cfg.get(split, {})
    num_samples = int(split_cfg.get("num_samples", data_cfg.get("num_samples", 10000)))
    seed = int(split_cfg.get("seed", data_cfg.get("seed", 0)))
    return SyntheticVLADataset(
        SyntheticVLAConfig(
            num_samples=num_samples,
            image_size=int(model_cfg.get("image_size", data_cfg.get("image_size", 128))),
            action_dim=int(model_cfg.get("action_dim", 7)),
            chunk_size=int(model_cfg.get("chunk_size", 8)),
            state_dim=int(model_cfg.get("state_dim", data_cfg.get("state_dim", 8))),
            seed=seed,
            noise_std=float(data_cfg.get("noise_std", 0.01)),
        )
    )
