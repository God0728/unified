#!/usr/bin/env python3
"""Fine-tune Mini VLA on robot datasets.

Examples:
    python scripts/finetune_mini_vla.py --config configs/mini_vla_finetune_lerobot.yaml
    python scripts/finetune_mini_vla.py --config configs/mini_vla_finetune_lerobot.yaml \
        --override data.name=lerobot/pusht train.pretrained_checkpoint=checkpoints/mini_vla_pretrain_debug/checkpoint_last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from mini_vla.data import MiniVLACollator, build_robot_dataset, build_tokenizer
from mini_vla.models import build_mini_vla_from_dict, count_parameters
from mini_vla.training.checkpointing import load_checkpoint
from mini_vla.training.config import load_config_with_overrides
from mini_vla.training.trainer import MiniVLATrainer
from mini_vla.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Mini VLA")
    parser.add_argument("--config", type=str, default="configs/mini_vla_finetune_lerobot.yaml")
    parser.add_argument("--override", action="append", default=None, help="Dotted config override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_with_overrides(args.config, args.override)
    seed_everything(int(config.get("seed", 0)))

    tokenizer = build_tokenizer(config)
    collator = MiniVLACollator(tokenizer)
    train_dataset = build_robot_dataset(config, split="train")
    val_dataset = build_robot_dataset(config, split="val") if config.get("data", {}).get("val") else None

    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg.get("eval_batch_size", batch_size)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collator,
            drop_last=False,
        )

    model = build_mini_vla_from_dict(config)
    pretrained = train_cfg.get("pretrained_checkpoint")
    if pretrained:
        payload = load_checkpoint(pretrained, model, optimizer=None, map_location="cpu")
        print(f"Loaded pretrained checkpoint from {pretrained} at step={payload.get('step')}")

    print(f"Mini VLA trainable parameters: {count_parameters(model):,}")
    requested_device = str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    trainer = MiniVLATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    summary = trainer.train()
    print(f"Fine-tuning finished. Summary: {summary}")
    print(f"Checkpoints saved under: {Path(train_cfg.get('output_dir', 'checkpoints/mini_vla_finetune')).resolve()}")


if __name__ == "__main__":
    main()
