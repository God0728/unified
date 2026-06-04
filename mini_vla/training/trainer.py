"""Training loop for Mini VLA."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .checkpointing import save_checkpoint
from .config import save_config


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


class MiniVLATrainer:
    """A compact trainer for single-GPU/CPU Mini VLA experiments."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        train_cfg = config.get("train", {})
        self.output_dir = Path(train_cfg.get("output_dir", "checkpoints/mini_vla_pretrain"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.log_every = int(train_cfg.get("log_every", 10))
        self.save_every = int(train_cfg.get("save_every", 500))
        self.eval_every = int(train_cfg.get("eval_every", 500))
        self.steps = int(train_cfg.get("steps", 1000))
        self.use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
        self.dtype = torch.bfloat16 if str(train_cfg.get("amp_dtype", "bf16")) == "bf16" else torch.float16
        self.optimizer = self._build_optimizer(train_cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.dtype == torch.float16)
        save_config(config, self.output_dir / "config_resolved.yaml")

    def _build_optimizer(self, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
        lr = float(train_cfg.get("lr", 3e-4))
        weight_decay = float(train_cfg.get("weight_decay", 0.01))
        betas = tuple(train_cfg.get("betas", [0.9, 0.95]))
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    def train(self) -> dict[str, Any]:
        self.model.train()
        step = 0
        epoch = 0
        best_val = math.inf
        running_loss = 0.0
        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        while step < self.steps:
            epoch += 1
            for batch_idx, batch in enumerate(self.train_loader):
                batch = move_batch_to_device(batch, self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                    out = self.model(
                        images=batch["images"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        state=batch.get("state"),
                        actions=batch["actions"],
                    )
                    loss = out["loss"] / self.grad_accum_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    step += 1
                    running_loss += float(loss.detach().cpu()) * self.grad_accum_steps

                    if step % self.log_every == 0:
                        elapsed = max(time.time() - start_time, 1e-6)
                        avg = running_loss / self.log_every
                        print(
                            f"step={step:06d} epoch={epoch:03d} "
                            f"train_loss={avg:.6f} steps_per_sec={step / elapsed:.3f}",
                            flush=True,
                        )
                        running_loss = 0.0

                    if self.val_loader is not None and step % self.eval_every == 0:
                        val_loss = self.evaluate(max_batches=int(self.config.get("eval", {}).get("max_batches", 50)))
                        print(f"step={step:06d} val_loss={val_loss:.6f}", flush=True)
                        if val_loss < best_val:
                            best_val = val_loss
                            save_checkpoint(
                                self.output_dir / "checkpoint_best.pt",
                                self.model,
                                self.optimizer,
                                step,
                                epoch,
                                self.config,
                                extra={"val_loss": val_loss},
                            )

                    if step % self.save_every == 0:
                        save_checkpoint(
                            self.output_dir / f"checkpoint_step_{step}.pt",
                            self.model,
                            self.optimizer,
                            step,
                            epoch,
                            self.config,
                        )

                    if step >= self.steps:
                        break

        save_checkpoint(
            self.output_dir / "checkpoint_last.pt",
            self.model,
            self.optimizer,
            step,
            epoch,
            self.config,
            extra={"best_val_loss": best_val},
        )
        return {"step": step, "epoch": epoch, "best_val_loss": best_val}

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> float:
        if self.val_loader is None:
            return math.nan
        self.model.eval()
        losses: list[float] = []
        for idx, batch in enumerate(self.val_loader):
            if idx >= max_batches:
                break
            batch = move_batch_to_device(batch, self.device)
            out = self.model(
                images=batch["images"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                state=batch.get("state"),
                actions=batch["actions"],
            )
            losses.append(float(out["loss"].detach().cpu()))
        self.model.train()
        return float(sum(losses) / max(1, len(losses)))
