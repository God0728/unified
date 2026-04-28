"""
Training Script for Unified Transition Diffusion Model
=======================================================
Usage:
  python scripts/train.py --config configs/unified.yaml --data path/to/data.json

Features:
  - Independent timestep training (UWM style)
  - EMA model averaging
  - Periodic evaluation and checkpointing
  - TensorBoard logging
"""

import sys
import os
import argparse
import yaml
import time
import json
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import create_dataloader
from models.diffusion import UnifiedTransitionDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified Transition Diffusion Model")
    parser.add_argument("--config", type=str, default="configs/unified.yaml", help="Config file path")
    parser.add_argument("--data", type=str, required=True, help="Path to transition dataset JSON")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model: torch.nn.Module):
        """Apply EMA weights to model."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model: torch.nn.Module):
        """Restore original weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


def evaluate(model, dataloader, device, num_eval_samples=8):
    """Quick evaluation: compute validation loss and generate samples."""
    model.eval()
    total_loss = 0
    total_loss_init = 0
    total_loss_final = 0
    num_batches = 0

    with torch.no_grad():
        for initial, final in dataloader:
            initial = initial.to(device)
            final = final.to(device)
            losses = model.compute_loss(initial, final)
            total_loss += losses['loss'].item()
            total_loss_init += losses['loss_init']
            total_loss_final += losses['loss_final']
            num_batches += 1
            if num_batches >= 10:  # Limit eval batches
                break

    avg_loss = total_loss / max(num_batches, 1)
    avg_loss_init = total_loss_init / max(num_batches, 1)
    avg_loss_final = total_loss_final / max(num_batches, 1)

    # Generate sample transitions
    sample_initial, sample_final = model.sample_transition(
        num_samples=num_eval_samples,
        device=device,
        use_ddim=True,
        ddim_steps=50,
    )

    model.train()
    return {
        'val_loss': avg_loss,
        'val_loss_init': avg_loss_init,
        'val_loss_final': avg_loss_final,
        'sample_initial': sample_initial.cpu(),
        'sample_final': sample_final.cpu(),
    }


def train(args):
    # Load config
    config_path = Path(__file__).resolve().parent.parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    train_cfg = config['training']
    if args.epochs is not None:
        train_cfg['num_epochs'] = args.epochs
    if args.batch_size is not None:
        train_cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        train_cfg['learning_rate'] = args.lr

    device = get_device(args.device)
    print(f"[Train] Device: {device}")

    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_dir = output_dir / "logs"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Create dataloader
    use_orientation = config['data'].get('use_orientation', True)
    dataloader, dataset = create_dataloader(
        data_path=args.data,
        batch_size=train_cfg['batch_size'],
        normalize=config['data'].get('normalize', True),
        augment=True,
        use_orientation=use_orientation,
    )
    print(f"[Train] Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Build model
    model = UnifiedTransitionDiffusion(config).to(device)
    num_params = model.get_num_params()
    print(f"[Train] Model parameters: {num_params:,}")

    # Optimizer with warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 1e-6),
    )

    # Learning rate scheduler with warmup
    warmup_steps = train_cfg.get('warmup_steps', 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA
    ema = None
    if train_cfg.get('use_ema', True):
        ema = EMAModel(model, decay=train_cfg.get('ema_decay', 0.9999))

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"[Train] Resumed from epoch {start_epoch}, step {global_step}")

    # Save normalizer
    normalizer_path = output_dir / "normalizer.pt"
    torch.save(dataset.normalizer.state_dict(), normalizer_path)

    # Save config
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    # Training loop
    print(f"\n[Train] Starting training for {train_cfg['num_epochs']} epochs...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Independent timesteps: YES (UWM style)")
    print(f"  Conditioning: drop={train_cfg.get('tr_1side_drop_prob', 0.15)}, "
          f"ovlp={train_cfg.get('tr_ovlp_prob', 0.5)}, inpat={train_cfg.get('tr_inpat_prob', 0.5)}")
    print()

    model.train()
    for epoch in range(start_epoch, train_cfg['num_epochs']):
        epoch_loss = 0
        epoch_loss_init = 0
        epoch_loss_final = 0
        epoch_loss_cls = 0
        num_batches = 0
        t_start = time.time()

        for initial, final in dataloader:
            initial = initial.to(device)
            final = final.to(device)

            # Forward pass
            losses = model.compute_loss(initial, final)
            loss = losses['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update
            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            epoch_loss_init += losses['loss_init']
            epoch_loss_final += losses['loss_final']
            epoch_loss_cls += losses.get('loss_change_cls', 0.0)
            num_batches += 1
            global_step += 1

            # TensorBoard logging
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/loss_init', losses['loss_init'], global_step)
            writer.add_scalar('train/loss_final', losses['loss_final'], global_step)
            writer.add_scalar('train/loss_change_cls', losses.get('loss_change_cls', 0.0), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_loss_init = epoch_loss_init / max(num_batches, 1)
        avg_loss_final = epoch_loss_final / max(num_batches, 1)
        avg_loss_cls = epoch_loss_cls / max(num_batches, 1)
        elapsed = time.time() - t_start

        if (epoch + 1) % train_cfg.get('log_interval', 50) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{train_cfg['num_epochs']} | "
                  f"Loss: {avg_loss:.4f} (init: {avg_loss_init:.4f}, final: {avg_loss_final:.4f}, "
                  f"cls: {avg_loss_cls:.4f}) | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {elapsed:.1f}s")

        # Evaluation
        if (epoch + 1) % train_cfg.get('eval_interval', 200) == 0:
            if ema is not None:
                ema.apply(model)

            eval_results = evaluate(model, dataloader, device)
            print(f"  [Eval] Val Loss: {eval_results['val_loss']:.4f} "
                  f"(init: {eval_results['val_loss_init']:.4f}, final: {eval_results['val_loss_final']:.4f})")

            writer.add_scalar('eval/val_loss', eval_results['val_loss'], global_step)

            if ema is not None:
                ema.restore(model)

        # Epoch snapshot: save generated samples for distribution analysis
        snapshot_interval = train_cfg.get('snapshot_interval', 1000)
        if snapshot_interval > 0 and (epoch + 1) % snapshot_interval == 0:
            snap_dir = output_dir / "snapshots"
            snap_dir.mkdir(exist_ok=True)
            if ema is not None:
                ema.apply(model)
            model.eval()
            with torch.no_grad():
                snap_ini, snap_fin = model.sample_transition(
                    num_samples=min(256, len(dataset)),
                    device=device, use_ddim=True, ddim_steps=50,
                )
            snap_ini_np = snap_ini.cpu().numpy()
            snap_fin_np = snap_fin.cpu().numpy()
            # Denormalize if normalizer available
            if hasattr(dataset, 'normalizer') and dataset.normalizer.fitted:
                snap_ini_np = dataset.normalizer.denormalize(snap_ini_np)
                snap_fin_np = dataset.normalizer.denormalize(snap_fin_np)
            np.savez(
                snap_dir / f"epoch_{epoch+1}.npz",
                initial=snap_ini_np, final=snap_fin_np, epoch=epoch+1,
            )
            print(f"  [Snapshot] Saved epoch {epoch+1} samples -> {snap_dir}/epoch_{epoch+1}.npz")
            model.train()
            if ema is not None:
                ema.restore(model)

        # Save checkpoint
        if (epoch + 1) % train_cfg.get('save_interval', 200) == 0:
            save_model = model
            if ema is not None:
                ema.apply(model)
                save_model = model

            checkpoint = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_loss': best_loss,
                'normalizer': dataset.normalizer.state_dict(),
            }

            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save(checkpoint, ckpt_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_dir / "checkpoint_best.pt"
                torch.save(checkpoint, best_path)
                print(f"  [Save] New best model (loss={best_loss:.4f}) -> {best_path}")

            if ema is not None:
                ema.restore(model)

    # Save final model
    if ema is not None:
        ema.apply(model)

    final_checkpoint = {
        'epoch': train_cfg['num_epochs'],
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'best_loss': best_loss,
        'normalizer': dataset.normalizer.state_dict(),
    }
    final_path = output_dir / "checkpoint_final.pt"
    torch.save(final_checkpoint, final_path)
    print(f"\n[Train] Training complete. Final model saved to {final_path}")
    print(f"  Best loss: {best_loss:.4f}")

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
