"""
Sampling Script for Unified Transition Diffusion Model
=======================================================
Supports:
  1. Joint generation: sample full (initial, final) transitions
  2. Conditional generation: given initial, generate final (or vice versa)
  3. Chain planning: generate A -> c1 -> c2 -> ... -> D with constraints

Usage:
  # Joint generation
  python scripts/sample.py --checkpoint checkpoints/checkpoint_best.pt --mode joint

  # Conditional: given initial, generate final
  python scripts/sample.py --checkpoint checkpoints/checkpoint_best.pt --mode forward \
      --condition_initial "0,0,0,0,0,0, 0,0.3,0,0,0,0, 0.5,0.5,1.0, -0.5,0.5,1.0, 1,1,1,1"

  # Chain planning
  python scripts/sample.py --checkpoint checkpoints/checkpoint_best.pt --mode chain \
      --start "..." --goal "..." --num_transitions 3
"""

import sys
import os
import argparse
import yaml
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import TransitionNormalizer, LimitsNormalizer
from models.diffusion import UnifiedTransitionDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from Unified Transition Diffusion Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="joint",
                        choices=["joint", "forward", "backward", "chain"],
                        help="Sampling mode")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--output", type=str, default="outputs/samples.json", help="Output file")

    # Conditional generation
    parser.add_argument("--condition_initial", type=str, default=None,
                        help="Comma-separated config vector for forward mode")
    parser.add_argument("--condition_final", type=str, default=None,
                        help="Comma-separated config vector for backward mode")

    # Chain planning
    parser.add_argument("--start", type=str, default=None,
                        help="Comma-separated start config for chain mode")
    parser.add_argument("--goal", type=str, default=None,
                        help="Comma-separated goal config for chain mode")
    parser.add_argument("--num_transitions", type=int, default=3,
                        help="Number of transitions in chain")
    parser.add_argument("--chain_mode", type=str, default="parallel",
                        choices=["parallel", "autoregressive"],
                        help="Chain sampling mode")

    return parser.parse_args()


def parse_config_str(s: str) -> np.ndarray:
    """Parse comma-separated string to numpy array."""
    return np.array([float(x.strip()) for x in s.split(',')])


def load_model(checkpoint_path: str, device: torch.device):
    """Load model and normalizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Build model
    model = UnifiedTransitionDiffusion(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load normalizer
    normalizer = LimitsNormalizer()
    normalizer.load_state_dict(checkpoint['normalizer'])

    return model, normalizer, config


def config_to_dict(config_vec: np.ndarray, slices: dict) -> dict:
    """Convert config vector to structured dict using slices from config."""
    key_map = {'left_foot': 'LF', 'right_foot': 'RF', 'left_hand': 'LH', 'right_hand': 'RH'}
    result = {'poses': {}, 'contacts': []}
    for key, abbr in key_map.items():
        s = slices[key]
        result['poses'][abbr] = config_vec[s[0]:s[1]].tolist()
    cs = slices['contacts']
    result['contacts'] = config_vec[cs[0]:cs[1]].tolist()
    return result


def sample_joint(model, normalizer, args, device, slices):
    """Generate full (initial, final) transition pairs."""
    print(f"[Sample] Joint generation: {args.num_samples} samples...")

    initial_samples, final_samples = model.sample_transition(
        num_samples=args.num_samples,
        device=device,
        guidance_scale=args.guidance_scale,
        use_ddim=True,
        ddim_steps=args.ddim_steps,
    )

    # Denormalize
    initial_np = normalizer.denormalize(initial_samples.cpu().numpy())
    final_np = normalizer.denormalize(final_samples.cpu().numpy())

    results = []
    for i in range(args.num_samples):
        results.append({
            'sample_id': i,
            'initial': config_to_dict(initial_np[i], slices),
            'final': config_to_dict(final_np[i], slices),
        })

    return results


def sample_forward(model, normalizer, args, device, slices):
    """Given initial config, generate final config."""
    assert args.condition_initial is not None, "Must provide --condition_initial for forward mode"

    initial_raw = parse_config_str(args.condition_initial)
    config_dim = model.config_dim
    assert len(initial_raw) == config_dim, f"Expected {config_dim}D, got {len(initial_raw)}D"

    # Normalize
    initial_norm = normalizer.normalize(initial_raw)
    initial_tensor = torch.tensor(initial_norm, dtype=torch.float32, device=device)
    initial_batch = initial_tensor.unsqueeze(0).expand(args.num_samples, -1)

    print(f"[Sample] Forward generation: given initial, generating {args.num_samples} final configs...")

    _, final_samples = model.sample_transition(
        num_samples=args.num_samples,
        device=device,
        condition_initial=initial_batch,
        guidance_scale=args.guidance_scale,
        use_ddim=True,
        ddim_steps=args.ddim_steps,
    )

    # Denormalize
    final_np = normalizer.denormalize(final_samples.cpu().numpy())

    results = []
    for i in range(args.num_samples):
        results.append({
            'sample_id': i,
            'initial': config_to_dict(initial_raw, slices),
            'final': config_to_dict(final_np[i], slices),
        })

    return results


def sample_backward(model, normalizer, args, device, slices):
    """Given final config, generate initial config."""
    assert args.condition_final is not None, "Must provide --condition_final for backward mode"

    final_raw = parse_config_str(args.condition_final)
    config_dim = model.config_dim
    assert len(final_raw) == config_dim, f"Expected {config_dim}D, got {len(final_raw)}D"

    # Normalize
    final_norm = normalizer.normalize(final_raw)
    final_tensor = torch.tensor(final_norm, dtype=torch.float32, device=device)
    final_batch = final_tensor.unsqueeze(0).expand(args.num_samples, -1)

    print(f"[Sample] Backward generation: given final, generating {args.num_samples} initial configs...")

    initial_samples, _ = model.sample_transition(
        num_samples=args.num_samples,
        device=device,
        condition_final=final_batch,
        guidance_scale=args.guidance_scale,
        use_ddim=True,
        ddim_steps=args.ddim_steps,
    )

    # Denormalize
    initial_np = normalizer.denormalize(initial_samples.cpu().numpy())

    results = []
    for i in range(args.num_samples):
        results.append({
            'sample_id': i,
            'initial': config_to_dict(initial_np[i], slices),
            'final': config_to_dict(final_raw, slices),
        })

    return results


def sample_chain(model, normalizer, args, device, slices):
    """Chain planning: generate A -> c1 -> c2 -> ... -> D."""
    assert args.start is not None, "Must provide --start for chain mode"
    assert args.goal is not None, "Must provide --goal for chain mode"

    start_raw = parse_config_str(args.start)
    goal_raw = parse_config_str(args.goal)
    config_dim = model.config_dim
    assert len(start_raw) == config_dim and len(goal_raw) == config_dim, \
        f"Expected {config_dim}D configs"

    # Normalize
    start_norm = normalizer.normalize(start_raw)
    goal_norm = normalizer.normalize(goal_raw)
    start_tensor = torch.tensor(start_norm, dtype=torch.float32, device=device)
    goal_tensor = torch.tensor(goal_norm, dtype=torch.float32, device=device)

    K = args.num_transitions
    print(f"[Sample] Chain planning: {K} transitions, {args.chain_mode} mode...")
    print(f"  Start -> {K-1} intermediate waypoints -> Goal")

    best_chain, all_chains = model.plan_chain(
        start_config=start_tensor,
        goal_config=goal_tensor,
        num_transitions=K,
        num_samples=args.num_samples,
        mode=args.chain_mode,
        use_ddim=True,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
    )

    # Denormalize best chain
    chain_configs = []
    for i, config_tensor in enumerate(best_chain):
        config_np = normalizer.denormalize(config_tensor.cpu().numpy())
        chain_configs.append({
            'waypoint_id': i,
            'label': 'start' if i == 0 else ('goal' if i == K else f'intermediate_{i}'),
            'config': config_to_dict(config_np, slices),
        })

    # Also save all candidate chains
    config_dim = model.config_dim
    all_chains_np = normalizer.denormalize(
        all_chains.cpu().numpy().reshape(-1, config_dim)
    ).reshape(args.num_samples, K + 1, config_dim)

    results = {
        'best_chain': chain_configs,
        'num_transitions': K,
        'chain_mode': args.chain_mode,
        'num_candidates': args.num_samples,
        'all_chains': [
            [config_to_dict(all_chains_np[s, w, :], slices) for w in range(K + 1)]
            for s in range(args.num_samples)
        ],
    }

    return results


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"[Sample] Device: {device}")

    # Load model
    model, normalizer, config = load_model(args.checkpoint, device)
    slices = config['data']['slices']
    print(f"[Sample] Model loaded from {args.checkpoint} (config_dim={model.config_dim})")

    # Run sampling
    if args.mode == "joint":
        results = sample_joint(model, normalizer, args, device, slices)
    elif args.mode == "forward":
        results = sample_forward(model, normalizer, args, device, slices)
    elif args.mode == "backward":
        results = sample_backward(model, normalizer, args, device, slices)
    elif args.mode == "chain":
        results = sample_chain(model, normalizer, args, device, slices)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Save results
    output_path = Path(__file__).resolve().parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Sample] Results saved to {output_path}")


if __name__ == "__main__":
    main()
