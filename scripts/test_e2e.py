"""
End-to-End Test for Unified Transition Diffusion Model
=======================================================
Tests the complete pipeline with the actual dataset format.
"""

import sys
import os
import time
import json
import yaml
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import (
    create_dataloader, TransitionNormalizer,LimitsNormalizer,
    pose_28d_to_config, config_to_pose_28d,
)
from models.diffusion import UnifiedTransitionDiffusion
from visualization.visualize import (
    plot_transition, plot_chain, plot_chain_trajectory, plot_multi_samples,
)

DATA_PATH = "/home/ubuntu/upload/balance_dataset_20260312_070726.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def config_to_viz_dict(config_vec: np.ndarray, slices: dict) -> dict:
    """Convert config vector to visualization dict using slices from config."""
    key_map = {'left_foot': 'LF', 'right_foot': 'RF', 'left_hand': 'LH', 'right_hand': 'RH'}
    result = {'poses': {}, 'contacts': []}
    for key, abbr in key_map.items():
        s = slices[key]
        result['poses'][abbr] = config_vec[s[0]:s[1]].tolist()
    cs = slices['contacts']
    result['contacts'] = config_vec[cs[0]:cs[1]].tolist()
    return result


def test_data_loading():
    """Test 1: Data format detection and loading."""
    print("=" * 60)
    print("TEST 1: Data Loading & Format Conversion")
    print("=" * 60)

    with open(DATA_PATH, 'r') as f:
        raw = json.load(f)

    print(f"  Format: flat (poses={len(raw['poses'])} rows, contacts={len(raw['contacts'])} rows)")
    print(f"  Pairs: {raw['num_pairs']}")
    print(f"  Pose format: {raw['pose_format']}")

    # Test conversion of first pair
    pose_init = np.array(raw['poses'][0], dtype=np.float32)
    contact_init = np.array(raw['contacts'][0], dtype=np.float32)
    config_22d = pose_28d_to_config_22d(pose_init, contact_init)

    print(f"\n  28D -> 22D conversion:")
    print(f"    LF pos: {config_22d[0:3].round(4)}")
    print(f"    LF euler: {config_22d[3:6].round(4)}")
    print(f"    RF pos: {config_22d[6:9].round(4)}")
    print(f"    RF euler: {config_22d[9:12].round(4)}")
    print(f"    LH pos: {config_22d[12:15].round(4)}")
    print(f"    RH pos: {config_22d[15:18].round(4)}")
    print(f"    Contacts: {config_22d[18:22]}")

    # Test round-trip conversion
    pose_28d_back, contacts_back = config_22d_to_pose_28d(config_22d)
    pos_error = np.abs(pose_init[:3] - pose_28d_back[:3]).max()
    print(f"\n  Round-trip position error: {pos_error:.8f}")

    print("  [PASS] Data loading successful")
    return raw


def test_dataset_creation():
    """Test 2: Dataset and DataLoader creation."""
    print("\n" + "=" * 60)
    print("TEST 2: Dataset Creation")
    print("=" * 60)

    config_path = Path(__file__).resolve().parent.parent / "configs" / "unified.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use smaller batch for testing
    config['training']['batch_size'] = 32

    use_orientation = config['data'].get('use_orientation', True)
    dataloader, dataset = create_dataloader(
        data_path=DATA_PATH,
        batch_size=config['training']['batch_size'],
        normalize=True,
        augment=True,
        use_orientation=use_orientation,
    )

    for initial, final in dataloader:
        print(f"  Batch: initial={initial.shape}, final={final.shape}")
        print(f"  Initial range: [{initial.min():.3f}, {initial.max():.3f}]")
        print(f"  Final range: [{final.min():.3f}, {final.max():.3f}]")
        break

    # Check normalizer stats
    print(f"\n  Normalizer mean: {dataset.normalizer.config_mean.round(4)}")
    print(f"  Normalizer std:  {dataset.normalizer.config_std.round(4)}")

    print("  [PASS] Dataset creation successful")
    return config, dataloader, dataset


def test_model(config, dataloader, dataset):
    """Test 3-4: Model construction and training."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Construction")
    print("=" * 60)

    device = torch.device("cpu")
    model = UnifiedTransitionDiffusion(config).to(device)
    num_params = model.get_num_params()
    print(f"  Parameters: {num_params:,}")

    # Quick forward test
    B = 4
    dim = config['data']['config_dim']
    init_test = torch.randn(B, dim)
    final_test = torch.randn(B, dim)
    losses = model.compute_loss(init_test, final_test)
    print(f"  Test loss: {losses['loss'].item():.4f}")
    print("  [PASS] Model construction successful")

    # Training
    print("\n" + "=" * 60)
    print("TEST 4: Training (50 epochs)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    t_start = time.time()

    for epoch in range(50):
        epoch_loss = 0
        n = 0
        for initial, final in dataloader:
            initial, final = initial.to(device), final.to(device)
            losses = model.compute_loss(initial, final)
            optimizer.zero_grad()
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += losses['loss'].item()
            n += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/50 | Loss: {epoch_loss/n:.4f}")

    elapsed = time.time() - t_start
    print(f"  Training time: {elapsed:.1f}s")
    print("  [PASS] Training successful")

    return model, device


def test_joint_sampling(model, dataset, device, slices):
    """Test 5: Joint (unconditional) sampling."""
    print("\n" + "=" * 60)
    print("TEST 5: Joint Sampling")
    print("=" * 60)

    model.eval()
    t_start = time.time()

    init_samples, final_samples = model.sample_transition(
        num_samples=8, device=device, use_ddim=True, ddim_steps=30,
    )
    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.1f}s for 8 samples")

    init_np = dataset.normalizer.denormalize(init_samples.cpu().numpy())
    final_np = dataset.normalizer.denormalize(final_samples.cpu().numpy())

    cs = slices['contacts']
    print(f"  Sample 0 initial: LF_pos={init_np[0,:3].round(3)}, contacts={init_np[0,cs[0]:cs[1]].round(1)}")
    print(f"  Sample 0 final:   LF_pos={final_np[0,:3].round(3)}, contacts={final_np[0,cs[0]:cs[1]].round(1)}")

    # Visualize
    init_dict = config_to_viz_dict(init_np[0], slices)
    final_dict = config_to_viz_dict(final_np[0], slices)
    plot_transition(init_dict, final_dict,
                    title="Joint Sampling: Generated Transition",
                    save_path=str(OUTPUT_DIR / "test_joint_sampling.png"))

    print("  [PASS] Joint sampling successful")
    return init_np, final_np


def test_conditional_forward(model, dataset, device, slices):
    """Test 6: Conditional forward (given initial -> generate final)."""
    print("\n" + "=" * 60)
    print("TEST 6: Conditional Forward Sampling")
    print("=" * 60)

    model.eval()
    cond = dataset.initials_tensor[0:1].to(device).expand(16, -1)

    t_start = time.time()
    _, final_samples = model.sample_transition(
        num_samples=16, device=device, condition_initial=cond,
        guidance_scale=2.0, use_ddim=True, ddim_steps=30,
    )
    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.1f}s for 16 samples")

    init_np = dataset.normalizer.denormalize(dataset.initials_tensor[0:1].cpu().numpy())
    final_np = dataset.normalizer.denormalize(final_samples.cpu().numpy())
    gt_final = dataset.normalizer.denormalize(dataset.finals_tensor[0:1].cpu().numpy())

    lh = slices['left_hand']
    print(f"  Condition LH: {init_np[0,lh[0]:lh[1]].round(4)}")
    print(f"  GT final  LH: {gt_final[0,lh[0]:lh[1]].round(4)}")
    print(f"  Gen final LH (s0): {final_np[0,lh[0]:lh[1]].round(4)}")
    print(f"  Gen final LH (s1): {final_np[1,lh[0]:lh[1]].round(4)}")

    # Visualize
    cond_dict = config_to_viz_dict(init_np[0], slices)
    sample_dicts = [config_to_viz_dict(final_np[i], slices) for i in range(8)]
    plot_multi_samples(sample_dicts, condition_config=cond_dict,
                       title="Forward: Given Initial, Generated Finals",
                       save_path=str(OUTPUT_DIR / "test_conditional_forward.png"))

    print("  [PASS] Conditional forward successful")


def test_conditional_backward(model, dataset, device, slices):
    """Test 7: Conditional backward (given final -> generate initial)."""
    print("\n" + "=" * 60)
    print("TEST 7: Conditional Backward Sampling")
    print("=" * 60)

    model.eval()
    cond = dataset.finals_tensor[0:1].to(device).expand(16, -1)

    t_start = time.time()
    init_samples, _ = model.sample_transition(
        num_samples=16, device=device, condition_final=cond,
        guidance_scale=2.0, use_ddim=True, ddim_steps=30,
    )
    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.1f}s for 16 samples")

    init_np = dataset.normalizer.denormalize(init_samples.cpu().numpy())
    final_np = dataset.normalizer.denormalize(dataset.finals_tensor[0:1].cpu().numpy())

    lh = slices['left_hand']
    print(f"  Condition (final) LH: {final_np[0,lh[0]:lh[1]].round(4)}")
    print(f"  Generated initial LH (s0): {init_np[0,lh[0]:lh[1]].round(4)}")

    print("  [PASS] Conditional backward successful")


def test_chain_planning(model, dataset, device, slices):
    """Test 8: Chain planning with constrained denoising."""
    print("\n" + "=" * 60)
    print("TEST 8: Chain Planning (CompDiffuser style)")
    print("=" * 60)

    model.eval()
    start_norm = dataset.initials_tensor[0].to(device)
    goal_norm = dataset.finals_tensor[min(5, len(dataset) - 1)].to(device)

    # Parallel mode
    print("  Parallel chain planning (K=3)...")
    t_start = time.time()
    best_chain, all_chains = model.plan_chain(
        start_config=start_norm, goal_config=goal_norm,
        num_transitions=3, num_samples=8,
        mode="parallel", use_ddim=True, ddim_steps=30,
    )
    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.1f}s")

    chain_configs = []
    for i, ct in enumerate(best_chain):
        cn = dataset.normalizer.denormalize(ct.cpu().numpy())
        cd = config_to_viz_dict(cn, slices)
        chain_configs.append(cd)
        lh, rh, cs = slices['left_hand'], slices['right_hand'], slices['contacts']
        label = "Start" if i == 0 else ("Goal" if i == len(best_chain)-1 else f"WP{i}")
        print(f"    {label}: LH={cn[lh[0]:lh[1]].round(3)}, RH={cn[rh[0]:rh[1]].round(3)}, "
              f"contacts={cn[cs[0]:cs[1]].round(1)}")

    plot_chain(chain_configs,
               title="Chain Planning (Parallel): Start -> WP1 -> WP2 -> Goal",
               save_path=str(OUTPUT_DIR / "test_chain_parallel.png"))

    plot_chain_trajectory(chain_configs,
                          title="Chain Trajectory",
                          save_path=str(OUTPUT_DIR / "test_chain_trajectory.png"))

    # Autoregressive mode
    print("\n  Autoregressive chain planning (K=3)...")
    t_start = time.time()
    best_chain_ar, _ = model.plan_chain(
        start_config=start_norm, goal_config=goal_norm,
        num_transitions=3, num_samples=8,
        mode="autoregressive", use_ddim=True, ddim_steps=30,
    )
    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.1f}s")

    chain_ar = []
    for ct in best_chain_ar:
        cn = dataset.normalizer.denormalize(ct.cpu().numpy())
        chain_ar.append(config_to_viz_dict(cn, slices))

    plot_chain(chain_ar,
               title="Chain Planning (Autoregressive)",
               save_path=str(OUTPUT_DIR / "test_chain_autoregressive.png"))

    print("  [PASS] Chain planning successful")


def main():
    print("=" * 60)
    print("UNIFIED TRANSITION DIFFUSION MODEL - END-TO-END TEST")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return

    # Test 1: Data loading
    test_data_loading()

    # Test 2: Dataset
    config, dataloader, dataset = test_dataset_creation()

    # Test 3-4: Model + Training
    model, device = test_model(config, dataloader, dataset)

    slices = config['data']['slices']

    # Test 5: Joint sampling
    test_joint_sampling(model, dataset, device, slices)

    # Test 6: Conditional forward
    test_conditional_forward(model, dataset, device, slices)

    # Test 7: Conditional backward
    test_conditional_backward(model, dataset, device, slices)

    # Test 8: Chain planning
    test_chain_planning(model, dataset, device, slices)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("test_*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
