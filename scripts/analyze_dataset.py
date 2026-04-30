"""Analyze SEIKO Talos dataset: structure & coverage with matplotlib.

Generates a multi-panel figure plus a text summary covering:
  - Per-effector position distribution (XY scatter + Z histogram)
  - Contact pattern frequency (16 patterns)
  - Per-effector flip frequency (which EE flips contact in a pair)
  - Non-flip drift distribution (data noise)
  - Flip displacement distribution (transition step size)
  - Pairwise effector distances (workspace coverage)
  - Quaternion validity check

Usage:
    python scripts/analyze_dataset.py \
        --data dataset/seiko_talos_0428.json \
        --output_dir visualization/figures/dataset_0428
"""
import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

EE_NAMES = ["left_foot", "right_foot", "left_hand", "right_hand"]
EE_SHORT = ["LF", "RF", "LH", "RH"]
EE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load(path):
    with open(path, "r") as f:
        d = json.load(f)
    poses = np.array(d["poses"], dtype=np.float64)            # (2N, 28)
    contacts = np.array(d["contacts"], dtype=np.int64)        # (2N, 4)
    n_pairs = d["num_pairs"]
    poses = poses.reshape(n_pairs, 2, 28)
    contacts = contacts.reshape(n_pairs, 2, 4)
    # split per EE: pos (3) + quat (4)
    poses = poses.reshape(n_pairs, 2, 4, 7)
    pos = poses[..., :3]    # (N, 2, 4, 3)
    quat = poses[..., 3:]   # (N, 2, 4, 4)
    return pos, quat, contacts, d


def plot_xy_scatter(ax, pos, contacts):
    """Scatter (x,y) for each effector, only when in contact."""
    # pos: (N,2,4,3); use both init+final stacked
    N = pos.shape[0]
    flat_pos = pos.reshape(-1, 4, 3)         # (2N, 4, 3)
    flat_c = contacts.reshape(-1, 4)         # (2N, 4)
    for i, (name, color) in enumerate(zip(EE_SHORT, EE_COLORS)):
        mask = flat_c[:, i] == 1
        p = flat_pos[mask, i]
        ax.scatter(p[:, 0], p[:, 1], s=2, alpha=0.3, c=color, label=f"{name} (n={mask.sum()})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Per-effector XY (in-contact only)")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)


def plot_z_hist(ax, pos, contacts):
    flat_pos = pos.reshape(-1, 4, 3)
    flat_c = contacts.reshape(-1, 4)
    for i, (name, color) in enumerate(zip(EE_SHORT, EE_COLORS)):
        mask = flat_c[:, i] == 1
        ax.hist(flat_pos[mask, i, 2], bins=50, alpha=0.5, color=color, label=name)
    ax.set_xlabel("z (m)")
    ax.set_ylabel("count")
    ax.set_title("Per-effector Z (in-contact only)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_contact_patterns(ax, contacts):
    # Patterns from both init+final
    flat = contacts.reshape(-1, 4)
    keys = ["".join(str(x) for x in row) for row in flat]
    cnt = Counter(keys)
    all_patterns = [f"{i:04b}" for i in range(16)]
    counts = [cnt.get(p, 0) for p in all_patterns]
    bars = ax.bar(range(16), counts, color="steelblue")
    ax.set_xticks(range(16))
    ax.set_xticklabels(all_patterns, rotation=45, fontsize=8, family="monospace")
    ax.set_xlabel("contact pattern (LF RF LH RH)")
    ax.set_ylabel("count")
    ax.set_title(f"Contact pattern frequency ({len(flat)} samples)")
    for b, c in zip(bars, counts):
        if c > 0:
            ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=7)
    ax.grid(alpha=0.3, axis="y")


def plot_flip_per_ee(ax, contacts):
    diff = contacts[:, 1] - contacts[:, 0]      # (N, 4)
    flip = (diff != 0).astype(int)
    counts = flip.sum(axis=0)
    n_flip_total = flip.sum(axis=1)
    bars = ax.bar(EE_SHORT, counts, color=EE_COLORS)
    ax.set_ylabel("# pairs flipping this EE")
    ax.set_title(
        f"Flip count per EE  |  single-flip rate = {(n_flip_total == 1).mean():.3%}"
    )
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(int(c)), ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")


def plot_non_flip_drift(ax, pos, contacts):
    """For pairs with exactly 1 flip, distance of non-flipping EEs between init/final."""
    diff = contacts[:, 1] - contacts[:, 0]
    flip = (diff != 0)
    single = flip.sum(axis=1) == 1
    pos_s = pos[single]                  # (M, 2, 4, 3)
    flip_s = flip[single]                # (M, 4)
    delta = np.linalg.norm(pos_s[:, 1] - pos_s[:, 0], axis=-1)  # (M, 4)
    drifts = delta[~flip_s]              # non-flipping EEs
    drifts_mm = drifts * 1000.0
    ax.hist(drifts_mm, bins=80, color="darkorange", alpha=0.8)
    ax.axvline(drifts_mm.mean(), color="k", ls="--", label=f"mean={drifts_mm.mean():.2f}mm")
    p99 = np.percentile(drifts_mm, 99)
    ax.axvline(p99, color="r", ls="--", label=f"p99={p99:.2f}mm")
    ax.set_xlabel("non-flip EE drift (mm)")
    ax.set_ylabel("count")
    ax.set_title("Non-flip drift (data noise indicator)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_flip_displacement(ax, pos, contacts):
    diff = contacts[:, 1] - contacts[:, 0]
    flip = (diff != 0)
    single = flip.sum(axis=1) == 1
    pos_s = pos[single]
    flip_s = flip[single]
    delta = np.linalg.norm(pos_s[:, 1] - pos_s[:, 0], axis=-1)
    disp = delta[flip_s]
    disp_mm = disp * 1000.0
    ax.hist(disp_mm, bins=60, color="steelblue", alpha=0.85)
    ax.axvline(disp_mm.mean(), color="k", ls="--", label=f"mean={disp_mm.mean():.1f}mm")
    ax.axvline(np.median(disp_mm), color="g", ls="--", label=f"median={np.median(disp_mm):.1f}mm")
    ax.set_xlabel("flipping EE displacement (mm)")
    ax.set_ylabel("count")
    ax.set_title("Flip-step displacement")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_pairwise_dist(ax, pos):
    """Coverage of relative geometry: distance between feet, between hands."""
    flat = pos.reshape(-1, 4, 3)         # (2N, 4, 3)
    d_feet = np.linalg.norm(flat[:, 0] - flat[:, 1], axis=-1)
    d_hands = np.linalg.norm(flat[:, 2] - flat[:, 3], axis=-1)
    d_lf_lh = np.linalg.norm(flat[:, 0] - flat[:, 2], axis=-1)
    d_rf_rh = np.linalg.norm(flat[:, 1] - flat[:, 3], axis=-1)
    bins = np.linspace(0, max(d_feet.max(), d_hands.max(), d_lf_lh.max(), d_rf_rh.max()), 60)
    ax.hist(d_feet, bins=bins, alpha=0.5, label=f"feet (μ={d_feet.mean():.2f})")
    ax.hist(d_hands, bins=bins, alpha=0.5, label=f"hands (μ={d_hands.mean():.2f})")
    ax.hist(d_lf_lh, bins=bins, alpha=0.5, label=f"LF-LH (μ={d_lf_lh.mean():.2f})")
    ax.hist(d_rf_rh, bins=bins, alpha=0.5, label=f"RF-RH (μ={d_rf_rh.mean():.2f})")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("count")
    ax.set_title("Pairwise EE distances (workspace shape)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_quat_norm(ax, quat):
    norms = np.linalg.norm(quat.reshape(-1, 4), axis=-1)
    ax.hist(norms, bins=80, color="purple", alpha=0.8)
    ax.set_xlabel("|q|")
    ax.set_ylabel("count")
    ax.set_title(
        f"Quaternion norms  |  mean={norms.mean():.5f}  min={norms.min():.4f}  max={norms.max():.4f}"
    )
    ax.set_yscale("log")
    ax.grid(alpha=0.3)


def text_summary(pos, quat, contacts, meta):
    lines = []
    n_pairs = pos.shape[0]
    lines.append(f"# Dataset summary")
    lines.append(f"description     : {meta.get('description', '')}")
    lines.append(f"end_effectors   : {meta.get('end_effectors')}")
    lines.append(f"num_pairs       : {n_pairs}")
    lines.append(f"num_samples     : {2 * n_pairs}")
    if "source_files" in meta:
        lines.append("source_files    :")
        for s in meta["source_files"]:
            lines.append(f"  - {s['path']}: {s['num_pairs']} pairs")

    diff = contacts[:, 1] - contacts[:, 0]
    flip = (diff != 0)
    n_flip = flip.sum(axis=1)
    lines.append("")
    lines.append(f"## Contact flips per pair")
    for k in range(5):
        cnt = int((n_flip == k).sum())
        if cnt:
            lines.append(f"  exactly {k} flip(s): {cnt}  ({cnt / n_pairs:.2%})")
    lines.append(f"  single-flip rate: {(n_flip == 1).mean():.3%}")

    lines.append("")
    lines.append(f"## Per-EE flip count (single-flip pairs)")
    single = n_flip == 1
    flip_s = flip[single]
    for s, c in zip(EE_SHORT, flip_s.sum(axis=0)):
        lines.append(f"  {s}: {int(c)}")

    pos_s = pos[single]
    flip_s_mask = flip[single]
    delta = np.linalg.norm(pos_s[:, 1] - pos_s[:, 0], axis=-1) * 1000.0
    nf = delta[~flip_s_mask]
    fl = delta[flip_s_mask]
    lines.append("")
    lines.append("## Non-flip drift (mm)")
    lines.append(f"  mean={nf.mean():.3f}  median={np.median(nf):.3f}  "
                 f"p95={np.percentile(nf, 95):.3f}  p99={np.percentile(nf, 99):.3f}  max={nf.max():.3f}")
    lines.append(f"  ratio > 1mm : {(nf > 1).mean():.3%}")
    lines.append(f"  ratio > 10mm: {(nf > 10).mean():.3%}")

    lines.append("")
    lines.append("## Flip displacement (mm)")
    lines.append(f"  mean={fl.mean():.2f}  median={np.median(fl):.2f}  "
                 f"min={fl.min():.2f}  max={fl.max():.2f}  p1={np.percentile(fl, 1):.2f}")

    lines.append("")
    lines.append("## Pose ranges (m)")
    flat = pos.reshape(-1, 4, 3)
    flat_c = contacts.reshape(-1, 4)
    for i, name in enumerate(EE_SHORT):
        m = flat_c[:, i] == 1
        if m.sum() == 0:
            continue
        p = flat[m, i]
        lines.append(f"  {name}: x[{p[:, 0].min():+.3f},{p[:, 0].max():+.3f}]  "
                     f"y[{p[:, 1].min():+.3f},{p[:, 1].max():+.3f}]  "
                     f"z[{p[:, 2].min():+.3f},{p[:, 2].max():+.3f}]  (n={m.sum()})")

    norms = np.linalg.norm(quat.reshape(-1, 4), axis=-1)
    lines.append("")
    lines.append(f"## Quaternion norms: mean={norms.mean():.6f}  "
                 f"min={norms.min():.4f}  max={norms.max():.4f}  "
                 f"out-of-tol(>1e-3): {(np.abs(norms - 1) > 1e-3).sum()}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output_dir", default="visualization/figures/dataset_analysis")
    args = ap.parse_args()

    pos, quat, contacts, meta = load(args.data)
    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    plot_xy_scatter(axes[0, 0], pos, contacts)
    plot_z_hist(axes[0, 1], pos, contacts)
    plot_quat_norm(axes[0, 2], quat)
    plot_contact_patterns(axes[1, 0], contacts)
    plot_flip_per_ee(axes[1, 1], contacts)
    plot_pairwise_dist(axes[1, 2], pos)
    plot_non_flip_drift(axes[2, 0], pos, contacts)
    plot_flip_displacement(axes[2, 1], pos, contacts)
    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.0, 1.0,
        f"Dataset: {os.path.basename(args.data)}\n"
        f"pairs: {pos.shape[0]}\n"
        f"samples: {2 * pos.shape[0]}\n"
        f"format: {meta.get('pose_format')}\n",
        fontsize=11, family="monospace", va="top",
    )

    fig.suptitle(f"Dataset analysis — {os.path.basename(args.data)}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = os.path.join(args.output_dir, "overview.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    summary = text_summary(pos, quat, contacts, meta)
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    print(summary)
    print()
    print(f"figure : {fig_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
