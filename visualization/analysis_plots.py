"""
分析图表工具
============
提供三类可视化:
  图2: 边际分布对比 (per-joint 直方图)
  图6: Chain 连接点一致性 (热力图 + 柱状图)
  图7: Chain 关节轨迹连贯性 (折线图)

使用方式:
  python visualization/analysis_plots.py \
      --checkpoint checkpoints/checkpoint_best.pt \
      --data dataset/balance_dataset.json \
      --chain_output outputs/seiko_chain_ee_test5.json \
      --output_dir visualization/figures
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 暖色系误差 colormap
_ERROR_CMAP = LinearSegmentedColormap.from_list(
    'warm_error', ['#f0f0f0', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000']
)

# ── Helpers ──────────────────────────────────────────────────

def _get_dim_labels(config_dim: int, slices: dict) -> list:
    """生成每个维度的可读标签."""
    labels = [''] * config_dim
    name_map = {
        'left_foot': 'LF', 'right_foot': 'RF',
        'left_hand': 'LH', 'right_hand': 'RH', 'contacts': 'Ct',
    }
    sub_labels_6d = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    sub_labels_3d = ['x', 'y', 'z']
    sub_labels_ct = ['LF', 'RF', 'LH', 'RH']
    for key, abbr in name_map.items():
        s = slices[key]
        dim = s[1] - s[0]
        if key == 'contacts':
            subs = sub_labels_ct[:dim]
        elif dim == 6:
            subs = sub_labels_6d
        else:
            subs = sub_labels_3d[:dim]
        for i, sub in enumerate(subs):
            labels[s[0] + i] = f'{abbr}_{sub}'
    return labels


def _load_training_data(data_path: str, use_orientation: bool):
    """从原始 JSON 加载并转为 config 向量 (归一化前)."""
    from data.dataset import create_dataloader
    _, dataset = create_dataloader(
        data_path=data_path,
        batch_size=64,
        normalize=False,
        augment=False,
        use_orientation=use_orientation,
    )
    all_initial = []
    all_final = []
    for i in range(len(dataset)):
        ini, fin = dataset[i]
        all_initial.append(ini.numpy())
        all_final.append(fin.numpy())
    return np.array(all_initial), np.array(all_final)


def _load_generated_samples(json_path: str, slices: dict, config_dim: int):
    """从 sample.py 输出的 JSON 加载生成样本, 还原为 (N, config_dim) 向量."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # joint/forward/backward 模式 → list of dicts
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict) and 'best_chain' in data:
        # chain 模式 → 不适用于边际分布图
        raise ValueError("Chain output 不适用于边际分布对比, 请提供 joint/forward/backward 模式的输出")
    else:
        raise ValueError(f"Unknown JSON format")

    initials, finals = [], []
    for s in samples:
        initials.append(_config_dict_to_vec(s['initial'], slices, config_dim))
        finals.append(_config_dict_to_vec(s['final'], slices, config_dim))
    return np.array(initials), np.array(finals)


def _config_dict_to_vec(config_dict: dict, slices: dict, config_dim: int) -> np.ndarray:
    """将 {'poses': {'LF': [...], ...}, 'contacts': [...]} 还原为 flat vector."""
    vec = np.zeros(config_dim)
    key_map = {'left_foot': 'LF', 'right_foot': 'RF', 'left_hand': 'LH', 'right_hand': 'RH'}
    for key, abbr in key_map.items():
        s = slices[key]
        vec[s[0]:s[1]] = config_dict['poses'][abbr]
    cs = slices['contacts']
    vec[cs[0]:cs[1]] = config_dict['contacts']
    return vec


def _load_chain_output(json_path: str, slices: dict, config_dim: int):
    """加载 chain 输出, 返回 best_chain (K+1, D) 和 all_chains (S, K+1, D)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    assert isinstance(data, dict) and 'best_chain' in data, \
        "需要 chain 模式的输出 JSON"

    best = []
    for wp in data['best_chain']:
        best.append(_config_dict_to_vec(wp['config'], slices, config_dim))
    best = np.array(best)  # (K+1, D)

    all_chains = []
    for chain in data['all_chains']:
        c = []
        for wp_dict in chain:
            c.append(_config_dict_to_vec(wp_dict, slices, config_dim))
        all_chains.append(c)
    all_chains = np.array(all_chains)  # (S, K+1, D)

    return best, all_chains, data.get('num_transitions', len(best) - 1)


# ═════════════════════════════════════════════════════════════
# 图2: 边际分布对比 (per-joint 直方图)
# ═════════════════════════════════════════════════════════════

def plot_marginal_distributions(
    train_data: np.ndarray,      # (N_train, D)
    gen_data: np.ndarray,        # (N_gen, D)
    dim_labels: list,
    title_prefix: str = "Initial",
    save_path: str = None,
    config_dim: int = 16,
    contact_start: int = 12,
):
    """
    Per-joint 边际分布对比直方图.

    蓝色 = 训练数据, 橙色 = 模型生成.
    Contact 维度使用柱状图 (离散 0/1).
    """
    D = config_dim
    ncols = 4
    nrows = int(np.ceil(D / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.4))
    axes = axes.flatten()

    fig.suptitle(
        f'Marginal Distribution Comparison — {title_prefix}',
        fontsize=14, fontweight='bold', y=1.02,
    )

    for d in range(D):
        ax = axes[d]
        tr = train_data[:, d]
        ge = gen_data[:, d]

        if d >= contact_start:
            # Contact 维度: 离散柱状图
            bins_ct = [-0.5, 0.5, 1.5]
            tr_counts, _ = np.histogram(tr, bins=bins_ct)
            ge_counts, _ = np.histogram(np.clip(ge, -0.5, 1.5), bins=bins_ct)
            tr_frac = tr_counts / tr_counts.sum()
            ge_frac = ge_counts / max(ge_counts.sum(), 1)
            x_pos = np.array([0, 1])
            w = 0.35
            ax.bar(x_pos - w/2, tr_frac, w, color='#4c72b0', alpha=0.85, label='Train')
            ax.bar(x_pos + w/2, ge_frac, w, color='#dd8452', alpha=0.85, label='Gen')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['0', '1'])
            ax.set_ylabel('Fraction')
        else:
            # 连续维度: 叠加直方图
            lo = min(tr.min(), ge.min())
            hi = max(tr.max(), ge.max())
            margin = (hi - lo) * 0.05
            bins = np.linspace(lo - margin, hi + margin, 40)
            ax.hist(tr, bins=bins, density=True, color='#4c72b0', alpha=0.6, label='Train')
            ax.hist(ge, bins=bins, density=True, color='#dd8452', alpha=0.6, label='Gen')
            ax.set_ylabel('Density')

        ax.set_title(dim_labels[d], fontsize=10, fontweight='medium')
        if d == 0:
            ax.legend(frameon=False, loc='upper right')

    # 隐藏多余子图
    for d in range(D, len(axes)):
        axes[d].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[Plot] Saved marginal distribution plot -> {save_path}")
    plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════
# 图6: Chain 连接点一致性
# ═════════════════════════════════════════════════════════════

def plot_chain_consistency(
    all_chains: np.ndarray,      # (S, K+1, D)
    best_idx: int,
    dim_labels: list,
    save_path: str = None,
    config_dim: int = 16,
):
    """
    Chain 连接误差分析:
      子图 A: per-joint 误差热力图
      子图 B: per-waypoint 总误差柱状图
    """
    S, Kp1, D = all_chains.shape
    K = Kp1 - 1
    num_connections = K - 1

    if num_connections < 1:
        print("[Plot] Only 1 transition — no connection points to analyze.")
        return None

    # 连接误差: finals[k] ≈ waypoint k+1, initials[k+1] ≈ waypoint k+1
    # 在 plan_chain 中, all_chains[:, k+1, :] = avg(finals[k], initials[k+1])
    # 我们用相邻 waypoint 之间的差异来近似

    # 为每条候选链计算相邻 waypoint 间的 per-dim 绝对差
    # connection k 连接 transition k 和 k+1, 对应 waypoint k+1
    diffs_all = []  # (S, num_connections, D)
    for s in range(S):
        chain_diffs = []
        for k in range(num_connections):
            # waypoint k+1 是连接点
            # 前后跳变 = |wp[k+2] - wp[k+1]| 和 |wp[k+1] - wp[k]| 的平均
            d_left = np.abs(all_chains[s, k+1, :] - all_chains[s, k, :])
            d_right = np.abs(all_chains[s, k+2, :] - all_chains[s, k+1, :])
            chain_diffs.append((d_left + d_right) / 2)
        diffs_all.append(chain_diffs)
    diffs_all = np.array(diffs_all)  # (S, num_connections, D)

    # 用最佳链的误差做主图
    best_diffs = diffs_all[best_idx]  # (num_connections, D)
    mean_diffs = diffs_all.mean(axis=0)  # (num_connections, D) 所有候选平均

    fig = plt.figure(figsize=(max(12, D * 0.6), 3.5 + num_connections * 0.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.3)

    # ── 子图 A: 热力图 ──
    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(
        mean_diffs, aspect='auto', cmap=_ERROR_CMAP,
        interpolation='nearest',
    )
    ax_heat.set_xticks(range(D))
    ax_heat.set_xticklabels(dim_labels, rotation=55, ha='right', fontsize=7)
    ax_heat.set_yticks(range(num_connections))
    ax_heat.set_yticklabels([f'Conn {k+1}' for k in range(num_connections)])
    ax_heat.set_xlabel('Joint Dimension')
    ax_heat.set_ylabel('Connection Point')
    ax_heat.set_title('Per-Joint Connection Error\n(avg across candidates)', fontweight='bold')

    # 在每个格子里标数值
    for i in range(num_connections):
        for j in range(D):
            val = mean_diffs[i, j]
            color = 'white' if val > mean_diffs.max() * 0.6 else 'black'
            ax_heat.text(j, i, f'{val:.3f}', ha='center', va='center',
                         fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Abs. Error', fontsize=8)

    # ── 子图 B: 柱状图 ──
    ax_bar = fig.add_subplot(gs[1])
    total_err_per_conn_best = np.linalg.norm(best_diffs, axis=-1)        # (num_connections,)
    total_err_per_conn_mean = np.linalg.norm(mean_diffs, axis=-1)

    x = np.arange(num_connections)
    w = 0.35
    ax_bar.barh(x - w/2, total_err_per_conn_mean, w, color='#7fbf7b', alpha=0.85, label='Avg')
    ax_bar.barh(x + w/2, total_err_per_conn_best, w, color='#1b7837', alpha=0.85, label='Best')
    ax_bar.set_yticks(x)
    ax_bar.set_yticklabels([f'Conn {k+1}' for k in range(num_connections)])
    ax_bar.set_xlabel('L2 Error')
    ax_bar.set_title('Total Error\nper Connection', fontweight='bold')
    ax_bar.legend(frameon=False, fontsize=7)
    ax_bar.invert_yaxis()

    fig.suptitle('Chain Connection Consistency Analysis', fontsize=14, fontweight='bold', y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path)
        print(f"[Plot] Saved chain consistency plot -> {save_path}")
    plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════
# 图7: Chain 关节轨迹连贯性
# ═════════════════════════════════════════════════════════════

def plot_chain_trajectory_lines(
    best_chain: np.ndarray,     # (K+1, D)
    all_chains: np.ndarray,     # (S, K+1, D)
    dim_labels: list,
    slices: dict,
    save_path: str = None,
    config_dim: int = 16,
):
    """
    每个末端执行器一个子图, 各维度画折线.
    所有候选链半透明叠加, 最佳链加粗.
    """
    S, Kp1, D = all_chains.shape

    # 分组
    groups = {}
    group_colors = {}
    palette = {
        'LF': '#4c72b0',   # 蓝
        'RF': '#55a868',   # 绿
        'LH': '#c44e52',   # 红
        'RH': '#8172b2',   # 紫
        'Ct': '#937860',   # 棕
    }
    name_to_key = {
        'left_foot': 'LF', 'right_foot': 'RF',
        'left_hand': 'LH', 'right_hand': 'RH', 'contacts': 'Ct',
    }
    for key, abbr in name_to_key.items():
        s = slices[key]
        dims = list(range(s[0], s[1]))
        groups[abbr] = dims
        group_colors[abbr] = palette[abbr]

    num_groups = len(groups)
    fig, axes = plt.subplots(1, num_groups, figsize=(num_groups * 3.5, 4.5), sharey=False)
    if num_groups == 1:
        axes = [axes]

    waypoints = np.arange(Kp1)
    wp_labels = ['Start'] + [f'WP{i}' for i in range(1, Kp1 - 1)] + ['Goal']

    for ax, (grp_name, dims) in zip(axes, groups.items()):
        base_color = group_colors[grp_name]
        sub_labels = [dim_labels[d] for d in dims]

        # 为同组的各维度分配不同深浅
        n_dims = len(dims)
        if n_dims == 1:
            dim_colors = [base_color]
        else:
            from matplotlib.colors import to_rgba
            base_rgba = np.array(to_rgba(base_color))
            dim_colors = []
            for i in range(n_dims):
                factor = 0.4 + 0.6 * (i / max(n_dims - 1, 1))
                c = base_rgba.copy()
                c[:3] = c[:3] * factor + (1 - factor) * np.array([1, 1, 1])
                dim_colors.append(c)

        for di, (d, dc, sl) in enumerate(zip(dims, dim_colors, sub_labels)):
            # 所有候选链半透明
            for s in range(S):
                ax.plot(waypoints, all_chains[s, :, d],
                        color=dc, alpha=0.12, linewidth=0.8)
            # 最佳链加粗
            ax.plot(waypoints, best_chain[:, d],
                    color=dc, alpha=1.0, linewidth=2.2,
                    marker='o', markersize=4, label=sl, zorder=5)

        ax.set_xticks(waypoints)
        ax.set_xticklabels(wp_labels, fontsize=7, rotation=30)
        ax.set_title(grp_name, fontsize=12, fontweight='bold', color=base_color)
        ax.set_xlabel('Waypoint')
        ax.set_ylabel('Value')
        ax.legend(fontsize=6, frameon=False, ncol=max(1, n_dims // 3),
                  loc='upper left', borderaxespad=0.3)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    fig.suptitle(
        f'Joint Trajectory Along Chain ({Kp1-1} transitions, {S} candidates)',
        fontsize=14, fontweight='bold', y=1.04,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[Plot] Saved trajectory plot -> {save_path}")
    plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════
# Epoch Snapshot: 训练中定期采样评估
# ═════════════════════════════════════════════════════════════

def plot_epoch_snapshots(
    snapshot_dir: str,
    dim_labels: list,
    train_initial: np.ndarray,
    train_final: np.ndarray,
    save_path: str = None,
    config_dim: int = 16,
    contact_start: int = 12,
    max_dims: int = 6,
):
    """
    读取 snapshot_dir 下的 epoch_*.npz 文件, 绘制模型学习进度.

    每个 npz 包含 'initial' 和 'final' 数组 (生成样本, 原始空间).
    选择前 max_dims 个位置维度, 展示不同 epoch 生成分布的演变.
    """
    snap_files = sorted(Path(snapshot_dir).glob('epoch_*.npz'),
                        key=lambda p: int(p.stem.split('_')[1]))
    if not snap_files:
        print(f"[Plot] No snapshot files found in {snapshot_dir}")
        return None

    epochs = []
    gen_initials = []
    gen_finals = []
    for sf in snap_files:
        ep = int(sf.stem.split('_')[1])
        d = np.load(sf)
        epochs.append(ep)
        gen_initials.append(d['initial'])
        gen_finals.append(d['final'])

    n_snaps = len(epochs)
    dims_to_show = list(range(min(max_dims, config_dim)))

    # 每行 = 一个维度, 每列 = 一个 epoch snapshot
    n_dims = len(dims_to_show)
    fig, axes = plt.subplots(n_dims, n_snaps, figsize=(n_snaps * 2.5, n_dims * 2),
                             squeeze=False)

    fig.suptitle('Learning Progress — Initial Config Marginals', fontsize=14, fontweight='bold', y=1.03)

    for col, (ep, gen_ini) in enumerate(zip(epochs, gen_initials)):
        for row, d in enumerate(dims_to_show):
            ax = axes[row, col]
            tr = train_initial[:, d]
            ge = gen_ini[:, d]
            lo = min(tr.min(), ge.min())
            hi = max(tr.max(), ge.max())
            margin = (hi - lo) * 0.05
            bins = np.linspace(lo - margin, hi + margin, 30)
            ax.hist(tr, bins=bins, density=True, color='#4c72b0', alpha=0.5)
            ax.hist(ge, bins=bins, density=True, color='#dd8452', alpha=0.5)
            if col == 0:
                ax.set_ylabel(dim_labels[d], fontsize=8)
            if row == 0:
                ax.set_title(f'Epoch {ep}', fontsize=9, fontweight='bold')
            ax.tick_params(labelbottom=False, labelleft=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[Plot] Saved epoch snapshot plot -> {save_path}")
    plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Generate analysis plots")
    p.add_argument("--data", type=str, default=None,
                   help="Path to training dataset JSON (for marginal distribution)")
    p.add_argument("--samples", type=str, default=None,
                   help="Path to generated samples JSON (joint/forward/backward mode)")
    p.add_argument("--chain_output", type=str, default=None,
                   help="Path to chain output JSON")
    p.add_argument("--snapshot_dir", type=str, default=None,
                   help="Directory with epoch_*.npz snapshots")
    p.add_argument("--output_dir", type=str, default="visualization/figures")
    p.add_argument("--config_dim", type=int, default=16, help="Config dimension")
    p.add_argument("--use_orientation", action='store_true', default=False)
    p.add_argument("--plots", type=str, default="all",
                   help="Comma-separated: marginal,consistency,trajectory,snapshots,all")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(__file__).resolve().parent.parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dim = 22 if args.use_orientation else 16

    # 根据 config_dim 确定 slices
    if config_dim == 22:
        slices = {
            'left_foot': [0, 6], 'right_foot': [6, 12],
            'left_hand': [12, 15], 'right_hand': [15, 18],
            'contacts': [18, 22],
        }
        contact_start = 18
    else:
        slices = {
            'left_foot': [0, 3], 'right_foot': [3, 6],
            'left_hand': [6, 9], 'right_hand': [9, 12],
            'contacts': [12, 16],
        }
        contact_start = 12

    dim_labels = _get_dim_labels(config_dim, slices)
    plots = args.plots.split(',')
    do_all = 'all' in plots

    # ── 图2: 边际分布 ──
    if do_all or 'marginal' in plots:
        if args.data and args.samples:
            print("[Plot] Generating marginal distribution plots...")
            train_ini, train_fin = _load_training_data(args.data, args.use_orientation)
            gen_ini, gen_fin = _load_generated_samples(args.samples, slices, config_dim)

            plot_marginal_distributions(
                train_ini, gen_ini, dim_labels,
                title_prefix="Initial",
                save_path=str(out_dir / "fig2_marginal_initial.png"),
                config_dim=config_dim,
                contact_start=contact_start,
            )
            plot_marginal_distributions(
                train_fin, gen_fin, dim_labels,
                title_prefix="Final",
                save_path=str(out_dir / "fig2_marginal_final.png"),
                config_dim=config_dim,
                contact_start=contact_start,
            )
        else:
            print("[Skip] Marginal plot: need --data and --samples")

    # ── 图6 & 图7: Chain 分析 ──
    if do_all or 'consistency' in plots or 'trajectory' in plots:
        if args.chain_output:
            print("[Plot] Loading chain output...")
            best_chain, all_chains, K = _load_chain_output(
                args.chain_output, slices, config_dim
            )

            # 找最佳链的索引
            S = all_chains.shape[0]
            errors = []
            for s in range(S):
                err = 0
                for k in range(K - 1):
                    # 两端差异
                    err += np.linalg.norm(all_chains[s, k+1, :] - best_chain[k+1, :])
                errors.append(err)
            best_idx = int(np.argmin(errors))

            if do_all or 'consistency' in plots:
                print("[Plot] Generating chain consistency plot...")
                plot_chain_consistency(
                    all_chains, best_idx, dim_labels,
                    save_path=str(out_dir / "fig6_chain_consistency.png"),
                    config_dim=config_dim,
                )

            if do_all or 'trajectory' in plots:
                print("[Plot] Generating chain trajectory plot...")
                plot_chain_trajectory_lines(
                    best_chain, all_chains, dim_labels, slices,
                    save_path=str(out_dir / "fig7_chain_trajectory.png"),
                    config_dim=config_dim,
                )
        else:
            print("[Skip] Chain plots: need --chain_output")

    # ── Epoch Snapshots ──
    if do_all or 'snapshots' in plots:
        if args.snapshot_dir and args.data:
            print("[Plot] Generating epoch snapshot plot...")
            train_ini, train_fin = _load_training_data(args.data, args.use_orientation)
            plot_epoch_snapshots(
                args.snapshot_dir, dim_labels, train_ini, train_fin,
                save_path=str(out_dir / "fig_epoch_snapshots.png"),
                config_dim=config_dim,
                contact_start=contact_start,
            )
        else:
            print("[Skip] Snapshot plot: need --snapshot_dir and --data")

    print(f"\n[Done] All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
