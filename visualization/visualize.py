"""
Visualization Tools for Unified Transition Diffusion Model
============================================================
Provides 3D visualization for:
  - Single transition (initial -> final)
  - Chain planning (A -> c1 -> c2 -> ... -> D)
  - Multi-sample comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from typing import List, Optional, Dict
import json


# Color scheme
COLORS = {
    'LF': '#2196F3',  # Blue
    'RF': '#4CAF50',  # Green
    'LH': '#FF9800',  # Orange
    'RH': '#E91E63',  # Pink
}

EFFECTOR_NAMES = {
    'LF': 'Left Foot',
    'RF': 'Right Foot',
    'LH': 'Left Hand',
    'RH': 'Right Hand',
}


def extract_positions(config_dict: dict) -> dict:
    """Extract 3D positions from config dict."""
    poses = config_dict['poses']
    return {
        'LF': np.array(poses['LF'][:3]),
        'RF': np.array(poses['RF'][:3]),
        'LH': np.array(poses['LH'][:3]),
        'RH': np.array(poses['RH'][:3]),
    }


def extract_contacts(config_dict: dict) -> dict:
    """Extract contact states."""
    contacts = config_dict['contacts']
    names = ['LF', 'RF', 'LH', 'RH']
    return {name: contacts[i] > 0.5 for i, name in enumerate(names)}


def plot_config(ax, config_dict: dict, alpha: float = 1.0, marker_size: int = 100,
                label_prefix: str = "", show_contacts: bool = True):
    """Plot a single configuration on a 3D axis."""
    positions = extract_positions(config_dict)
    contacts = extract_contacts(config_dict)

    for name, pos in positions.items():
        color = COLORS[name]
        is_contact = contacts[name]
        marker = 's' if is_contact else 'o'  # square for contact, circle for free
        edge_color = 'black' if is_contact else 'gray'

        ax.scatter(pos[0], pos[1], pos[2],
                   c=color, s=marker_size, marker=marker,
                   alpha=alpha, edgecolors=edge_color, linewidths=1.5,
                   label=f"{label_prefix}{EFFECTOR_NAMES[name]}" if label_prefix else None)

    # Draw body frame (connect effectors)
    lf, rf = positions['LF'], positions['RF']
    lh, rh = positions['LH'], positions['RH']

    # Feet connection
    ax.plot([lf[0], rf[0]], [lf[1], rf[1]], [lf[2], rf[2]],
            'k-', alpha=alpha * 0.3, linewidth=1)
    # Hands connection
    ax.plot([lh[0], rh[0]], [lh[1], rh[1]], [lh[2], rh[2]],
            'k-', alpha=alpha * 0.3, linewidth=1)
    # Left side
    ax.plot([lf[0], lh[0]], [lf[1], lh[1]], [lf[2], lh[2]],
            'k--', alpha=alpha * 0.2, linewidth=0.8)
    # Right side
    ax.plot([rf[0], rh[0]], [rf[1], rh[1]], [rf[2], rh[2]],
            'k--', alpha=alpha * 0.2, linewidth=0.8)


def plot_transition(initial_dict: dict, final_dict: dict,
                    title: str = "Transition", save_path: Optional[str] = None):
    """Plot a single transition: initial -> final."""
    fig = plt.figure(figsize=(14, 6))

    # Initial state
    ax1 = fig.add_subplot(131, projection='3d')
    plot_config(ax1, initial_dict, alpha=1.0, label_prefix="")
    ax1.set_title("Initial State")
    _set_axes(ax1)

    # Final state
    ax2 = fig.add_subplot(132, projection='3d')
    plot_config(ax2, final_dict, alpha=1.0)
    ax2.set_title("Final State")
    _set_axes(ax2)

    # Overlay
    ax3 = fig.add_subplot(133, projection='3d')
    plot_config(ax3, initial_dict, alpha=0.3, marker_size=60)
    plot_config(ax3, final_dict, alpha=1.0, marker_size=100)

    # Draw arrows from initial to final positions
    init_pos = extract_positions(initial_dict)
    final_pos = extract_positions(final_dict)
    for name in ['LF', 'RF', 'LH', 'RH']:
        p0 = init_pos[name]
        p1 = final_pos[name]
        ax3.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                 color=COLORS[name], alpha=0.6, linewidth=1.5, linestyle='--')

    ax3.set_title("Overlay (faded=initial)")
    _set_axes(ax3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved: {save_path}")
    plt.close()


def plot_chain(chain_configs: List[dict], title: str = "Chain Planning",
               save_path: Optional[str] = None):
    """Plot a chain of waypoints: A -> c1 -> c2 -> ... -> D."""
    K = len(chain_configs)
    ncols = min(K, 4)
    nrows = (K + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows + 1))

    for i, config_dict in enumerate(chain_configs):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        plot_config(ax, config_dict, alpha=1.0)

        if i == 0:
            label = "Start (A)"
        elif i == K - 1:
            label = "Goal (D)"
        else:
            label = f"Waypoint {i}"
        ax.set_title(label, fontsize=11)
        _set_axes(ax)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved: {save_path}")
    plt.close()


def plot_chain_trajectory(chain_configs: List[dict], title: str = "Chain Trajectory",
                          save_path: Optional[str] = None):
    """Plot all waypoints overlaid on a single 3D plot with trajectory lines."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    K = len(chain_configs)
    alphas = np.linspace(0.3, 1.0, K)

    for i, config_dict in enumerate(chain_configs):
        positions = extract_positions(config_dict)
        contacts = extract_contacts(config_dict)
        alpha = alphas[i]
        size = 50 + i * 30

        for name, pos in positions.items():
            color = COLORS[name]
            is_contact = contacts[name]
            marker = 's' if is_contact else 'o'
            ax.scatter(pos[0], pos[1], pos[2],
                       c=color, s=size, marker=marker,
                       alpha=alpha, edgecolors='black', linewidths=1)

    # Draw trajectory lines for each effector
    for name in ['LF', 'RF', 'LH', 'RH']:
        trajectory = np.array([extract_positions(c)[name] for c in chain_configs])
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=COLORS[name], alpha=0.6, linewidth=2, linestyle='-',
                label=EFFECTOR_NAMES[name])

    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(title, fontsize=14, fontweight='bold')
    _set_axes(ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved: {save_path}")
    plt.close()


def plot_multi_samples(samples: List[dict], condition_config: Optional[dict] = None,
                       title: str = "Multi-Sample Generation",
                       save_path: Optional[str] = None):
    """Plot multiple generated samples (e.g., multiple finals given one initial)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot condition (if any) with strong emphasis
    if condition_config is not None:
        plot_config(ax, condition_config, alpha=1.0, marker_size=200)

    # Plot all samples with lower alpha
    for sample in samples:
        plot_config(ax, sample, alpha=0.3, marker_size=50)

    ax.set_title(title, fontsize=14, fontweight='bold')
    _set_axes(ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved: {save_path}")
    plt.close()


def _set_axes(ax):
    """Set consistent axis properties."""
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True, alpha=0.3)

    # Try to set equal aspect ratio
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass
