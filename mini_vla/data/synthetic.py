"""Synthetic pretraining data for Mini VLA.

This dataset is not meant to solve robotics. It provides a deterministic,
learnable Vision-Language-Action signal so the whole training stack can be
validated before moving to real robot datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticVLAConfig:
    num_samples: int = 10000
    image_size: int = 128
    action_dim: int = 7
    chunk_size: int = 8
    state_dim: int = 8
    seed: int = 0
    noise_std: float = 0.01


class SyntheticVLADataset(Dataset):
    """Generate simple colored-object manipulation samples.

    Each sample contains a square object in an RGB image, a language instruction
    that specifies a high-level operation and a continuous action chunk derived
    from object pose, color and instruction. The mapping is deterministic per
    index, making the dataset suitable for reproducibility tests.
    """

    colors = {
        "red": (1.0, 0.05, 0.05),
        "green": (0.05, 0.85, 0.10),
        "blue": (0.05, 0.20, 1.0),
        "yellow": (1.0, 0.85, 0.05),
    }
    objects = ["cube", "block", "peg", "disk"]
    tasks = [
        "pick the {color} {object}",
        "move the {color} {object} to the target zone",
        "push the {color} {object} forward",
        "place the {color} {object} into the tray",
    ]

    def __init__(self, config: SyntheticVLAConfig):
        self.config = config
        if config.action_dim < 7:
            raise ValueError("SyntheticVLADataset expects action_dim >= 7")

    def __len__(self) -> int:
        return self.config.num_samples

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.config.seed + idx * 9973)

    def __getitem__(self, idx: int) -> dict:
        rng = self._rng(idx)
        image_size = self.config.image_size
        color_name = list(self.colors.keys())[int(rng.integers(0, len(self.colors)))]
        object_name = self.objects[int(rng.integers(0, len(self.objects)))]
        task_id = int(rng.integers(0, len(self.tasks)))
        instruction = self.tasks[task_id].format(color=color_name, object=object_name)

        # Object and target positions in normalized coordinates.
        obj_xy = rng.uniform(0.18, 0.82, size=2).astype(np.float32)
        target_xy = rng.uniform(0.18, 0.82, size=2).astype(np.float32)
        gripper_open = float(rng.uniform(0.0, 1.0))
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        if self.config.state_dim > 0:
            base_state = np.array(
                [obj_xy[0], obj_xy[1], target_xy[0], target_xy[1], gripper_open, task_id / 3.0],
                dtype=np.float32,
            )
            state[: min(len(base_state), self.config.state_dim)] = base_state[: self.config.state_dim]
            if self.config.state_dim > len(base_state):
                state[len(base_state) :] = rng.normal(0, 0.05, size=self.config.state_dim - len(base_state))

        image = torch.zeros(3, image_size, image_size, dtype=torch.float32)
        image += 0.04
        # Draw target zone as a faint white cross.
        tx, ty = (target_xy * (image_size - 1)).astype(int)
        image[:, max(0, ty - 1) : min(image_size, ty + 2), :] += 0.08
        image[:, :, max(0, tx - 1) : min(image_size, tx + 2)] += 0.08
        # Draw object square.
        ox, oy = (obj_xy * (image_size - 1)).astype(int)
        half = int(rng.integers(max(3, image_size // 32), max(5, image_size // 14)))
        color = torch.tensor(self.colors[color_name], dtype=torch.float32).view(3, 1, 1)
        y0, y1 = max(0, oy - half), min(image_size, oy + half)
        x0, x1 = max(0, ox - half), min(image_size, ox + half)
        image[:, y0:y1, x0:x1] = color
        image.clamp_(0.0, 1.0)

        actions = self._build_actions(obj_xy, target_xy, color_name, task_id, gripper_open, rng)
        return {
            "images": image,
            "state": torch.tensor(state, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "instruction": instruction,
            "metadata": {
                "idx": idx,
                "color": color_name,
                "object": object_name,
                "task_id": task_id,
            },
        }

    def _build_actions(
        self,
        obj_xy: np.ndarray,
        target_xy: np.ndarray,
        color_name: str,
        task_id: int,
        gripper_open: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        color_index = list(self.colors.keys()).index(color_name)
        delta = target_xy - obj_xy
        base = np.zeros(self.config.action_dim, dtype=np.float32)
        base[0] = delta[0]
        base[1] = delta[1]
        base[2] = 0.05 + 0.05 * color_index
        base[3] = np.sin(task_id)
        base[4] = np.cos(task_id)
        base[5] = float(color_index) / max(1, len(self.colors) - 1)
        base[6] = 1.0 - gripper_open if task_id in {0, 3} else gripper_open
        if self.config.action_dim > 7:
            base[7:] = rng.normal(0, 0.05, size=self.config.action_dim - 7)

        chunk = np.repeat(base[None, :], self.config.chunk_size, axis=0)
        ramp = np.linspace(0.0, 1.0, self.config.chunk_size, dtype=np.float32)[:, None]
        chunk[:, 0:2] = ramp * delta[None, :]
        chunk += rng.normal(0, self.config.noise_std, size=chunk.shape).astype(np.float32)
        return chunk.astype(np.float32)
