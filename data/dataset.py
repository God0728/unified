"""
Unified Transition Dataset
==========================
Loads transition pairs (initial_config, final_config) from JSON files.

Data format (from balance_dataset JSON):
  - 'poses': list of 28D vectors, rows [2i, 2i+1] = (initial, final) pair
    Each 28D = [LF(7D), RF(7D), LH(7D), RH(7D)]
    where 7D = [px, py, pz, qw, qx, qy, qz]
  - 'contacts': list of 4D vectors [LF, RF, LH, RH] booleans

We convert to the model's representation:
  use_orientation=True (22D):
    - Feet: 6D each (px, py, pz, roll, pitch, yaw) from quaternion
    - Hands: 3D each (px, py, pz) - point contact
    - Contacts: 4D booleans
    Total config_dim = 6 + 6 + 3 + 3 + 4 = 22D
  use_orientation=False (16D):
    - All end-effectors: 3D each (px, py, pz)
    - Contacts: 4D booleans
    Total config_dim = 3 + 3 + 3 + 3 + 4 = 16D
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple
import math


# ============================================================
# Quaternion to Euler conversion
# ============================================================

def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz


def pose_28d_to_config(pose_28d: np.ndarray, contacts_4d: np.ndarray, use_orientation: bool = True) -> np.ndarray:
    """
    Convert 28D pose + 4D contacts to config.

    28D = [LF(7D), RF(7D), LH(7D), RH(7D)]
    use_orientation=True  -> 22D = [LF_pos(3)+LF_euler(3), RF_pos(3)+RF_euler(3), LH_pos(3), RH_pos(3), contacts(4)]
    use_orientation=False -> 16D = [LF_pos(3), RF_pos(3), LH_pos(3), RH_pos(3), contacts(4)]
    """
    # Left foot
    lf_pos = pose_28d[0:3]

    # Right foot
    rf_pos = pose_28d[7:10]

    # Left hand: pos only
    lh_pos = pose_28d[14:17]

    # Right hand: pos only
    rh_pos = pose_28d[21:24]

    # Contacts
    contacts = contacts_4d.astype(np.float32)

    if use_orientation:
        lf_quat = pose_28d[3:7]  # qw, qx, qy, qz
        lf_euler = np.array(quat_to_euler(*lf_quat))
        lf = np.concatenate([lf_pos, lf_euler])  # 6D

        rf_quat = pose_28d[10:14]
        rf_euler = np.array(quat_to_euler(*rf_quat))
        rf = np.concatenate([rf_pos, rf_euler])  # 6D
    else:
        lf = lf_pos  # 3D
        rf = rf_pos  # 3D

    config = np.concatenate([lf, rf, lh_pos, rh_pos, contacts])
    return config.astype(np.float32)


def config_to_pose_28d(config: np.ndarray, use_orientation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert config back to 28D pose + 4D contacts.

    use_orientation=True:  22D = [LF(6D), RF(6D), LH(3D), RH(3D), contacts(4D)]
    use_orientation=False: 16D = [LF(3D), RF(3D), LH(3D), RH(3D), contacts(4D)]
    28D = [LF(7D), RF(7D), LH(7D), RH(7D)]

    For hands (and feet when use_orientation=False), quaternion is set to identity (1,0,0,0).
    """
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

    if use_orientation:
        lf_pos = config[0:3]
        lf_euler = config[3:6]
        lf_quat = np.array(euler_to_quat(*lf_euler))
        lf_7d = np.concatenate([lf_pos, lf_quat])

        rf_pos = config[6:9]
        rf_euler = config[9:12]
        rf_quat = np.array(euler_to_quat(*rf_euler))
        rf_7d = np.concatenate([rf_pos, rf_quat])

        lh_pos = config[12:15]
        rh_pos = config[15:18]
        contacts = config[18:22]
    else:
        lf_pos = config[0:3]
        lf_7d = np.concatenate([lf_pos, identity_quat])

        rf_pos = config[3:6]
        rf_7d = np.concatenate([rf_pos, identity_quat])

        lh_pos = config[6:9]
        rh_pos = config[9:12]
        contacts = config[12:16]

    lh_7d = np.concatenate([lh_pos, identity_quat])
    rh_7d = np.concatenate([rh_pos, identity_quat])

    pose_28d = np.concatenate([lf_7d, rf_7d, lh_7d, rh_7d])
    return pose_28d.astype(np.float32), contacts.astype(np.float32)


# ============================================================
# Normalizer
# ============================================================

class TransitionNormalizer:
    """Normalizes transition data using shared statistics across all configs."""

    def __init__(self):
        self.config_mean: Optional[np.ndarray] = None
        self.config_std: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, all_configs: np.ndarray, num_contact_dims: int = 4):
        """
        Fit normalizer on all configurations (both initial and final pooled).
        Only normalizes pose dimensions; last num_contact_dims are left as-is (0/1).

        Args:
            all_configs: (N, config_dim) array
            num_contact_dims: number of trailing contact dimensions to skip
        """
        self.config_mean = all_configs.mean(axis=0)
        self.config_std = all_configs.std(axis=0)
        self.config_std = np.clip(self.config_std, 1e-6, None)

        # Don't normalize contact dims: set mean=0, std=1 so they pass through unchanged
        if num_contact_dims > 0:
            self.config_mean[-num_contact_dims:] = 0.0
            self.config_std[-num_contact_dims:] = 1.0

        self.fitted = True

    def normalize(self, config: np.ndarray) -> np.ndarray:
        assert self.fitted, "Normalizer not fitted"
        return (config - self.config_mean) / self.config_std

    def denormalize(self, config: np.ndarray) -> np.ndarray:
        assert self.fitted, "Normalizer not fitted"
        return config * self.config_std + self.config_mean

    def state_dict(self) -> Dict:
        return {
            'config_mean': self.config_mean,
            'config_std': self.config_std,
            'fitted': self.fitted,
        }

    def load_state_dict(self, state: Dict):
        self.config_mean = state['config_mean']
        self.config_std = state['config_std']
        self.fitted = state['fitted']

class LimitsNormalizer:
    """
    LimitsNormalizer: maps [xmin, xmax] to [-1, 1].
    Contact dimensions (last num_contact_dims) are left unchanged (0/1).
    """

    def __init__(self):
        self.mins: Optional[np.ndarray] = None
        self.maxs: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, all_configs: np.ndarray, num_contact_dims: int = 4):
        """
        Fit normalizer: compute per-dimension min/max.
        Contact dims are skipped (kept as 0/1).

        Args:
            all_configs: (N, config_dim) array
            num_contact_dims: number of trailing contact dimensions to skip
        """
        self.mins = all_configs.min(axis=0)
        self.maxs = all_configs.max(axis=0)

        # Avoid division by zero for constant dimensions
        range_vals = self.maxs - self.mins
        range_vals = np.clip(range_vals, 1e-6, None)
        self.maxs = self.mins + range_vals

        # Don't normalize contact dims: set mins=0, maxs=1 so (x-0)/(1-0)=x, 2*x-1 maps 0->-1, 1->1
        if num_contact_dims > 0:
            self.mins[-num_contact_dims:] = 0.0
            self.maxs[-num_contact_dims:] = 1.0

        self.fitted = True

    def normalize(self, config: np.ndarray) -> np.ndarray:
        """Map [xmin, xmax] -> [-1, 1]."""
        assert self.fitted, "Normalizer not fitted"
        # [0, 1]
        x = (config - self.mins) / (self.maxs - self.mins)
        # [-1, 1]
        x = 2 * x - 1
        return x

    def denormalize(self, config: np.ndarray) -> np.ndarray:
        """Map [-1, 1] -> [xmin, xmax]."""
        assert self.fitted, "Normalizer not fitted"
        # [-1, 1] -> [0, 1]
        x = (config + 1) / 2.0
        return x * (self.maxs - self.mins) + self.mins

    def state_dict(self) -> Dict:
        return {
            'mins': self.mins,
            'maxs': self.maxs,
            'fitted': self.fitted,
        }

    def load_state_dict(self, state: Dict):
        self.mins = state['mins']
        self.maxs = state['maxs']
        self.fitted = state['fitted']
# ============================================================
# Dataset
# ============================================================

class UnifiedTransitionDataset(Dataset):
    """
    Dataset of (initial_config, final_config) transition pairs.

    Handles the actual data format:
      - JSON with 'poses' (list of 28D) and 'contacts' (list of 4D)
      - Rows [2i, 2i+1] = (initial, final) pair

    Converts to 22D representation internally.
    """

    def __init__(
        self,
        data_path: str,
        normalizer: Optional[LimitsNormalizer] = None,
        normalize: bool = True,
        augment: bool = True,
        use_orientation: bool = True,
        clean_non_flip: bool = True,
    ):
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.augment = augment
        self.use_orientation = use_orientation
        self.clean_non_flip = clean_non_flip

        # Load raw data
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)

        # Parse based on format
        if isinstance(raw_data, dict) and 'poses' in raw_data:
            # New format: {poses: [...], contacts: [...]}
            self._load_flat_format(raw_data)
        elif isinstance(raw_data, list):
            # Old format: [{initial: {...}, final: {...}}, ...]
            self._load_pair_format(raw_data)
        else:
            raise ValueError(f"Unknown data format in {data_path}")

        # ---- Data cleaning: enforce single-effector change prior ----
        # SEIKO data is supposed to satisfy "exactly one effector flips contact
        # and only that effector's pose changes". In practice the other 3
        # effectors drift by ~mm due to optimization noise. We force them to be
        # byte-identical between initial and final so the supervision signal
        # for the "non-changed effector" prior is clean.
        if self.clean_non_flip:
            self._clean_non_flip_effectors()

        # Pool all configs for normalization
        all_configs = np.concatenate([self.initials, self.finals], axis=0)

        # Fit or use provided normalizer
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = LimitsNormalizer()
            if self.normalize:
                self.normalizer.fit(all_configs)

        # Normalize
        if self.normalize and self.normalizer.fitted:
            self.initials_norm = self.normalizer.normalize(self.initials)
            self.finals_norm = self.normalizer.normalize(self.finals)
        else:
            self.initials_norm = self.initials.copy()
            self.finals_norm = self.finals.copy()

        # Data augmentation
        if self.augment:
            self._augment_data()

        # Convert to tensors
        self.initials_tensor = torch.tensor(self.initials_norm, dtype=torch.float32)
        self.finals_tensor = torch.tensor(self.finals_norm, dtype=torch.float32)

        print(f"[Dataset] Loaded {len(self)} transition pairs from {self.data_path.name}")
        print(f"  Config dim: {self.initials_tensor.shape[1]}")
        print(f"  Original pairs: {len(self.initials)//2 if self.augment else len(self.initials)}, "
              f"After augmentation: {len(self)}")

    def _load_flat_format(self, raw_data: dict):
        """Load from flat format: poses[2i]/poses[2i+1] = initial/final."""
        poses = raw_data['poses']
        contacts = raw_data['contacts']
        num_pairs = raw_data.get('num_pairs', len(poses) // 2)

        self.initials = []
        self.finals = []

        for i in range(num_pairs):
            pose_init = np.array(poses[2 * i], dtype=np.float32)
            pose_final = np.array(poses[2 * i + 1], dtype=np.float32)
            contact_init = np.array(contacts[2 * i], dtype=np.float32)
            contact_final = np.array(contacts[2 * i + 1], dtype=np.float32)

            config_init = pose_28d_to_config(pose_init, contact_init, self.use_orientation)
            config_final = pose_28d_to_config(pose_final, contact_final, self.use_orientation)

            self.initials.append(config_init)
            self.finals.append(config_final)

        self.initials = np.stack(self.initials)
        self.finals = np.stack(self.finals)

    def _load_pair_format(self, raw_data: list):
        """Load from pair format: [{initial: {...}, final: {...}}, ...]."""
        self.initials = []
        self.finals = []

        for pair in raw_data:
            # Handle nested dict format
            init_data = pair['initial']
            final_data = pair['final']

            if 'poses' in init_data:
                # Nested format with poses dict
                init_poses = init_data['poses']
                init_contacts = np.array(init_data['contacts'], dtype=np.float32)

                lf = np.array(init_poses['LF'], dtype=np.float32)
                rf = np.array(init_poses['RF'], dtype=np.float32)
                lh = np.array(init_poses['LH'], dtype=np.float32)
                rh = np.array(init_poses['RH'], dtype=np.float32)
                pose_init = np.concatenate([lf, rf, lh, rh])

                final_poses = final_data['poses']
                final_contacts = np.array(final_data['contacts'], dtype=np.float32)
                lf = np.array(final_poses['LF'], dtype=np.float32)
                rf = np.array(final_poses['RF'], dtype=np.float32)
                lh = np.array(final_poses['LH'], dtype=np.float32)
                rh = np.array(final_poses['RH'], dtype=np.float32)
                pose_final = np.concatenate([lf, rf, lh, rh])
            else:
                raise ValueError("Unknown pair format")

            config_init = pose_28d_to_config(pose_init, init_contacts, self.use_orientation)
            config_final = pose_28d_to_config(pose_final, final_contacts, self.use_orientation)

            self.initials.append(config_init)
            self.finals.append(config_final)

        self.initials = np.stack(self.initials)
        self.finals = np.stack(self.finals)

    def _augment_data(self):
        """Add reverse transitions as augmentation."""
        reverse_initials = self.finals_norm.copy()
        reverse_finals = self.initials_norm.copy()
        self.initials_norm = np.concatenate([self.initials_norm, reverse_initials], axis=0)
        self.finals_norm = np.concatenate([self.finals_norm, reverse_finals], axis=0)

    def _clean_non_flip_effectors(self):
        """
        Force non-flipping effectors to be byte-identical between initial and final.

        For each pair, identify which effector flipped its contact bit. For all
        OTHER effectors, copy initial[..effector_dims..] -> final[..effector_dims..].
        Pairs with not-exactly-1 contact flip are kept as-is (rare; data is
        supposed to be single-flip but defensive).

        Stats are printed so users can see how much drift was removed.
        """
        if self.use_orientation:
            # 22D: LF[0:6], RF[6:12], LH[12:15], RH[15:18], contacts[18:22]
            eff_slices = [(0, 6), (6, 12), (12, 15), (15, 18)]
            contact_start = 18
        else:
            # 16D: LF[0:3], RF[3:6], LH[6:9], RH[9:12], contacts[12:16]
            eff_slices = [(0, 3), (3, 6), (6, 9), (9, 12)]
            contact_start = 12

        N = self.initials.shape[0]
        ci = self.initials[:, contact_start:contact_start + 4]
        cf = self.finals[:, contact_start:contact_start + 4]
        flip_mask = (ci != cf)                 # (N, 4)
        num_flips = flip_mask.sum(axis=1)      # (N,)
        flip_idx = flip_mask.argmax(axis=1)    # (N,) — only valid where num_flips == 1

        # Measure pre-clean drift on non-flipping effectors (for logging)
        pre_drift_max = 0.0
        pre_drift_mean_list = []
        cleaned_count = 0

        for n in range(N):
            if num_flips[n] != 1:
                continue
            cleaned_count += 1
            for e in range(4):
                if e == flip_idx[n]:
                    continue
                s, end = eff_slices[e]
                d = np.linalg.norm(self.finals[n, s:end] - self.initials[n, s:end])
                pre_drift_mean_list.append(d)
                if d > pre_drift_max:
                    pre_drift_max = d
                # Force final == initial on this effector's pose dims
                self.finals[n, s:end] = self.initials[n, s:end]

        if pre_drift_mean_list:
            pre_drift_mean = float(np.mean(pre_drift_mean_list))
            pre_drift_p99 = float(np.percentile(pre_drift_mean_list, 99))
        else:
            pre_drift_mean = 0.0
            pre_drift_p99 = 0.0

        skipped = N - cleaned_count
        print(f"[Dataset] clean_non_flip: cleaned {cleaned_count}/{N} pairs "
              f"(skipped {skipped} non-single-flip pairs); "
              f"pre-clean drift on non-flip effectors -> "
              f"mean={pre_drift_mean:.4f} p99={pre_drift_p99:.4f} max={pre_drift_max:.4f}")

    def __len__(self):
        return len(self.initials_norm)

    def __getitem__(self, idx):
        return self.initials_tensor[idx], self.finals_tensor[idx]


def create_dataloader(
    data_path: str,
    batch_size: int = 256,
    normalize: bool = True,
    augment: bool = True,
    use_orientation: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    clean_non_flip: bool = True,
) -> Tuple[DataLoader, UnifiedTransitionDataset]:
    """Create a DataLoader for the unified transition dataset."""
    dataset = UnifiedTransitionDataset(
        data_path=data_path,
        normalize=normalize,
        augment=augment,
        use_orientation=use_orientation,
        clean_non_flip=clean_non_flip,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True if len(dataset) > batch_size else False,
    )
    return dataloader, dataset
