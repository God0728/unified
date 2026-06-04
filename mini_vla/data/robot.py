"""Robot dataset adapters for Mini VLA fine-tuning.

The adapters emit the same schema as the synthetic pretraining dataset:
``images``, ``state``, ``actions`` and ``instruction``. This lets pretraining
and fine-tuning reuse the same collator, model and trainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class RobotDatasetConfig:
    dataset_type: str = "local_jsonl"
    path: str | None = None
    name: str | None = None
    split: str = "train"
    image_size: int = 128
    action_dim: int = 7
    chunk_size: int = 8
    state_dim: int = 8
    image_column: str = "image"
    instruction_column: str = "instruction"
    action_column: str = "actions"
    state_column: str = "state"
    max_samples: int | None = None


def _resize_image_to_tensor(image: Any, image_size: int) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().float()
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
            pass
        elif tensor.ndim == 3 and tensor.shape[-1] in {1, 3}:
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"unsupported image tensor shape: {tuple(tensor.shape)}")
        if tensor.max() > 2:
            tensor = tensor / 255.0
        if tensor.shape[-2:] != (image_size, image_size):
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor.clamp(0.0, 1.0)

    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, dict) and "path" in image:
        image = Image.open(image["path"])
    elif isinstance(image, dict) and "bytes" in image:
        import io

        image = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    if not isinstance(image, Image.Image):
        raise ValueError(f"unsupported image value type: {type(image)!r}")
    image = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _coerce_vector(value: Any, dim: int) -> torch.Tensor:
    arr = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    out = np.zeros(dim, dtype=np.float32)
    if dim > 0 and arr.size > 0:
        out[: min(dim, arr.size)] = arr[:dim]
    return torch.from_numpy(out)


def _coerce_actions(value: Any, chunk_size: int, action_dim: int) -> torch.Tensor:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"actions must be 1D or 2D, got shape {arr.shape}")
    out = np.zeros((chunk_size, action_dim), dtype=np.float32)
    rows = min(chunk_size, arr.shape[0])
    cols = min(action_dim, arr.shape[1])
    out[:rows, :cols] = arr[:rows, :cols]
    if rows < chunk_size and rows > 0:
        out[rows:] = out[rows - 1]
    return torch.from_numpy(out)


class LocalJsonlRobotDataset(Dataset):
    """Fine-tuning dataset from JSONL manifest records.

    Each line should contain at least ``image_path`` or ``image`` and ``actions``.
    Optional keys are ``instruction`` and ``state``. Relative image paths are
    resolved relative to the manifest file.
    """

    def __init__(self, config: RobotDatasetConfig):
        if config.path is None:
            raise ValueError("local_jsonl dataset requires data.path")
        self.config = config
        self.path = Path(config.path)
        self.root = self.path.parent
        with self.path.open("r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if config.max_samples is not None:
            self.records = self.records[: config.max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image_value = record.get("image_path", record.get("image", record.get(self.config.image_column)))
        if isinstance(image_value, str) and not image_value.startswith(("/", "http://", "https://")):
            image_value = self.root / image_value
        instruction = record.get(
            self.config.instruction_column,
            record.get("instruction", record.get("language", record.get("task", "perform the task"))),
        )
        actions = record.get(self.config.action_column)
        if actions is None:
            actions = record.get("actions", record.get("action"))
        if actions is None:
            raise ValueError("record is missing actions/action field")
        state = record.get(self.config.state_column)
        if state is None:
            state = record.get("state", record.get("observation.state"))
        return {
            "images": _resize_image_to_tensor(image_value, self.config.image_size),
            "state": _coerce_vector(state, self.config.state_dim),
            "actions": _coerce_actions(actions, self.config.chunk_size, self.config.action_dim),
            "instruction": str(instruction),
            "metadata": {"idx": idx, "source": str(self.path)},
        }


class HuggingFaceRobotDataset(Dataset):
    """Adapter for Hugging Face datasets, including LeRobot-style datasets."""

    def __init__(self, config: RobotDatasetConfig):
        if config.name is None:
            raise ValueError("hf_robot dataset requires data.name, e.g. lerobot/pusht")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Fine-tuning from Hugging Face datasets requires `datasets`. "
                "Install it with `pip install datasets`."
            ) from exc
        self.config = config
        self.dataset = load_dataset(config.name, split=config.split)
        if config.max_samples is not None:
            self.dataset = self.dataset.select(range(min(config.max_samples, len(self.dataset))))

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_first_existing(self, row: dict[str, Any], candidates: list[str]) -> Any:
        for key in candidates:
            if key in row:
                return row[key]
        dotted = [key for key in candidates if "." in key]
        for key in dotted:
            cursor: Any = row
            ok = True
            for part in key.split("."):
                if isinstance(cursor, dict) and part in cursor:
                    cursor = cursor[part]
                else:
                    ok = False
                    break
            if ok:
                return cursor
        return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataset[idx]
        image = self._get_first_existing(
            row,
            [
                self.config.image_column,
                "observation.image",
                "observation.images.front",
                "observation.images.main",
                "image",
            ],
        )
        instruction = self._get_first_existing(
            row,
            [self.config.instruction_column, "task", "language_instruction", "instruction"],
        )
        actions = self._get_first_existing(row, [self.config.action_column, "action", "actions"])
        state = self._get_first_existing(row, [self.config.state_column, "observation.state", "state"])
        if image is None:
            raise ValueError(
                "could not find an image column; set data.image_column in the config"
            )
        if actions is None:
            raise ValueError(
                "could not find an action column; set data.action_column in the config"
            )
        return {
            "images": _resize_image_to_tensor(image, self.config.image_size),
            "state": _coerce_vector(state, self.config.state_dim),
            "actions": _coerce_actions(actions, self.config.chunk_size, self.config.action_dim),
            "instruction": str(instruction or "perform the task"),
            "metadata": {"idx": idx, "source": self.config.name or "hf_robot"},
        }


def build_robot_dataset(config: dict[str, Any], split: str = "train") -> Dataset:
    data_cfg = config.get("data", {})
    split_cfg = data_cfg.get(split, {})
    model_cfg = config.get("model", {})
    merged = {**data_cfg, **split_cfg}
    dataset_type = str(merged.get("type", data_cfg.get("type", "local_jsonl"))).lower()
    cfg = RobotDatasetConfig(
        dataset_type=dataset_type,
        path=merged.get("path"),
        name=merged.get("name"),
        split=str(merged.get("split", split)),
        image_size=int(model_cfg.get("image_size", merged.get("image_size", 128))),
        action_dim=int(model_cfg.get("action_dim", merged.get("action_dim", 7))),
        chunk_size=int(model_cfg.get("chunk_size", merged.get("chunk_size", 8))),
        state_dim=int(model_cfg.get("state_dim", merged.get("state_dim", 8))),
        image_column=str(merged.get("image_column", "image")),
        instruction_column=str(merged.get("instruction_column", "instruction")),
        action_column=str(merged.get("action_column", "actions")),
        state_column=str(merged.get("state_column", "state")),
        max_samples=merged.get("max_samples"),
    )
    if dataset_type == "local_jsonl":
        return LocalJsonlRobotDataset(cfg)
    if dataset_type in {"hf_robot", "lerobot", "huggingface"}:
        return HuggingFaceRobotDataset(cfg)
    raise ValueError(f"unsupported fine-tune dataset type: {dataset_type}")
