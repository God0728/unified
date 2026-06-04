"""Configuration utilities for Mini VLA training scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into ``base`` and return ``base``."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        if path.suffix.lower() == ".json":
            return json.load(f)
    raise ValueError(f"unsupported config suffix: {path.suffix}")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse CLI overrides in dotted-key form, e.g. ``train.steps=100``."""
    result: dict[str, Any] = {}
    if not items:
        return result
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"override must be key=value, got {item}")
        key, raw_value = item.split("=", 1)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def load_config_with_overrides(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = load_config(path)
    return deep_update(config, parse_overrides(overrides))
