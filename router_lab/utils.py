"""Shared helpers: config loading, device/seed setup, paths."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Config:
    raw: dict

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)


def load_config(path: str | os.PathLike = "config.yaml") -> Config:
    """Read the config; relative `paths` are resolved against the config file's
    directory, so every script works from any working directory."""
    path = Path(path)
    with open(path, "r") as f:
        cfg = Config(yaml.safe_load(f))
    for key, value in cfg.raw.get("paths", {}).items():
        if not Path(value).is_absolute():
            cfg.raw["paths"][key] = str((path.parent / value).resolve())
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:                       # torch is optional here: nothing in the paper's
        import torch           # pipeline needs it except SkillAggregation
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str = "auto") -> str:
    import torch
    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_dtype(device: str):
    """Pick a sensible dtype for the given device."""
    import torch
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def ensure_dirs(cfg: Config) -> tuple[Path, Path]:
    results = Path(cfg["paths"]["results_dir"])
    cache = Path(cfg["paths"]["cache_dir"])
    results.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return results, cache
