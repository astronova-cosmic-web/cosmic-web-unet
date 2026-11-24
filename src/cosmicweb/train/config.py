"""Experiment configuration: one YAML file per experiment, plus command-line overrides.

Every key has a default (``DEFAULTS``); an experiment file only lists what it
changes. Unknown keys are errors, so a typo cannot silently fall back to a
default. Overrides use dotted keys, e.g. ``train.seed=1`` or ``data.noise=low``;
values are parsed as YAML (``1e-3`` → float, ``null`` → None, ``[1, 2]`` → list).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DEFAULTS", "load_experiment", "apply_overrides", "run_name"]

DEFAULTS: dict[str, Any] = {
    "data": {
        "config": "data.yaml",  # data pipeline config, relative to this experiment file
        "dataset_root": None,  # overrides dataset_root in the data config
        "root": None,  # built dataset folder; None = the data config's output_dir
        "noise": "high",
        "mask": "mid",
        "train": {"mode": "random", "patch_size": 256, "crops_per_map": 4, "augment": True,
                  "resample_each_epoch": True},
        "val": {"mode": "tiles", "patch_size": 256, "crops_per_map": 4, "augment": False,
                "resample_each_epoch": False},  # val must keep augment/resample off (checked)
        "batch_size": 16,
        "num_workers": 2,
    },
    "model": {"base_channels": 32, "depth": 4, "norm": "batch", "padding_mode": "zeros"},
    "loss": {"name": "mse"},
    "optim": {"name": "adamw", "lr": 1.0e-3, "weight_decay": 0.0},
    "schedule": {"name": "cosine", "min_lr": 1.0e-5},
    "train": {
        "epochs": 30,
        "seed": 0,
        "amp": True,  # mixed precision on CUDA; ignored on CPU
        "grad_clip": None,  # max gradient norm, or None
        "early_stopping_patience": None,  # epochs without val improvement, or None
        "deterministic": False,  # slower; see docs: bitwise GPU determinism is not guaranteed
        "limit_train_batches": None,  # for smoke tests and pilot runs
        "limit_val_batches": None,
    },
    "output": {"runs_dir": "runs", "name": None, "tensorboard": True},
}


def _merge(base: dict, update: dict, where: str) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"unknown key {where}{key!r}; allowed: {sorted(base)}")
        if isinstance(base[key], dict) and key not in ("loss",):
            if not isinstance(value, dict):
                raise ValueError(f"{where}{key} must be a mapping")
            out[key] = _merge(base[key], value, f"{where}{key}.")
        elif isinstance(base[key], float) and isinstance(value, (str, int)) and not isinstance(value, bool):
            try:  # PyYAML reads "1e-4" (no decimal point) as a string
                out[key] = float(value)
            except ValueError:
                raise ValueError(f"{where}{key} must be a number, got {value!r}") from None
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``key.sub=value`` overrides; returns a new config."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.sub=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise ValueError(f"unknown override key {dotted!r}")
            node = node[k]
        if keys[-1] not in node and keys[0] != "loss":  # loss settings are free-form (see build_loss)
            raise ValueError(f"unknown override key {dotted!r}")
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def load_experiment(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load an experiment YAML on top of ``DEFAULTS``, apply overrides and resolve paths.

    Adds ``experiment`` (the file's stem) and makes ``data.config`` absolute.
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _merge(DEFAULTS, raw, "")
    cfg = apply_overrides(cfg, overrides or [])
    cfg = _merge(DEFAULTS, cfg, "")  # re-validate after overrides
    data_cfg = Path(os.path.expandvars(os.path.expanduser(str(cfg["data"]["config"]))))
    cfg["data"]["config"] = str(data_cfg if data_cfg.is_absolute() else (path.parent / data_cfg).resolve())
    cfg["experiment"] = path.stem
    if not isinstance(cfg["loss"], dict) or "name" not in cfg["loss"]:
        raise ValueError("loss must be a mapping with a 'name'")
    if cfg["train"]["epochs"] < 1 or cfg["data"]["batch_size"] < 1:
        raise ValueError("train.epochs and data.batch_size must be >= 1")
    if cfg["schedule"]["name"] == "cosine" and cfg["schedule"]["min_lr"] > cfg["optim"]["lr"]:
        raise ValueError(f"schedule.min_lr ({cfg['schedule']['min_lr']}) is above optim.lr "
                         f"({cfg['optim']['lr']}): the cosine schedule would raise the learning rate")
    patience = cfg["train"]["early_stopping_patience"]
    if patience is not None and patience < 1:
        raise ValueError("train.early_stopping_patience must be >= 1 or null")
    return cfg


def run_name(cfg: dict) -> str:
    """Default run folder name, e.g. ``baseline_high-mid_seed0``."""
    if cfg["output"]["name"]:
        return str(cfg["output"]["name"])
    return f"{cfg['experiment']}_{cfg['data']['noise']}-{cfg['data']['mask']}_seed{cfg['train']['seed']}"
