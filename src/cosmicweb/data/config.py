"""YAML configuration for building the dataset and generating observations.

See ``configs/data.yaml`` for an annotated example. Relative
``dataset_root`` is resolved from the config file's folder; relative ``source``
and ``output_dir`` from ``dataset_root``. ``~`` and ``$VARS`` are expanded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .masks import MaskSpec
from .noise import pixel_noise_sigma
from .splits import SPLITS

__all__ = ["BuildConfig", "ObservationConfig", "DataConfig", "load_config"]


def _take(d: dict, keys: set[str], where: str) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(d) - keys
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}; allowed: {sorted(keys)}")
    return d


def _path(value: str | Path, base: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (p if p.is_absolute() else base / p).resolve()


@dataclass(frozen=True)
class BuildConfig:
    """How raw maps become the frozen κ dataset."""

    source: Path  # folder of FITS maps, or a .tar archive of them
    output_dir: Path
    redshifts: tuple[float, ...] | None = (0.5,)  # None keeps every redshift
    max_maps: int | None = None  # first N maps by name, for quick test builds
    map_size_deg: float = 3.5  # side of each square map
    split_fractions: dict[str, float] = field(default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15})
    seed: int = 42  # split shuffle

    def __post_init__(self) -> None:
        if self.max_maps is not None and self.max_maps <= 0:
            raise ValueError("max_maps must be positive or null")
        if self.map_size_deg <= 0:
            raise ValueError("map_size_deg must be positive")
        if set(self.split_fractions) != set(SPLITS):
            raise ValueError(f"split_fractions needs exactly {SPLITS}")


@dataclass(frozen=True)
class ObservationConfig:
    """How a clean κ map becomes a noisy, masked shear observation."""

    seed: int = 1234  # noise/mask seed; keep fixed so val/test inputs never change
    sigma_e: float = 0.28  # total intrinsic ellipticity dispersion (per component: /√2)
    scale: float = 10.0  # multiplies shear and κ so values are O(1) for the network
    zero_mean_target: bool = True  # subtract the full-map κ mean (shear cannot measure it)
    noise_levels: dict[str, dict[str, float]] = field(default_factory=lambda: {"none": {"sigma_pix": 0.0}})
    mask_levels: dict[str, MaskSpec] = field(default_factory=lambda: {"none": MaskSpec()})

    def __post_init__(self) -> None:
        for name, level in self.noise_levels.items():
            if len(level) != 1 or not set(level) <= {"n_gal", "sigma_pix"}:
                raise ValueError(f"noise level {name!r} must set exactly one of n_gal or sigma_pix")
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def sigma_pix(self, noise: str, pixel_area: float) -> float:
        """Per-component pixel noise σ for a named noise level."""
        try:
            level = self.noise_levels[noise]
        except KeyError:
            raise KeyError(f"unknown noise level {noise!r}; have {sorted(self.noise_levels)}") from None
        if "sigma_pix" in level:
            return float(level["sigma_pix"])
        return pixel_noise_sigma(self.sigma_e, float(level["n_gal"]), pixel_area)

    def mask(self, mask: str) -> MaskSpec:
        try:
            return self.mask_levels[mask]
        except KeyError:
            raise KeyError(f"unknown mask level {mask!r}; have {sorted(self.mask_levels)}") from None


@dataclass(frozen=True)
class DataConfig:
    dataset_root: Path
    build: BuildConfig
    observation: ObservationConfig
    path: Path | None = None


def load_config(path: str | Path, dataset_root: str | Path | None = None) -> DataConfig:
    """Load a data config. ``dataset_root`` overrides the file's value."""
    path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _take(raw, {"dataset_root", "source", "output_dir", "build", "observation"}, str(path))
    root = _path(dataset_root if dataset_root is not None else raw.get("dataset_root", "."), path.parent)
    for key in ("source", "output_dir"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key {key!r}")

    b = dict(_take(raw.get("build") or {}, {"redshifts", "max_maps", "map_size_deg", "split_fractions", "seed"},
                   "build"))
    if b.get("redshifts") is not None:
        b["redshifts"] = tuple(float(z) for z in b["redshifts"])
    if "split_fractions" in b:
        b["split_fractions"] = {k: float(v) for k, v in b["split_fractions"].items()}
    build = BuildConfig(source=_path(raw["source"], root), output_dir=_path(raw["output_dir"], root), **b)

    o = dict(_take(raw.get("observation") or {},
                   {"seed", "sigma_e", "scale", "zero_mean_target", "noise_levels", "mask_levels"}, "observation"))
    if "noise_levels" in o:
        o["noise_levels"] = {str(k): {kk: float(vv) for kk, vv in _take(v, {"n_gal", "sigma_pix"},
                                                                         f"noise level {k}").items()}
                             for k, v in o["noise_levels"].items()}
    if "mask_levels" in o:
        fields = {"fraction", "edge_px", "gap_height_px", "star_radius_px"}
        o["mask_levels"] = {str(k): MaskSpec(**{kk: tuple(vv) if isinstance(vv, list) else vv
                                                for kk, vv in _take(v, fields, f"mask level {k}").items()})
                            for k, v in o["mask_levels"].items()}
    return DataConfig(dataset_root=root, build=build, observation=ObservationConfig(**o), path=path)
