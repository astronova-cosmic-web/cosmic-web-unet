"""Data pipeline: forward model, shape noise, masks, line-of-sight splits, dataset.

Typical use::

    from cosmicweb.data import load_config, build_dataset, ShearMapDataset

    cfg = load_config("configs/data.yaml")
    build_dataset(cfg)                                   # once: writes clean κ + manifests
    train = ShearMapDataset(cfg.build.output_dir, "train", cfg.observation, "high", "mid",
                            mode="random", augment=True, resample_each_epoch=True)

``ShearMapDataset`` needs PyTorch and is imported lazily, so the NumPy-only parts
work without it.
"""

from .build import build_dataset
from .config import BuildConfig, DataConfig, ObservationConfig, load_config
from .forward_model import SYMMETRIES, apply_symmetry, kappa_to_shear, shear_to_kappa
from .io import MapInfo, MapSource, load_kappa, parse_map_name
from .masks import MaskSpec, generate_mask
from .noise import add_shape_noise, pixel_area_arcmin2, pixel_noise_sigma
from .observation import make_observation, map_key
from .splits import SPLITS, LeakageError, assign_splits, check_leakage

__all__ = [
    "BuildConfig", "DataConfig", "LeakageError", "MapInfo", "MapSource", "MaskSpec", "ObservationConfig",
    "SPLITS", "SYMMETRIES", "ShearMapDataset", "add_shape_noise", "apply_symmetry", "assign_splits",
    "build_dataset", "check_leakage", "generate_mask", "kappa_to_shear", "load_config", "load_kappa",
    "make_observation", "map_key", "parse_map_name", "pixel_area_arcmin2", "pixel_noise_sigma",
    "shear_to_kappa",
]


def __getattr__(name):
    if name == "ShearMapDataset":
        from .dataset import ShearMapDataset

        return ShearMapDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
