"""PyTorch dataset: clean κ from a built dataset → noisy, masked shear on the fly."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ObservationConfig
from .forward_model import SYMMETRIES, apply_symmetry
from .observation import make_observation
from .splits import SPLITS

__all__ = ["ShearMapDataset"]

MODES = ("full", "tiles", "random")


class ShearMapDataset(Dataset):
    """Items are dicts with ``x`` (3, P, P) = [γ₁, γ₂, mask] and ``y`` (1, P, P) = κ,
    both multiplied by ``observation.scale``, plus ``map_index``, ``y0``, ``x0``.

    Each item regenerates the full-map observation and then crops it, so shear is
    never computed on a crop and every view of a map sees the same noise and mask.

    Args:
        root: a folder written by :func:`cosmicweb.data.build.build_dataset`.
        split: ``"train"``, ``"val"`` or ``"test"``.
        observation: noise/mask settings (``load_config(...).observation``).
        noise, mask: level names from ``observation.noise_levels`` / ``mask_levels``.
        mode: ``"full"`` maps, non-overlapping ``"tiles"``, or ``"random"`` crops
            (positions change with :meth:`set_epoch`).
        augment: random flips/transposes with correct shear signs. Training only.
        resample_each_epoch: draw new noise and masks every epoch (training only;
            val/test must keep it False so their inputs never change).

    Call :meth:`set_epoch` at the start of every epoch. With
    ``persistent_workers=True`` the workers keep their own copies and miss it.
    """

    def __init__(self, root: str | Path, split: str, observation: ObservationConfig, noise: str, mask: str, *,
                 mode: str = "full", patch_size: int = 256, crops_per_map: int = 4,
                 augment: bool = False, resample_each_epoch: bool = False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if patch_size <= 0 or crops_per_map <= 0:
            raise ValueError("patch_size and crops_per_map must be positive")
        self.root = Path(root)
        self.split, self.mode = split, mode
        self.patch_size, self.crops_per_map = int(patch_size), int(crops_per_map)
        self.augment, self.resample_each_epoch = bool(augment), bool(resample_each_epoch)
        self.obs = observation
        self.noise, self.mask = noise, mask
        self.epoch = 0

        meta = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        self.map_shape = tuple(meta["map_shape"])
        self.sigma_pix = observation.sigma_pix(noise, meta["pixel_area_arcmin2"])
        self.mask_spec = observation.mask(mask)
        with (self.root / "manifest.csv").open(newline="", encoding="utf-8") as fh:
            rows = sorted((r for r in csv.DictReader(fh) if r["split"] == split), key=lambda r: int(r["row"]))
        self.names = [r["name"] for r in rows]
        self._keys = [int(r["map_key"]) for r in rows]
        self._path = self.root / f"kappa_{split}.npy"
        self._kappa = None
        if self.kappa.shape != (len(rows), *self.map_shape):
            raise ValueError(f"{self._path} has shape {self.kappa.shape}, manifest expects "
                             f"{(len(rows), *self.map_shape)}")
        if mode != "full" and (min(self.map_shape) < patch_size
                               or (mode == "tiles" and any(n % patch_size for n in self.map_shape))):
            raise ValueError(f"patch_size={patch_size} does not fit map shape {self.map_shape} for mode={mode!r}")
        self._tiles_x = self.map_shape[1] // patch_size

    @property
    def kappa(self) -> np.ndarray:
        if self._kappa is None:  # opened lazily so each DataLoader worker maps the file itself
            self._kappa = np.load(self._path, mmap_mode="r")
        return self._kappa

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_kappa"] = None  # never pickle the memory map (it would copy the data)
        return state

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _per_map(self) -> int:
        if self.mode == "full":
            return 1
        if self.mode == "tiles":
            return (self.map_shape[0] // self.patch_size) * self._tiles_x
        return self.crops_per_map

    def __len__(self) -> int:
        return len(self.names) * self._per_map()

    def _rng(self, index: int, stream: int) -> np.random.Generator:
        return np.random.default_rng([self.obs.seed, self.epoch, index, stream])

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        map_index, sub = divmod(index, self._per_map())
        p = self.patch_size
        if self.mode == "full":
            y0 = x0 = 0
        elif self.mode == "tiles":
            ty, tx = divmod(sub, self._tiles_x)
            y0, x0 = ty * p, tx * p
        else:
            rng = self._rng(index, 0)
            y0 = int(rng.integers(0, self.map_shape[0] - p + 1))
            x0 = int(rng.integers(0, self.map_shape[1] - p + 1))

        x, y = make_observation(np.asarray(self.kappa[map_index]), self.sigma_pix, self.mask_spec,
                                self.obs.seed, self._keys[map_index],
                                epoch=self.epoch if self.resample_each_epoch else None,
                                zero_mean_target=self.obs.zero_mean_target)
        if self.mode != "full":
            x, y = x[:, y0:y0 + p, x0:x0 + p], y[y0:y0 + p, x0:x0 + p]
        y = y[None]
        if self.augment:
            rng = self._rng(index, 1)
            for op in SYMMETRIES:
                if rng.random() < 0.5:
                    x, y = apply_symmetry(x, y, op)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        x[:2] *= self.obs.scale
        y *= self.obs.scale
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y), "map_index": map_index, "y0": y0, "x0": x0}
