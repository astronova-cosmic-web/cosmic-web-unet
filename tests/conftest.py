from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import gaussian_filter


def random_kappa(seed: int, n: int = 64) -> np.ndarray:
    """Smooth Gaussian random field with κ-like amplitude (std ≈ 0.02)."""
    field = gaussian_filter(np.random.default_rng(seed).normal(size=(n, n)), 2.0, mode="wrap")
    return (0.02 * field / field.std() + 0.01).astype(np.float32)


def write_fits(path: Path, data: np.ndarray, angle: float | None = 3.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data)
    if angle is not None:
        hdu.header["ANGLE"] = angle
    hdu.writeto(path, overwrite=True)
    return path


def make_map_folder(folder: Path, n_real: int = 10, n: int = 64, redshifts=("0.50",)) -> list[Path]:
    """MassiveNuS-style tree: Maps<z>/WLconv_z<z>_<NNNN>r.fits."""
    paths = []
    for z in redshifts:
        for r in range(1, n_real + 1):
            name = f"WLconv_z{z}_{r:04d}r.fits"
            paths.append(write_fits(folder / f"Maps{z.replace('.', '')}" / name,
                                    random_kappa(1000 * r + int(float(z) * 100), n)))
    return paths


def tar_folder(folder: Path, tar_path: Path) -> Path:
    with tarfile.open(tar_path, "w") as tar:
        tar.add(folder, arcname="convergence_gal_test")
    return tar_path


def write_config(path: Path, source: str, output: str = "out", extra_build: str = "", root: str = ".") -> Path:
    path.write_text(f"""
dataset_root: {root}
source: {source}
output_dir: {output}
build:
  map_size_deg: 3.5
  split_fractions: {{train: 0.6, val: 0.2, test: 0.2}}
  seed: 7
{extra_build}
observation:
  seed: 99
  sigma_e: 0.28
  scale: 10.0
  noise_levels:
    none: {{sigma_pix: 0.0}}
    low: {{n_gal: 30}}
    high: {{n_gal: 6}}
  mask_levels:
    none: {{fraction: 0.0}}
    mid: {{fraction: 0.15, edge_px: 2, gap_height_px: [1, 3], star_radius_px: [1, 3]}}
""", encoding="utf-8")
    return path


@pytest.fixture
def map_folder(tmp_path) -> Path:
    folder = tmp_path / "raw"
    make_map_folder(folder)
    return folder


@pytest.fixture
def built(tmp_path, map_folder):
    """A built dataset: (config, dataset.json contents)."""
    from cosmicweb.data import build_dataset, load_config

    cfg = load_config(write_config(tmp_path / "cfg.yaml", map_folder.as_posix()))
    return cfg, build_dataset(cfg, progress=False)
