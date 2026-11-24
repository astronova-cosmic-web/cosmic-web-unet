from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
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
