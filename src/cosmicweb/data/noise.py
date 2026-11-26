"""Galaxy shape noise for pixelised shear maps."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["pixel_area_arcmin2", "pixel_noise_sigma", "add_shape_noise"]


def pixel_area_arcmin2(map_size_deg: float, n_pixels: int) -> float:
    """Area of one pixel of a square map ``map_size_deg`` on a side, in arcmin²."""
    if map_size_deg <= 0 or n_pixels <= 0:
        raise ValueError("map_size_deg and n_pixels must be positive")
    return (map_size_deg * 60.0 / n_pixels) ** 2


def pixel_noise_sigma(sigma_e: float, n_gal: float, pixel_area: float) -> float:
    """Per-component shear noise standard deviation in one pixel.

    σ_pix = σ_e / sqrt(2 · n_gal · A_pix)

    ``sigma_e`` is the *total* intrinsic ellipticity dispersion, so each component
    has σ_e/√2. ``n_gal`` is in galaxies/arcmin² and ``pixel_area`` in arcmin².
    """
    if sigma_e < 0 or n_gal <= 0 or pixel_area <= 0:
        raise ValueError("need sigma_e >= 0, n_gal > 0 and pixel_area > 0")
    return sigma_e / math.sqrt(2.0 * n_gal * pixel_area)


def add_shape_noise(gamma: np.ndarray, sigma_pix: float, rng: np.random.Generator) -> np.ndarray:
    """Add independent Gaussian noise of std ``sigma_pix`` to every component and pixel."""
    if sigma_pix < 0:
        raise ValueError("sigma_pix must be >= 0")
    gamma = np.asarray(gamma, dtype=np.float64)
    # float32 draws are ~2x faster; precision is ample since inputs are stored as float32
    return gamma + sigma_pix * rng.standard_normal(gamma.shape, dtype=np.float32)
