"""Turn one clean κ map into a noisy, masked shear observation, reproducibly."""

from __future__ import annotations

import zlib

import numpy as np

from .forward_model import kappa_to_shear
from .masks import MaskSpec, generate_mask
from .noise import add_shape_noise

__all__ = ["map_key", "make_observation"]

_NOISE_STREAM, _MASK_STREAM = 0, 1


def map_key(name: str) -> int:
    """Stable integer ID from a map's file name (independent of file order or location)."""
    return zlib.crc32(name.encode("utf-8"))


def make_observation(kappa: np.ndarray, sigma_pix: float, mask: MaskSpec, seed: int, key: int,
                     epoch: int | None = None, zero_mean_target: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return ``x`` float32 (3, H, W) = [γ₁, γ₂, mask] and target ``y`` float32 (H, W).

    Steps, always on the full map: κ → shear → + noise → × mask. Masked shear is
    exactly 0. Values are in physical units (no scaling).

    Randomness is keyed by ``(seed, key)``, plus ``epoch`` when given, so the same
    map always gets the same noise and mask. Noise and mask use separate streams:
    changing the mask level leaves the noise unchanged, and every noise level uses
    the same unit-normal draws scaled by ``sigma_pix`` (common random numbers),
    which keeps comparisons across noise levels clean.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    extra = [] if epoch is None else [int(epoch)]
    noise_rng = np.random.default_rng([seed, key, _NOISE_STREAM, *extra])
    mask_rng = np.random.default_rng([seed, key, _MASK_STREAM, *extra])

    gamma = add_shape_noise(np.stack(kappa_to_shear(kappa)), sigma_pix, noise_rng)
    m = generate_mask(kappa.shape, mask, mask_rng)
    x = np.empty((3, *kappa.shape), dtype=np.float32)
    x[:2] = gamma * m
    x[2] = m
    y = kappa - kappa.mean() if zero_mean_target else kappa
    return x, y.astype(np.float32)
