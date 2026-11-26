"""Flat-sky Kaiser–Squires forward model: convergence κ → shear (γ₁, γ₂).

In Fourier space, with ℓ² = ℓx² + ℓy²,

    γ̃₁ = (ℓx² − ℓy²)/ℓ² · κ̃ ,    γ̃₂ = 2 ℓx ℓy/ℓ² · κ̃ .

Array convention: axis 0 is y (rows), axis 1 is x (columns). Frequencies are in
cycles/pixel; the pixel scale cancels in the kernel ratios.

* The DC mode is zeroed: a constant κ (mass sheet) produces no shear.
* On even-sized grids the Nyquist row/column is zeroed: the γ₂ kernel is odd in
  frequency, so those modes cannot keep a real field real. They carry a negligible
  fraction of the power.
* For every retained mode d₁² + d₂² = 1, so the inverse is κ̃ = d₁γ̃₁ + d₂γ̃₂.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import fft as sfft

__all__ = ["kappa_to_shear", "shear_to_kappa", "retained_modes", "apply_symmetry", "SYMMETRIES"]


@lru_cache(maxsize=8)
def retained_modes(shape: tuple[int, int]) -> np.ndarray:
    """rfft2-grid weights: 1 for modes the forward model keeps, 0 for DC and Nyquist."""
    ny, nx = shape
    keep = np.ones((ny, nx // 2 + 1))
    keep[0, 0] = 0.0
    if ny % 2 == 0:
        keep[ny // 2, :] = 0.0
    if nx % 2 == 0:
        keep[:, -1] = 0.0
    keep.setflags(write=False)
    return keep


@lru_cache(maxsize=8)
def _ks_kernels(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    ly = sfft.fftfreq(ny)[:, None]
    lx = sfft.rfftfreq(nx)[None, :]
    l2 = lx**2 + ly**2
    l2[0, 0] = 1.0  # DC is zeroed by retained_modes; avoid 0/0
    keep = retained_modes(shape)
    d1 = (lx**2 - ly**2) / l2 * keep
    d2 = 2.0 * lx * ly / l2 * keep
    d1.setflags(write=False)
    d2.setflags(write=False)
    return d1, d2


def _as_map(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    return a


def kappa_to_shear(kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 (γ₁, γ₂) for a periodic 2-D convergence map.

    Apply this to the full simulated map, never to a crop: the FFT treats its
    input as periodic, which is only (approximately) true for the full map.
    """
    k = _as_map(kappa, "kappa")
    d1, d2 = _ks_kernels(k.shape)
    k_ft = sfft.rfft2(k)
    return sfft.irfft2(d1 * k_ft, s=k.shape), sfft.irfft2(d2 * k_ft, s=k.shape)


def shear_to_kappa(gamma1: np.ndarray, gamma2: np.ndarray) -> np.ndarray:
    """Kaiser–Squires inverse. Exact inverse of :func:`kappa_to_shear` on retained modes."""
    g1, g2 = _as_map(gamma1, "gamma1"), _as_map(gamma2, "gamma2")
    if g1.shape != g2.shape:
        raise ValueError(f"shape mismatch: {g1.shape} vs {g2.shape}")
    d1, d2 = _ks_kernels(g1.shape)
    return sfft.irfft2(d1 * sfft.rfft2(g1) + d2 * sfft.rfft2(g2), s=g1.shape)


SYMMETRIES = ("flip_x", "flip_y", "transpose")


def apply_symmetry(x: np.ndarray, y: np.ndarray, op: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply one image symmetry to an input/target pair, keeping the shear physical.

    ``x`` is (C, H, W) with channels [γ₁, γ₂, ...]; ``y`` is (..., H, W). Shear is
    spin-2, so reflections flip the sign of γ₂ and a transpose (x ↔ y) flips γ₁.
    Compositions of these three generate all 8 rotations/reflections of a square.
    """
    if op == "flip_x":
        x, y = x[..., ::-1], y[..., ::-1]
        sign = (1.0, -1.0)
    elif op == "flip_y":
        x, y = x[..., ::-1, :], y[..., ::-1, :]
        sign = (1.0, -1.0)
    elif op == "transpose":
        x, y = np.swapaxes(x, -2, -1), np.swapaxes(y, -2, -1)
        sign = (-1.0, 1.0)
    else:
        raise ValueError(f"unknown symmetry {op!r}; expected one of {SYMMETRIES}")
    x = np.array(x)  # contiguous copy, never a view into the caller's data
    y = np.array(y)
    x[0] *= sign[0]
    x[1] *= sign[1]
    return x, y
