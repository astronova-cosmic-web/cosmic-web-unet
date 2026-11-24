import itertools

import numpy as np
import pytest
from scipy import fft as sfft

from cosmicweb.data import SYMMETRIES, apply_symmetry, kappa_to_shear, shear_to_kappa
from cosmicweb.data.forward_model import retained_modes

from conftest import random_kappa


def project(kappa):
    """κ restricted to the modes the forward model keeps (no DC, no Nyquist)."""
    return sfft.irfft2(sfft.rfft2(kappa) * retained_modes(kappa.shape), s=kappa.shape)


@pytest.mark.parametrize("shape", [(64, 64), (63, 65), (32, 48)])
def test_shear_is_finite_and_zero_mean(shape):
    g1, g2 = kappa_to_shear(np.random.default_rng(0).normal(size=shape))
    assert np.isfinite(g1).all() and np.isfinite(g2).all()
    assert abs(g1.mean()) < 1e-12 and abs(g2.mean()) < 1e-12


@pytest.mark.parametrize("shape", [(64, 64), (63, 65)])
def test_round_trip_recovers_kappa_up_to_dropped_modes(shape):
    kappa = np.random.default_rng(1).normal(size=shape)
    np.testing.assert_allclose(shear_to_kappa(*kappa_to_shear(kappa)), project(kappa), atol=1e-10)


def test_power_is_conserved():
    kappa = random_kappa(2).astype(np.float64)
    g1, g2 = kappa_to_shear(kappa)
    assert np.sum(g1**2 + g2**2) / np.sum(project(kappa) ** 2) == pytest.approx(1.0, abs=1e-10)


def test_sign_convention_on_plane_waves():
    n = 64
    yy, xx = np.mgrid[:n, :n]
    wave_x = np.cos(2 * np.pi * 3 * xx / n)  # varies along x only: γ₁ = κ, γ₂ = 0
    g1, g2 = kappa_to_shear(wave_x)
    np.testing.assert_allclose(g1, wave_x, atol=1e-12)
    np.testing.assert_allclose(g2, 0, atol=1e-12)
    wave_y = np.cos(2 * np.pi * 3 * yy / n)  # varies along y only: γ₁ = −κ
    np.testing.assert_allclose(kappa_to_shear(wave_y)[0], -wave_y, atol=1e-12)
    wave_d = np.cos(2 * np.pi * 3 * (xx + yy) / n)  # along the diagonal: γ₁ = 0, γ₂ = κ
    g1, g2 = kappa_to_shear(wave_d)
    np.testing.assert_allclose(g1, 0, atol=1e-12)
    np.testing.assert_allclose(g2, wave_d, atol=1e-12)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        kappa_to_shear(np.zeros((4, 4, 4)))
    bad = np.zeros((8, 8))
    bad[1, 1] = np.nan
    with pytest.raises(ValueError):
        kappa_to_shear(bad)


def _spatial(op, a):
    return {"flip_x": lambda a: a[..., ::-1], "flip_y": lambda a: a[..., ::-1, :],
            "transpose": lambda a: np.swapaxes(a, -2, -1)}[op](a)


@pytest.mark.parametrize("ops", [c for r in range(1, 4) for c in itertools.permutations(SYMMETRIES, r)])
def test_symmetries_keep_shear_consistent_with_kappa(ops):
    """Transforming (shear, κ) must equal recomputing shear from the transformed κ."""
    kappa = random_kappa(3, n=32).astype(np.float64)
    kappa[:, :5] += 0.05  # break the field's own symmetries
    mask = np.ones_like(kappa)
    mask[3:7, 10:20] = 0
    x = np.concatenate([np.stack(kappa_to_shear(kappa)), mask[None]])
    y = kappa[None]
    expected_k, expected_m = kappa, mask
    for op in ops:
        x, y = apply_symmetry(x, y, op)
        expected_k, expected_m = _spatial(op, expected_k), _spatial(op, expected_m)
    np.testing.assert_allclose(x[:2], np.stack(kappa_to_shear(expected_k)), atol=1e-12)
    np.testing.assert_array_equal(x[2], expected_m)
    np.testing.assert_array_equal(y[0], expected_k)


def test_apply_symmetry_does_not_modify_inputs():
    x, y = np.ones((3, 4, 4)), np.ones((1, 4, 4))
    apply_symmetry(x, y, "transpose")
    assert (x == 1).all()
    with pytest.raises(ValueError):
        apply_symmetry(x, y, "rotate")
