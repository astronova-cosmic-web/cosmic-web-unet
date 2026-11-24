import math

import numpy as np
import pytest

from cosmicweb.data import add_shape_noise, pixel_area_arcmin2, pixel_noise_sigma


def test_massivenus_pixel_area_and_noise():
    area = pixel_area_arcmin2(3.5, 512)
    assert area == pytest.approx((210 / 512) ** 2)
    assert area == pytest.approx(0.1682, abs=1e-4)
    assert pixel_noise_sigma(0.28, 6.0, area) == pytest.approx(0.28 / math.sqrt(12 * area))
    assert pixel_noise_sigma(0.28, 6.0, area) == pytest.approx(0.197, abs=1e-3)


def test_noise_has_requested_std_and_scales_with_n_gal():
    g = add_shape_noise(np.zeros((2, 256, 256)), 0.2, np.random.default_rng(0))
    assert g.std() == pytest.approx(0.2, rel=0.01)
    assert pixel_noise_sigma(0.28, 24, 1.0) == pytest.approx(pixel_noise_sigma(0.28, 6, 1.0) / 2)
    with pytest.raises(ValueError):
        pixel_noise_sigma(0.28, 0, 1.0)
