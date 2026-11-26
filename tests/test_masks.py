import numpy as np
import pytest

from cosmicweb.data import MaskSpec, generate_mask


@pytest.mark.parametrize("spec", [
    MaskSpec(0.05, edge_px=4, gap_height_px=(4, 8), star_radius_px=(2, 8)),
    MaskSpec(0.15, edge_px=8),
    MaskSpec(0.30, edge_px=8, star_radius_px=(3, 20)),
])
def test_mask_fraction_is_reached(spec):
    rng = np.random.default_rng(0)
    achieved = np.array([1 - generate_mask((512, 512), spec, rng).mean() for _ in range(40)])
    assert abs(achieved.mean() - spec.fraction) < 0.01
    assert achieved.min() >= spec.fraction  # stars are added until the target is met


def test_mask_edges_gap_and_dtype():
    m = generate_mask((128, 128), MaskSpec(0.2, edge_px=3), np.random.default_rng(1))
    assert m.dtype == np.float32 and set(np.unique(m)) <= {0.0, 1.0}
    assert (m[:3] == 0).all() and (m[-3:] == 0).all() and (m[:, :3] == 0).all() and (m[:, -3:] == 0).all()
    assert (m.sum(axis=1) == 0).any()  # a full-width chip gap exists


def test_mask_is_deterministic_and_zero_fraction_is_empty():
    spec = MaskSpec(0.15)
    a = generate_mask((64, 64), spec, np.random.default_rng(5))
    b = generate_mask((64, 64), spec, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    assert generate_mask((64, 64), MaskSpec(0.0), np.random.default_rng(0)).min() == 1.0


@pytest.mark.parametrize("kwargs", [dict(fraction=1.0), dict(fraction=-0.1), dict(edge_px=-1),
                                    dict(star_radius_px=(5, 2)), dict(gap_height_px=(0, 3))])
def test_invalid_mask_specs(kwargs):
    with pytest.raises(ValueError):
        MaskSpec(**kwargs)


def test_edge_cannot_cover_map():
    with pytest.raises(ValueError):
        generate_mask((10, 10), MaskSpec(0.0, edge_px=5), np.random.default_rng(0))
