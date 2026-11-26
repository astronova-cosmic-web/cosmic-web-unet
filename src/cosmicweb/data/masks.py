"""Survey masks: map edges, a chip gap and bright-star holes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["MaskSpec", "generate_mask"]


@dataclass(frozen=True)
class MaskSpec:
    """Mask settings. Sizes are in pixels and do not scale with the map size.

    ``fraction`` is the target masked fraction of the whole map, including the
    edge border and chip gap; star holes are added until it is reached.
    """

    fraction: float = 0.0
    edge_px: int = 0
    gap_height_px: tuple[int, int] = (6, 12)
    star_radius_px: tuple[int, int] = (3, 13)

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction < 1.0:
            raise ValueError("mask fraction must be in [0, 1)")
        if self.edge_px < 0:
            raise ValueError("edge_px must be >= 0")
        for name in ("gap_height_px", "star_radius_px"):
            lo, hi = getattr(self, name)
            if not 1 <= lo <= hi:
                raise ValueError(f"{name} must be [lo, hi] with 1 <= lo <= hi")
            object.__setattr__(self, name, (int(lo), int(hi)))


def generate_mask(shape: tuple[int, int], spec: MaskSpec, rng: np.random.Generator,
                  max_stars: int = 20000) -> np.ndarray:
    """Return a float32 mask (1 = observed, 0 = missing).

    Order: edge border, then one full-width horizontal chip gap (capped at half of
    the target area), then circular star holes until ``spec.fraction`` is reached.
    Overlapping holes are not double-counted.
    """
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("mask shape must be positive")
    mask = np.ones((h, w), dtype=np.float32)
    if spec.edge_px:
        if 2 * spec.edge_px >= min(h, w):
            raise ValueError(f"edge_px={spec.edge_px} masks the whole {h}x{w} map")
        e = spec.edge_px
        mask[:e, :] = mask[-e:, :] = mask[:, :e] = mask[:, -e:] = 0.0
    if spec.fraction == 0.0:
        return mask

    target = spec.fraction * h * w
    n_masked = int(np.count_nonzero(mask == 0))
    gap_cap = int(max(0.0, 0.5 * target - n_masked) // w)
    if gap_cap > 0:
        gap_h = min(int(rng.integers(spec.gap_height_px[0], spec.gap_height_px[1] + 1)), gap_cap, h)
        y0 = int(rng.integers(0, h - gap_h + 1))
        n_masked += int(np.count_nonzero(mask[y0:y0 + gap_h]))
        mask[y0:y0 + gap_h, :] = 0.0

    r_lo, r_hi = spec.star_radius_px
    n_stars = 0
    while n_masked < target and n_stars < max_stars:
        r = float(rng.uniform(r_lo, np.nextafter(r_hi, np.inf)))
        cy, cx = float(rng.uniform(0, h)), float(rng.uniform(0, w))
        ys = slice(max(int(cy - r), 0), min(int(cy + r) + 2, h))
        xs = slice(max(int(cx - r), 0), min(int(cx + r) + 2, w))
        yy, xx = np.ogrid[ys, xs]
        inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        region = mask[ys, xs]
        n_masked += int(np.count_nonzero(region[inside]))
        region[inside] = 0.0
        n_stars += 1
    if n_masked < target:
        warnings.warn(f"mask reached {n_masked / (h * w):.2%} < requested {spec.fraction:.2%} "
                      f"after {max_stars} stars", RuntimeWarning, stacklevel=2)
    return mask
