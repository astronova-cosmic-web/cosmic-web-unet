import numpy as np
from scipy.ndimage import gaussian_filter


def random_kappa(seed: int, n: int = 64) -> np.ndarray:
    """Smooth Gaussian random field with κ-like amplitude (std ≈ 0.02)."""
    field = gaussian_filter(np.random.default_rng(seed).normal(size=(n, n)), 2.0, mode="wrap")
    return (0.02 * field / field.std() + 0.01).astype(np.float32)
