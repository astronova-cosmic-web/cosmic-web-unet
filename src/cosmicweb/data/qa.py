"""Visual QA for generated observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .forward_model import kappa_to_shear

__all__ = ["plot_examples", "plot_condition_grid"]


def plot_examples(dataset, indices=(0,), path: str | Path | None = None, *, show_clean_shear: bool = True):
    """Plot γ₁, γ₂, mask, target κ, and optionally clean γ₁ for the given dataset items.

    Values are shown divided by ``dataset.obs.scale`` (physical units).
    Clean shear is computed on ``mode="full"`` items so forward-model halos can be
    verified even when shape noise dominates the noisy shear panels.
    """
    import matplotlib.pyplot as plt

    indices = list(indices)
    include_clean = show_clean_shear and dataset.mode == "full"
    ncols = 5 if include_clean else 4
    fig, axes = plt.subplots(len(indices), ncols, figsize=(4 * ncols, 4 * len(indices)), squeeze=False)
    scale = dataset.obs.scale
    for row, i in zip(axes, indices):
        item = dataset[i]
        x, y = item["x"].numpy(), item["y"].numpy()[0] / scale
        valid = x[2] > 0
        lim = float(np.percentile(np.abs(x[:2, valid]) / scale, 99)) if valid.any() else 1.0
        name = dataset.names[item["map_index"]]
        for ax, c, label in ((row[0], 0, "γ₁"), (row[1], 1, "γ₂")):
            im = ax.imshow(x[c] / scale, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim)
            ax.set_title(f"{label} (noisy, masked)\n{name}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        im = row[2].imshow(x[2], cmap="gray", origin="lower", vmin=0, vmax=1)
        row[2].set_title(f"mask: {1 - valid.mean():.1%} masked", fontsize=9)
        fig.colorbar(im, ax=row[2], fraction=0.046, ticks=[0, 1])
        lo, hi = np.percentile(y, [1, 99])
        im = row[3].imshow(y, cmap="viridis", origin="lower", vmin=lo, vmax=hi)
        row[3].set_title("target κ", fontsize=9)
        fig.colorbar(im, ax=row[3], fraction=0.046)
        if include_clean:
            g1_clean, _ = kappa_to_shear(y)
            clean_lim = float(np.percentile(np.abs(g1_clean), 99)) or 1.0
            im = row[4].imshow(g1_clean, cmap="RdBu_r", origin="lower", vmin=-clean_lim, vmax=clean_lim)
            row[4].set_title("γ₁ (clean forward model)", fontsize=9)
            fig.colorbar(im, ax=row[4], fraction=0.046)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{dataset.split} · noise={dataset.noise} (σ_pix={dataset.sigma_pix:.3f}) · "
                 f"mask={dataset.mask} · mode={dataset.mode}")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_condition_grid(
    root: str | Path,
    observation,
    *,
    split: str = "val",
    index: int = 0,
    noise_levels: tuple[str, ...] = ("low", "mid", "high"),
    mask_levels: tuple[str, ...] = ("low", "mid", "high"),
    mode: str = "full",
    path: str | Path | None = None,
):
    """Plot a noise × mask grid of γ₁ observations for one map."""
    import matplotlib.pyplot as plt

    from .dataset import ShearMapDataset

    nrows, ncols = len(noise_levels), len(mask_levels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    scale = observation.scale
    map_name = None
    for r, noise in enumerate(noise_levels):
        for c, mask in enumerate(mask_levels):
            ds = ShearMapDataset(root, split, observation, noise, mask, mode=mode)
            item = ds[index]
            if map_name is None:
                map_name = ds.names[item["map_index"]]
            x = item["x"].numpy()
            valid = x[2] > 0
            lim = float(np.percentile(np.abs(x[0, valid]) / scale, 99)) if valid.any() else 1.0
            ax = axes[r, c]
            im = ax.imshow(x[0] / scale, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim)
            ax.set_title(f"noise={noise} (σ={ds.sigma_pix:.3f})\nmask={mask} ({1 - valid.mean():.1%})", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{split} · {map_name} · mode={mode}")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
