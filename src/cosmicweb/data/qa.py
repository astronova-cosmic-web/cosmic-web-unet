"""Visual QA for generated observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["plot_examples"]


def plot_examples(dataset, indices=(0,), path: str | Path | None = None):
    """Plot γ₁, γ₂, mask and target κ for the given dataset items; optionally save.

    Values are shown divided by ``dataset.obs.scale`` (physical units).
    """
    import matplotlib.pyplot as plt

    indices = list(indices)
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)), squeeze=False)
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
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{dataset.split} · noise={dataset.noise} (σ_pix={dataset.sigma_pix:.3f}) · "
                 f"mask={dataset.mask} · mode={dataset.mode}")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
