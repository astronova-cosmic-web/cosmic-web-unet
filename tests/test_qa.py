import pytest

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from cosmicweb.data import ShearMapDataset  # noqa: E402
from cosmicweb.data.qa import plot_condition_grid, plot_examples  # noqa: E402


def test_plot_examples_full_and_clean_shear(built, tmp_path):
    cfg, _ = built
    ds = ShearMapDataset(cfg.build.output_dir, "val", cfg.observation, "mid", "mid", mode="full")

    out_clean = tmp_path / "qa_clean.png"
    fig = plot_examples(ds, indices=(0,), path=out_clean, show_clean_shear=True)
    assert out_clean.exists() and out_clean.stat().st_size > 0
    assert fig.axes[0].get_figure() is fig
    assert len([ax for ax in fig.axes if ax.get_title()]) == 5

    out_plain = tmp_path / "qa_plain.png"
    fig_plain = plot_examples(ds, indices=(0,), path=out_plain, show_clean_shear=False)
    assert out_plain.exists() and out_plain.stat().st_size > 0
    assert len([ax for ax in fig_plain.axes if ax.get_title()]) == 4


def test_plot_examples_random_crop(built, tmp_path):
    cfg, _ = built
    ds = ShearMapDataset(
        cfg.build.output_dir, "train", cfg.observation, "low", "low", mode="random", patch_size=32
    )
    out = tmp_path / "qa_crop.png"
    fig = plot_examples(ds, indices=(0,), path=out, show_clean_shear=True)
    assert out.exists() and out.stat().st_size > 0
    # Clean shear is omitted for cropped items because full-map FFT is required.
    assert len([ax for ax in fig.axes if ax.get_title()]) == 4


def test_plot_condition_grid(built, tmp_path):
    cfg, _ = built
    out = tmp_path / "qa_grid.png"
    fig = plot_condition_grid(
        cfg.build.output_dir,
        cfg.observation,
        split="val",
        index=0,
        noise_levels=("low", "high"),
        mask_levels=("low", "high"),
        path=out,
    )
    assert out.exists() and out.stat().st_size > 0
    assert len([ax for ax in fig.axes if ax.get_title()]) == 4

