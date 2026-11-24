import csv
import json

import numpy as np
import pytest

from cosmicweb.data import build_dataset, load_config
from cosmicweb.data.build import GeometryError

from conftest import make_map_folder, random_kappa, tar_folder, write_config, write_fits


def _read_manifest(out):
    with (out / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_build_outputs(built):
    cfg, meta = built
    out = cfg.build.output_dir
    assert meta["counts"] == {"train": 6, "val": 2, "test": 2}
    assert meta["leakage_check"] == "passed" and meta["failed_maps"] == {}
    assert meta["map_shape"] == [64, 64]
    assert meta["pixel_area_arcmin2"] == pytest.approx((3.5 * 60 / 64) ** 2)
    assert json.loads((out / "dataset.json").read_text()) == meta
    assert not (out / ".partial").exists()
    rows = _read_manifest(out)
    for split, n in meta["counts"].items():
        arr = np.load(out / f"kappa_{split}.npy")
        assert arr.shape == (n, 64, 64) and arr.dtype == np.float32
        split_rows = sorted((r for r in rows if r["split"] == split), key=lambda r: int(r["row"]))
        assert [r["name"] for r in split_rows] == sorted(r["name"] for r in split_rows)
        for r in split_rows:  # each stored map is exactly its source file
            realization = int(r["realization"])
            np.testing.assert_array_equal(arr[int(r["row"])], random_kappa(1000 * realization + 50))
        with (out / "splits" / f"{split}.csv").open() as fh:
            assert len(fh.readlines()) == n + 1


def test_tar_and_folder_builds_are_identical(tmp_path, map_folder):
    tar = tar_folder(map_folder, tmp_path / "maps.tar")
    a = build_dataset(load_config(write_config(tmp_path / "a.yaml", map_folder.as_posix(), "out_a")), progress=False)
    b = build_dataset(load_config(write_config(tmp_path / "b.yaml", tar.as_posix(), "out_b")), progress=False)
    assert a["fingerprint"] == b["fingerprint"]
    for f in ("kappa_train.npy", "kappa_val.npy", "kappa_test.npy", "manifest.csv"):
        assert (tmp_path / "out_a" / f).read_bytes() == (tmp_path / "out_b" / f).read_bytes()


def test_refuses_to_overwrite_unless_asked(built):
    cfg, meta = built
    with pytest.raises(FileExistsError):
        build_dataset(cfg, progress=False)
    assert build_dataset(cfg, overwrite=True, progress=False)["fingerprint"] == meta["fingerprint"]


def test_corrupt_and_mismatched_maps_are_skipped_and_recorded(tmp_path, map_folder):
    bad = map_folder / "Maps050" / "WLconv_z0.50_0003r.fits"
    bad.write_bytes(b"corrupt")
    write_fits(map_folder / "Maps050" / "WLconv_z0.50_0099r.fits", random_kappa(5, 32))
    cfg = load_config(write_config(tmp_path / "c.yaml", map_folder.as_posix()))
    with pytest.warns(RuntimeWarning, match="skipping"):
        meta = build_dataset(cfg, progress=False)
    assert set(meta["failed_maps"]) == {"WLconv_z0.50_0003r.fits", "WLconv_z0.50_0099r.fits"}
    rows = _read_manifest(cfg.build.output_dir)
    assert len(rows) == 9 and sum(meta["counts"].values()) == 9
    for split, n in meta["counts"].items():  # arrays were compacted to the surviving maps
        assert np.load(cfg.build.output_dir / f"kappa_{split}.npy").shape[0] == n


def test_header_angle_mismatch_aborts(tmp_path):
    folder = tmp_path / "raw"
    write_fits(folder / "WLconv_z0.50_0001r.fits", random_kappa(0), angle=5.0)
    cfg = load_config(write_config(tmp_path / "c.yaml", folder.as_posix()))
    with pytest.raises(GeometryError, match="ANGLE"):
        build_dataset(cfg, progress=False)
    assert not (cfg.build.output_dir / "dataset.json").exists()


def test_redshift_filter_and_max_maps(tmp_path):
    folder = tmp_path / "raw"
    make_map_folder(folder, n_real=6, redshifts=("0.50", "1.00"))
    cfg = load_config(write_config(tmp_path / "c.yaml", folder.as_posix(), extra_build="  redshifts: [1.0]"))
    meta = build_dataset(cfg, progress=False)
    assert sum(meta["counts"].values()) == 6
    assert all(r["redshift"] == "1.0" for r in _read_manifest(cfg.build.output_dir))

    cfg = load_config(write_config(tmp_path / "d.yaml", folder.as_posix(), "out_d",
                                   extra_build="  redshifts: null\n  max_maps: 5"))
    assert sum(build_dataset(cfg, progress=False)["counts"].values()) == 5


def test_lines_of_sight_stay_together_across_redshifts(tmp_path):
    folder = tmp_path / "raw"
    make_map_folder(folder, n_real=10, redshifts=("0.50", "1.00"))
    cfg = load_config(write_config(tmp_path / "c.yaml", folder.as_posix(), extra_build="  redshifts: null"))
    build_dataset(cfg, progress=False)
    by_real = {}
    for r in _read_manifest(cfg.build.output_dir):
        by_real.setdefault(r["realization"], set()).add(r["split"])
    assert len(by_real) == 10 and all(len(s) == 1 for s in by_real.values())


def test_config_paths_and_validation(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = load_config(write_config(cfg_dir / "c.yaml", "raw", "gen/out", root="../data"))
    assert cfg.dataset_root == (tmp_path / "data").resolve()
    assert cfg.build.source == (tmp_path / "data" / "raw").resolve()
    assert cfg.build.output_dir == (tmp_path / "data" / "gen" / "out").resolve()
    assert load_config(cfg_dir / "c.yaml", dataset_root=tmp_path).build.source == (tmp_path / "raw").resolve()
    assert cfg.observation.mask("mid").gap_height_px == (1, 3)
    assert cfg.observation.sigma_pix("high", 1.0) == pytest.approx(0.28 / np.sqrt(12))
    with pytest.raises(KeyError):
        cfg.observation.sigma_pix("extreme", 1.0)

    typo = cfg_dir / "typo.yaml"
    typo.write_text("source: raw\noutput_dir: out\nbuild:\n  max_map: 5\n")
    with pytest.raises(ValueError, match="max_map"):
        load_config(typo)


def test_example_config_loads():
    from pathlib import Path

    cfg = load_config(Path(__file__).parents[1] / "configs" / "data.yaml", dataset_root=".")
    obs = cfg.observation
    assert {"low", "mid", "high"} <= set(obs.noise_levels) and {"low", "mid", "high"} <= set(obs.mask_levels)
    area = (3.5 * 60 / 512) ** 2
    assert obs.sigma_pix("low", area) < obs.sigma_pix("mid", area) < obs.sigma_pix("high", area)
