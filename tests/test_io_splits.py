import math

import numpy as np
import pytest

from cosmicweb.data import (LeakageError, MapInfo, MapSource, assign_splits, check_leakage, load_kappa,
                            parse_map_name)

from conftest import make_map_folder, random_kappa, tar_folder, write_fits


def test_parse_map_name():
    m = parse_map_name("some/dir/WLconv_z0.50_0123r.fits")
    assert m == MapInfo("WLconv_z0.50_0123r.fits", 0.5, "0123")
    m = parse_map_name("other_map.fits")
    assert math.isnan(m.redshift) and m.realization == ""
    assert parse_map_name("WLconv_z1.00_0042r_extra.fit").realization == "0042"


def test_load_kappa_reads_angle_and_cleans_nans(tmp_path):
    data = random_kappa(0, 16)
    data[2, 3] = np.nan
    path = write_fits(tmp_path / "a.fits", data, angle=3.5)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        kappa, angle, n_bad = load_kappa(path.read_bytes(), "a.fits")
    assert kappa.dtype == np.float32 and kappa.dtype.isnative
    assert angle == 3.5 and n_bad == 1 and kappa[2, 3] == 0
    _, angle, n_bad = load_kappa(write_fits(tmp_path / "b.fits", random_kappa(0, 16), angle=None).read_bytes())
    assert angle is None and n_bad == 0
    with pytest.raises(OSError):
        load_kappa(b"not a FITS file")


def test_folder_and_tar_sources_agree(tmp_path):
    folder = tmp_path / "raw"
    make_map_folder(folder, n_real=4, redshifts=("0.50", "1.00"))
    (folder / "Maps050" / "._WLconv_z0.50_0001r.fits").write_bytes(b"macOS junk")
    (folder / "README.txt").write_text("not a map")
    tar = tar_folder(folder, tmp_path / "maps.tar")
    with MapSource(folder) as a, MapSource(tar) as b:
        assert a.maps() == b.maps()
        assert len(a.maps()) == 8
        names = [m.name for m in a.maps()]
        assert dict(a.read(names)) == dict(b.read(names))


def test_duplicate_names_rejected(tmp_path):
    write_fits(tmp_path / "x" / "m_0001r.fits", random_kappa(0, 8))
    write_fits(tmp_path / "y" / "m_0001r.fits", random_kappa(1, 8))
    with pytest.raises(ValueError, match="unique"):
        MapSource(tmp_path)


def _maps(n_real, redshifts=(0.5, 1.0)):
    return [MapInfo(f"WLconv_z{z:.2f}_{r:04d}r.fits", z, f"{r:04d}") for r in range(n_real) for z in redshifts]


def test_splits_keep_lines_of_sight_together():
    maps = _maps(100)
    split = assign_splits(maps, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert set(split) == {m.name for m in maps}
    by_real = {}
    for m in maps:
        by_real.setdefault(m.realization, set()).add(split[m.name])
    assert all(len(s) == 1 for s in by_real.values())
    counts = {s: sum(v == s for v in split.values()) // 2 for s in ("train", "val", "test")}
    assert counts == {"train": 70, "val": 15, "test": 15}


def test_splits_are_deterministic_and_seed_dependent():
    maps, f = _maps(50), {"train": 0.7, "val": 0.15, "test": 0.15}
    assert assign_splits(maps, f, 1) == assign_splits(list(reversed(maps)), f, 1)
    assert assign_splits(maps, f, 1) != assign_splits(maps, f, 2)


def test_small_split_keeps_every_split_non_empty():
    split = assign_splits(_maps(3, (0.5,)), {"train": 0.7, "val": 0.15, "test": 0.15}, 0)
    assert sorted(split.values()) == ["test", "train", "val"]


def test_maps_without_realization_are_their_own_group():
    maps = [MapInfo(f"map{i}.fits", math.nan, "") for i in range(10)]
    with pytest.warns(RuntimeWarning, match="no realization"):
        split = assign_splits(maps, {"train": 0.6, "val": 0.2, "test": 0.2}, 0)
    assert sorted(split.values()).count("train") == 6


def test_bad_fractions():
    with pytest.raises(ValueError):
        assign_splits(_maps(5), {"train": 0.7, "val": 0.2, "test": 0.2}, 0)
    with pytest.raises(ValueError):
        assign_splits(_maps(5), {"train": 1.0}, 0)


def _row(split, name, real, sha1):
    return {"split": split, "name": name, "realization": real, "sha1": sha1}


def test_leakage_check():
    ok = [_row("train", "a", "1", "h1"), _row("train", "b", "1", "h2"), _row("test", "c", "2", "h3")]
    assert check_leakage(ok) == {"train": 2, "val": 0, "test": 1}
    with pytest.raises(LeakageError, match="realization"):
        check_leakage(ok + [_row("val", "d", "1", "h4")])
    with pytest.raises(LeakageError, match="sha1"):  # identical map content under another name
        check_leakage(ok + [_row("val", "d", "3", "h1")])


def test_transient_read_errors_are_retried(tmp_path, monkeypatch):
    import cosmicweb.data.io as io_mod

    monkeypatch.setattr(io_mod, "RETRY_DELAY_S", 0.0)
    folder = tmp_path / "raw"
    make_map_folder(folder, n_real=3)
    tar = tar_folder(folder, tmp_path / "maps.tar")
    with MapSource(folder) as d:
        expected = dict(d.read([m.name for m in d.maps()]))
    with MapSource(tar, progress=True) as src:
        real, calls = src._tar.extractfile, []

        def flaky(member):  # fails once, like a Drive I/O hiccup; the reopened handle is healthy
            calls.append(member)
            if len(calls) == 1:
                raise OSError(5, "Input/output error")
            return real(member)

        src._tar.extractfile = flaky
        with pytest.warns(RuntimeWarning, match="retrying"):
            got = dict(src.read(list(expected)))
    assert got == expected


def test_persistent_read_errors_raise(tmp_path, monkeypatch):
    import types

    import cosmicweb.data.io as io_mod

    monkeypatch.setattr(io_mod, "RETRY_DELAY_S", 0.0)
    folder = tmp_path / "raw"
    make_map_folder(folder, n_real=1)
    with MapSource(folder) as src:
        name = src.maps()[0].name

        def broken():
            raise OSError(5, "Input/output error")

        src._index[name] = types.SimpleNamespace(read_bytes=broken)
        with pytest.warns(RuntimeWarning), pytest.raises(OSError):
            list(src.read([name]))
