"""Build the frozen κ dataset from MassiveNuS FITS maps.

Output layout (``output_dir``)::

    kappa_train.npy  kappa_val.npy  kappa_test.npy   float32 (N, H, W), clean κ
    manifest.csv        one row per map: split, row, name, realization, sha1, ...
    splits/<split>.csv  map names per split (small enough to share or commit)
    dataset.json        geometry, counts, fingerprint, leakage check, build config

Only clean κ is stored. Noisy, masked inputs are generated on the fly by
:class:`cosmicweb.data.dataset.ShearMapDataset`, so every noise × mask condition
reuses the same files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .config import DataConfig
from .io import MapSource, load_kappa
from .noise import pixel_area_arcmin2
from .observation import map_key
from .splits import SPLITS, assign_splits, check_leakage

__all__ = ["build_dataset", "MANIFEST_FIELDS"]

MANIFEST_FIELDS = ("split", "row", "name", "realization", "redshift", "map_key",
                   "kappa_mean", "kappa_std", "n_nonfinite", "sha1")
FORMAT_VERSION = 1


class GeometryError(ValueError):
    """The maps do not match the configured geometry; the whole build is invalid."""


def _replace(src: Path, dst: Path, attempts: int = 30) -> None:
    """``os.replace`` that retries briefly on Windows, where antivirus or indexing
    services can hold a freshly written file open for a moment."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or i == attempts - 1:
                raise
            time.sleep(min(0.05 * 2**i, 1.0))


def _progress(it, total: int, enabled: bool):
    if not enabled:
        return it
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return it
    return tqdm(it, total=total, desc="Reading maps", unit="map")


def build_dataset(cfg: DataConfig, *, overwrite: bool = False, progress: bool = True) -> dict:
    """Build the dataset described by ``cfg.build``; return the ``dataset.json`` contents."""
    bc = cfg.build
    out = bc.output_dir
    if (out / "dataset.json").exists() and not overwrite:
        raise FileExistsError(f"{out} already holds a built dataset; pass overwrite=True to rebuild")

    with MapSource(bc.source) as src:
        maps = src.maps()
        if bc.redshifts is not None:
            maps = [m for m in maps if any(math.isclose(m.redshift, z, abs_tol=1e-6) for z in bc.redshifts)]
        if bc.max_maps is not None:
            maps = maps[:bc.max_maps]
        if not maps:
            raise FileNotFoundError(f"no FITS maps with redshift in {bc.redshifts} found in {bc.source}")

        split_of = assign_splits(maps, bc.split_fractions, bc.seed)
        info = {m.name: m for m in maps}
        names_by_split = {s: sorted(n for n, sp in split_of.items() if sp == s) for s in SPLITS}
        row_of = {n: i for s in SPLITS for i, n in enumerate(names_by_split[s])}

        partial = out / ".partial"
        if partial.exists():
            shutil.rmtree(partial)
        (partial / "splits").mkdir(parents=True)

        arrays: dict[str, np.memmap] = {}
        stats: dict[str, dict] = {}
        failed: dict[str, str] = {}
        shape = None
        for name, data in _progress(src.read(info), len(info), progress):
            try:
                kappa, angle, n_bad = load_kappa(data, name)
                if shape is None:
                    if kappa.shape[0] != kappa.shape[1]:
                        raise GeometryError(f"{name}: maps must be square, got {kappa.shape}")
                    shape = kappa.shape
                    for s in SPLITS:
                        arrays[s] = open_memmap(partial / f"kappa_{s}.npy", mode="w+", dtype=np.float32,
                                                shape=(len(names_by_split[s]), *shape))
                if angle is not None and not math.isclose(angle, bc.map_size_deg, rel_tol=1e-3):
                    raise GeometryError(f"{name}: FITS header ANGLE={angle} deg but config "
                                        f"map_size_deg={bc.map_size_deg}")
                if kappa.shape != shape:
                    raise ValueError(f"shape {kappa.shape} differs from first map {shape}")
            except GeometryError:
                raise
            except Exception as exc:  # unreadable or inconsistent map: record and skip
                failed[name] = f"{type(exc).__name__}: {exc}"
                warnings.warn(f"skipping {name}: {failed[name]}", RuntimeWarning, stacklevel=2)
                continue
            arrays[split_of[name]][row_of[name]] = kappa
            stats[name] = {"kappa_mean": float(kappa.mean(dtype=np.float64)),
                           "kappa_std": float(kappa.std(dtype=np.float64)),
                           "n_nonfinite": n_bad,
                           "sha1": hashlib.sha1(kappa.tobytes()).hexdigest()}

    if not stats:
        raise RuntimeError(f"no map could be read; first error: {next(iter(failed.values()))}")

    rows = []
    for s in SPLITS:
        kept = [n for n in names_by_split[s] if n in stats]
        arr = arrays.pop(s)
        arr.flush()
        if len(kept) != len(names_by_split[s]):  # drop rows of failed maps
            final = partial / f"kappa_{s}.compact.npy"
            compact = open_memmap(final, mode="w+", dtype=np.float32, shape=(len(kept), *shape))
            for i, n in enumerate(kept):
                compact[i] = arr[row_of[n]]
            compact.flush()
            del compact, arr  # release the mappings before replacing files (required on Windows)
            _replace(final, partial / f"kappa_{s}.npy")
        else:
            del arr
        for i, n in enumerate(kept):
            m = info[n]
            rows.append({"split": s, "row": i, "name": n, "realization": m.realization,
                         "redshift": m.redshift, "map_key": map_key(n), **stats[n]})

    counts = check_leakage(rows)
    with (partial / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    for s in SPLITS:
        with (partial / "splits" / f"{s}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "realization", "redshift"])
            w.writerows((r["name"], r["realization"], r["redshift"]) for r in rows if r["split"] == s)

    fingerprint = hashlib.sha256("".join(f"{r['split']},{r['row']},{r['sha1']}\n" for r in rows).encode())
    build = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(bc).items()}
    meta = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fingerprint.hexdigest(),
        "map_shape": list(shape),
        "map_size_deg": bc.map_size_deg,
        "pixel_area_arcmin2": pixel_area_arcmin2(bc.map_size_deg, shape[0]),
        "counts": counts,
        "leakage_check": "passed",
        "failed_maps": failed,
        "build": build,
    }
    text = json.dumps(meta, indent=2)
    (partial / "dataset.json").write_text(text, encoding="utf-8")

    out.mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(exist_ok=True)
    (out / "dataset.json").unlink(missing_ok=True)  # an interrupted overwrite must not look complete
    for f in sorted(partial.rglob("*")):
        if f.is_file() and f.name != "dataset.json":
            _replace(f, out / f.relative_to(partial))
    _replace(partial / "dataset.json", out / "dataset.json")  # last: marks the build complete
    shutil.rmtree(partial)
    return json.loads(text)  # exactly what dataset.json holds
