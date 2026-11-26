"""Train/val/test splits grouped by line of sight, and leakage checks.

All maps that share a realization ID (for example the same line of sight at
different source redshifts) always go to the same split.

Caveat: MassiveNuS builds its many ray-traced realizations from one N-body run
per cosmology, so different realizations are not fully independent. Splitting by
realization removes exact-duplicate structure, not all correlation.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath

import numpy as np

from .io import MapInfo

__all__ = ["SPLITS", "assign_splits", "check_leakage", "LeakageError"]

SPLITS = ("train", "val", "test")


class LeakageError(RuntimeError):
    pass


def _group_key(m: MapInfo) -> str:
    return f"r:{m.realization}" if m.realization else f"name:{PurePosixPath(m.name).stem}"


def assign_splits(maps: Iterable[MapInfo], fractions: Mapping[str, float], seed: int) -> dict[str, str]:
    """Return ``{map name: split}``, shuffling whole realization groups with ``seed``."""
    if set(fractions) != set(SPLITS):
        raise ValueError(f"split fractions need exactly the keys {SPLITS}")
    if any(f < 0 for f in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("split fractions must be >= 0 and sum to 1")

    groups: dict[str, list[str]] = {}
    unlabelled = 0
    for m in maps:
        groups.setdefault(_group_key(m), []).append(m.name)
        unlabelled += not m.realization
    if unlabelled:
        warnings.warn(f"{unlabelled} map(s) have no realization ID in their name; each is "
                      "treated as its own line of sight", RuntimeWarning, stacklevel=2)

    keys = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(keys))
    n = len(keys)
    n_val = round(fractions["val"] * n)
    n_test = round(fractions["test"] * n)
    if n >= 3:  # keep every requested split non-empty
        n_val = max(n_val, 1) if fractions["val"] > 0 else 0
        n_test = max(n_test, 1) if fractions["test"] > 0 else 0
    n_train = n - n_val - n_test
    if n_train < 0 or (fractions["train"] > 0 and n_train == 0):
        raise ValueError(f"only {n} line(s) of sight: too few to split {dict(fractions)}")

    bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}
    out: dict[str, str] = {}
    for split, (a, b) in bounds.items():
        for i in order[a:b]:
            for name in groups[keys[i]]:
                out[name] = split
    return out


def check_leakage(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    """Raise :class:`LeakageError` if any map, line of sight or identical map content
    appears in more than one split. ``rows`` need ``split``, ``name``,
    ``realization`` and ``sha1`` keys. Returns per-split map counts.
    """
    seen: dict[tuple[str, str], set[str]] = {}
    counts = dict.fromkeys(SPLITS, 0)
    for row in rows:
        counts[row["split"]] += 1
        for kind in ("name", "realization", "sha1"):
            if row[kind]:
                seen.setdefault((kind, row[kind]), set()).add(row["split"])
    problems = [f"{kind} {value!r} is in {sorted(splits)}"
                for (kind, value), splits in sorted(seen.items()) if len(splits) > 1]
    if problems:
        raise LeakageError(f"{len(problems)} leak(s) between splits, e.g. " + "; ".join(problems[:5]))
    return counts
