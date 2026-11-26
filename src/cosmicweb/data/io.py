"""Find and read convergence maps from a folder of FITS files or a .tar archive.

Maps are identified by file basename (e.g. ``WLconv_z0.50_0001r.fits``), so a tar
archive and the same files extracted to a folder build byte-identical datasets.
"""

from __future__ import annotations

import io
import math
import re
import tarfile
import time
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from astropy.io import fits

__all__ = ["MapInfo", "parse_map_name", "load_kappa", "MapSource"]

FITS_SUFFIXES = (".fits", ".fit")
READ_ATTEMPTS = 4
RETRY_DELAY_S = 2.0  # doubled after each failed attempt
_Z_RE = re.compile(r"_z(\d+(?:\.\d+)?)", re.IGNORECASE)
_R_RE = re.compile(r"_(\d+)r(?=[._]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class MapInfo:
    name: str  # file basename; unique within a dataset
    redshift: float  # nan when the name has no _z<value>
    realization: str  # line-of-sight realization ID, "" when the name has none


def parse_map_name(name: str) -> MapInfo:
    """Parse MassiveNuS-style names such as ``WLconv_z0.50_0001r.fits``."""
    base = PurePosixPath(name.replace("\\", "/")).name
    z, r = _Z_RE.search(base), _R_RE.search(base)
    return MapInfo(base, float(z.group(1)) if z else math.nan, r.group(1) if r else "")


def _is_map_name(base: str) -> bool:
    # "._*" are macOS resource-fork files that often end up inside tar archives
    return base.lower().endswith(FITS_SUFFIXES) and not base.startswith("._")


def load_kappa(data: bytes, name: str = "<bytes>") -> tuple[np.ndarray, float | None, int]:
    """Read a 2-D map from FITS bytes.

    Returns ``(kappa, angle_deg, n_nonfinite)``: native-endian float32 with NaN/inf
    replaced by 0, the side length from the ``ANGLE`` header keyword if present, and
    how many pixels were replaced.
    """
    with fits.open(io.BytesIO(data), memmap=False) as hdul:
        hdu = next((h for h in hdul if h.data is not None and np.ndim(h.data) == 2), None)
        if hdu is None:
            raise ValueError(f"{name}: no 2-D image in any HDU")
        kappa = np.array(hdu.data, dtype=np.float32)
        angle = hdu.header.get("ANGLE", hdul[0].header.get("ANGLE"))
    bad = ~np.isfinite(kappa)
    n_bad = int(bad.sum())
    if n_bad:
        warnings.warn(f"{name}: {n_bad} non-finite pixels set to 0", RuntimeWarning, stacklevel=2)
        kappa[bad] = 0.0
    return kappa, (float(angle) if angle is not None else None), n_bad


class MapSource:
    """A set of FITS maps in a folder (searched recursively) or a tar archive.

    Use as a context manager. ``maps()`` lists them; ``read(names)`` yields
    ``(name, bytes)`` in the order that is cheapest for the storage (archive order
    for tar files, so each member is read once, front to back).

    Reads that fail with an I/O error (for example a network-mounted Google
    Drive hiccup) are retried a few times before giving up.
    """

    def __init__(self, path: str | Path, progress: bool = False):
        self.path = Path(path)
        self._tar: tarfile.TarFile | None = None
        self._index: dict[str, object] = {}
        if self.path.is_dir():
            entries = ((p.name, p) for p in self.path.rglob("*") if p.is_file())
        elif self.path.is_file() and tarfile.is_tarfile(self.path):
            self._tar = tarfile.open(self.path, "r:*")
            entries = ((PurePosixPath(m.name).name, m) for m in self._scan_tar(progress) if m.isfile())
        else:
            raise FileNotFoundError(f"{self.path} is neither a folder nor a tar archive")
        for base, ref in entries:
            if not _is_map_name(base):
                continue
            if base in self._index:
                raise ValueError(f"two maps named {base!r} in {self.path}; map names must be unique")
            self._index[base] = ref

    def _scan_tar(self, progress: bool) -> Iterator[tarfile.TarInfo]:
        """Read every member header once (needed to list a tar), showing bytes scanned."""
        bar = None
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=self.path.stat().st_size, unit="B", unit_scale=True, desc="Indexing tar")
            except ImportError:
                pass
        try:
            for member in self._tar:  # iterating also caches members, so reads never rescan
                if bar is not None:
                    bar.update(max(0, member.offset_data + member.size - bar.n))
                yield member
        finally:
            if bar is not None:
                bar.close()

    def _read_with_retry(self, name: str, read):
        for attempt in range(READ_ATTEMPTS):
            try:
                return read()
            except OSError as exc:
                if attempt == READ_ATTEMPTS - 1:
                    raise
                warnings.warn(f"{name}: read failed ({exc}); retrying", RuntimeWarning, stacklevel=3)
                time.sleep(RETRY_DELAY_S * 2**attempt)
                if self._tar is not None:  # reopen: the old handle may be broken. Cached TarInfo
                    self._tar.close()      # objects still work, so this does not rescan.
                    self._tar = tarfile.open(self.path, "r:*")

    def __enter__(self) -> MapSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def maps(self) -> list[MapInfo]:
        return [parse_map_name(n) for n in sorted(self._index)]

    def read(self, names: Iterable[str]) -> Iterator[tuple[str, bytes]]:
        names = list(names)
        missing = [n for n in names if n not in self._index]
        if missing:
            raise KeyError(f"maps not in {self.path}: {missing[:5]}")
        if self._tar is None:
            for n in sorted(names):
                yield n, self._read_with_retry(n, self._index[n].read_bytes)
        else:
            for n in sorted(names, key=lambda n: self._index[n].offset):
                # extractfile(TarInfo) seeks straight to the member: no rescan
                yield n, self._read_with_retry(n, lambda m=self._index[n]: self._tar.extractfile(m).read())
