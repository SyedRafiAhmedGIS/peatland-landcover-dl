"""Composite cache: compact uint16 encoding and a stable key, so a repeat run skips the download."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

NODATA_U16 = 65535
KEY_FIELDS = ("year", "months", "max_cloud", "max_scenes_per_tile", "bands", "epsg")


def encode(a: np.ndarray) -> np.ndarray:
    """float32 DN -> uint16 (65535 = no data). Rounding to 1 DN is far below sensor noise."""
    return np.where(np.isfinite(a), np.clip(np.rint(a), 0, NODATA_U16 - 1), NODATA_U16).astype(np.uint16)


def decode(u: np.ndarray) -> np.ndarray:
    a = u.astype("float32")
    a[u == NODATA_U16] = np.nan
    return a


def cache_key(bbox, shape, affine6, imagery_cfg: dict) -> str:
    """Stable 10-character key for (area, grid, imagery settings)."""
    payload = [list(bbox), [int(s) for s in shape], [round(float(v), 6) for v in affine6], {k: imagery_cfg[k] for k in KEY_FIELDS}]
    return hashlib.md5(json.dumps(payload, default=str).encode()).hexdigest()[:10]


def cache_path(interim: str | Path, key: str) -> Path:
    return Path(interim) / f"composite_{key}.npz"


def save_composite(path: Path, comp: np.ndarray, meta: dict) -> np.ndarray:
    """Encode, save and return the DECODED array so cached and fresh runs are numerically identical."""
    enc = encode(comp)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, comp=enc, meta=json.dumps(meta))
    return decode(enc)


def load_composite(path: Path) -> tuple[np.ndarray, dict]:
    z = np.load(path)
    return decode(z["comp"]), json.loads(str(z["meta"]))
