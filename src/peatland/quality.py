"""Data-quality gate. Checks are recomputed from the final arrays, independent of how the split was built."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .grid import Grid
from .labels import CLASS_NAMES, NODATA, PEAT
from .split import BUFFER, TEST, TRAIN, VAL


class QualityError(AssertionError):
    """Raised when a critical data-quality check fails."""


def run_checks(grid: Grid, feats: np.ndarray, lab: np.ndarray, tile_split: np.ndarray, valid_frac: np.ndarray,
               cfg: dict) -> dict:
    """Run all checks and return the report dict (does not raise)."""
    tile, block = cfg["tiling"]["tile"], cfg["tiling"]["block"]
    per, buf = block // tile, cfg["tiling"]["buffer_tiles"]
    vbuf = cfg["tiling"].get("val_buffer_tiles", buf)
    checks: list[dict] = []

    def chk(name: str, passed: bool, detail: str, critical: bool = True) -> None:
        checks.append({"name": name, "status": "pass" if passed else ("fail" if critical else "warn"), "detail": detail})

    rx, ry = grid.resolution
    chk("CRS matches the configured projected CRS", grid.crs == f"EPSG:{cfg['imagery']['epsg']}", grid.crs)
    chk("Pixel size matches config", abs(rx - cfg["imagery"]["resolution"]) < 1e-6 and abs(ry - rx) < 1e-6, f"{rx} x {ry} m")
    chk("Image and label shapes match the grid", feats.shape[1:] == lab.shape == tuple(grid.shape), f"{feats.shape} vs {lab.shape}")
    labelled = lab != NODATA
    if feats.shape[1:] != lab.shape:  # cannot index features with the label mask; report instead of crashing
        chk("No NaN/Inf in features at labelled pixels", False, "skipped: feature and label shapes differ")
    else:
        chk("No NaN/Inf in features at labelled pixels", bool(np.isfinite(feats[:, labelled]).all()), "every labelled pixel checked")
    present = np.unique(lab)
    chk("Labels contain only classes 0-6 or 255", bool(np.isin(present, [*range(7), NODATA]).all()), str(present.tolist()))

    n_tr, n_tc = tile_split.shape
    v = tile_split.reshape(n_tr // per, per, n_tc // per, per).transpose(0, 2, 1, 3).reshape(n_tr // per, n_tc // per, -1)
    in_split = (v >= TRAIN) & (v <= TEST)
    lo, hi = np.where(in_split, v, 9).min(-1), np.where(in_split, v, -9).max(-1)
    chk("No spatial block appears in two splits", bool(((lo == hi) | ~in_split.any(-1)).all()),
        f"{int(in_split.any(-1).sum())} blocks checked")
    def min_gap(a: int, b: int) -> int | None:
        """Smallest chessboard distance (in tiles) from any tile of split b to the nearest tile of split a."""
        if not ((tile_split == a).any() and (tile_split == b).any()):
            return None
        return int(ndi.distance_transform_cdt(tile_split != a, metric="chessboard")[tile_split == b].min())

    for a, b, an, bn, need in ((TEST, TRAIN, "test", "train", buf + 1), (TEST, VAL, "test", "validation", buf + 1),
                               (VAL, TRAIN, "validation", "train", vbuf + 1)):
        gap = min_gap(a, b)
        if gap is not None:
            chk(f"{bn.capitalize()} tiles keep the buffer distance from {an} tiles", gap >= need, f"min gap {gap} tiles (need {need})")
    chk("Train, validation and test are all non-empty", all((tile_split == k).any() for k in (TRAIN, VAL, TEST)),
        str([int((tile_split == k).sum()) for k in (TRAIN, VAL, TEST)]))

    pix = np.kron(tile_split, np.ones((tile, tile), np.int8))
    counts: dict[str, list[int]] = {}
    for k, nm in ((TRAIN, "train"), (VAL, "val"), (TEST, "test")):
        counts[nm] = np.bincount(lab[(pix == k) & labelled], minlength=7).tolist()
        chk(f"Peat bogs present in {nm}", counts[nm][PEAT] > 0, f"{counts[nm][PEAT]:,} px", critical=False)
        missing = [CLASS_NAMES[i] for i in range(7) if counts[nm][i] == 0]
        chk(f"All 7 classes present in {nm}", not missing, "missing: " + (", ".join(missing) or "none"), critical=False)
    kept = (tile_split >= TRAIN) & (tile_split < BUFFER)
    nodata = 1 - valid_frac[kept] if kept.any() else np.array([1.0])
    chk("Nodata fraction in kept tiles is at most 70%", float(nodata.max()) <= 0.70, f"max {nodata.max() * 100:.1f}%")

    status = "fail" if any(c["status"] == "fail" for c in checks) else ("warn" if any(c["status"] == "warn" for c in checks) else "pass")
    per_split = {k: dict(zip(CLASS_NAMES, v, strict=True)) for k, v in counts.items()}
    return {"status": status, "checks": checks, "class_counts_per_split": per_split}


def write_report(report: dict, path: str | Path, extra: dict | None = None) -> None:
    """Write the report to JSON and raise QualityError if a critical check failed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({**(extra or {}), **report}, indent=2))
    if report["status"] == "fail":
        failed = [c["name"] for c in report["checks"] if c["status"] == "fail"]
        raise QualityError(f"Data-quality gate FAILED: {failed}")
