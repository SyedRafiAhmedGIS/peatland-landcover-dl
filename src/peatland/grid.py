"""Study-area grid on the 10 m UTM lattice. Imagery and labels are always built on this same grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import Transformer


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid: pixel shape, affine transform and EPSG code."""

    shape: tuple[int, int]
    affine: Affine
    epsg: int

    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def resolution(self) -> tuple[float, float]:
        return abs(self.affine.a), abs(self.affine.e)

    def to_dict(self) -> dict:
        return {"shape": list(self.shape), "affine": list(self.affine)[:6], "epsg": self.epsg}

    @classmethod
    def from_dict(cls, d: dict) -> Grid:
        return cls(tuple(d["shape"]), Affine(*d["affine"]), int(d["epsg"]))


def make_grid(bbox: tuple[float, float, float, float], fraction: float, res: int, epsg: int, block_px: int) -> Grid:
    """Project a lon/lat box to UTM, snap to the 10 m lattice and trim to a whole number of blocks."""
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    pts = [tr.transform(x, y) for x in (bbox[0], bbox[2]) for y in (bbox[1], bbox[3])]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w, half_h = (max(xs) - min(xs)) / 2 * fraction, (max(ys) - min(ys)) / 2 * fraction
    xmin, ymin = np.floor((cx - half_w) / res) * res, np.floor((cy - half_h) / res) * res
    width = int(2 * half_w / res) // block_px * block_px
    height = int(2 * half_h / res) // block_px * block_px
    if width == 0 or height == 0:
        raise ValueError("study area is smaller than one spatial block")
    return Grid((height, width), Affine(res, 0, xmin, 0, -res, ymin + height * res), epsg)


def grid_lonlat(grid: Grid) -> tuple[float, float, float, float]:
    """Return the grid's lon/lat bounding box (for STAC searches)."""
    tr = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    h, w = grid.shape
    pts = [tr.transform(*(grid.affine @ (c, r))) for c in (0, w) for r in (0, h)]
    return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
