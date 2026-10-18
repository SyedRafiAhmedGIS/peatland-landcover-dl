"""Tile bookkeeping. Tiles nest exactly inside spatial blocks, so a tile never straddles two blocks."""

from __future__ import annotations

import numpy as np

from .labels import NODATA


def tile_stats(lab: np.ndarray, tile: int, block: int, min_valid: float):
    """Return (valid fraction per tile, usable-tile mask, usable-block mask) for a label raster."""
    h, w = lab.shape
    if h % block or w % block or block % tile:
        raise ValueError("raster must be a whole number of blocks and block must be a multiple of tile")
    n_tr, n_tc, per = h // tile, w // tile, block // tile
    valid_frac = (lab != NODATA).reshape(n_tr, tile, n_tc, tile).mean((1, 3))
    tile_ok = valid_frac >= min_valid
    block_ok = tile_ok.reshape(h // block, per, w // block, per).any((1, 3))
    return valid_frac, tile_ok, block_ok


def peat_per_tile(lab: np.ndarray, tile: int, peat_class: int) -> np.ndarray:
    """Number of peat pixels in each tile."""
    h, w = lab.shape
    return (lab == peat_class).reshape(h // tile, tile, w // tile, tile).sum((1, 3))


def tile_slices(r: int, c: int, tile: int) -> tuple[slice, slice]:
    """Row/column slices of tile (r, c)."""
    return slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile)


def pixel_split_map(tile_split: np.ndarray, tile: int) -> np.ndarray:
    """Expand the tile-level split grid to a pixel-level map."""
    return np.kron(tile_split, np.ones((tile, tile), np.int8))


def epoch_indices(n_tiles: int, per_epoch: int, rng: np.random.Generator) -> np.ndarray:
    """Tile order for one training epoch: whole shuffled passes over every tile, cut to exactly `per_epoch` tiles.

    Fixes the old behaviour where `randperm(n)[:per_epoch]` silently returned only `n` tiles when `per_epoch > n`
    (e.g. 640 requested, 168 available), which made the U-Net train for ~10 gradient steps per epoch.
    """
    if n_tiles <= 0:
        raise ValueError("no training tiles")
    passes = -(-per_epoch // n_tiles)  # ceiling division
    return np.concatenate([rng.permutation(n_tiles) for _ in range(passes)])[:per_epoch]
