"""Spatial block split with an exclusion buffer.

Tile-grid codes: -1 unused, 0 train, 1 validation, 2 test, 3 buffer (excluded from everything).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

UNUSED, TRAIN, VAL, TEST, BUFFER = -1, 0, 1, 2, 3


def assign_split(tile_ok: np.ndarray, per_block: int, fractions: dict, seed: int, buffer_tiles: int,
                 val_buffer_tiles: int | None = None) -> np.ndarray:
    """Assign whole blocks to train/val/test, then exclude a ring of tiles around test and validation.

    `buffer_tiles` separates TEST from train and validation (the strict, reported evaluation). `val_buffer_tiles`
    separates VALIDATION from train; it defaults to `buffer_tiles`. Validation only drives early stopping, so a smaller
    value (0 in config.yaml) keeps far more training tiles while the test blocks stay fully isolated.
    """
    vbuf = buffer_tiles if val_buffer_tiles is None else val_buffer_tiles
    n_tr, n_tc = tile_ok.shape
    block_ok = tile_ok.reshape(n_tr // per_block, per_block, n_tc // per_block, per_block).any((1, 3))
    blocks = np.argwhere(block_ok)
    if len(blocks) < 3:
        raise ValueError("need at least 3 usable blocks to make a train/val/test split")
    perm = np.random.default_rng(seed).permutation(len(blocks))
    n_test = max(1, round(fractions["test"] * len(blocks)))
    n_val = max(1, round(fractions["val"] * len(blocks)))
    block_split = np.full(block_ok.shape, UNUSED, np.int8)
    for k, i in enumerate(perm):
        block_split[tuple(blocks[i])] = TEST if k < n_test else (VAL if k < n_test + n_val else TRAIN)
    tiles = np.kron(block_split, np.ones((per_block, per_block), np.int8))
    tiles[~tile_ok] = UNUSED
    struct = np.ones((3, 3), bool)
    if buffer_tiles > 0:
        near_test = ndi.binary_dilation(tiles == TEST, struct, iterations=buffer_tiles)
        tiles[near_test & ((tiles == TRAIN) | (tiles == VAL))] = BUFFER
    if vbuf > 0:
        near_val = ndi.binary_dilation(tiles == VAL, struct, iterations=vbuf)
        tiles[near_val & (tiles == TRAIN)] = BUFFER
    return tiles


def peat_shares(tiles: np.ndarray, peat_tile: np.ndarray) -> tuple[float, float, float]:
    """Fraction of the kept peat pixels that fall in train, validation and test."""
    kept = peat_tile[(tiles >= 0) & (tiles < BUFFER)].sum()
    if kept == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(float(peat_tile[tiles == k].sum() / kept) for k in (TRAIN, VAL, TEST))


def search_split(tile_ok, peat_tile, per_block, fractions, seed, buffer_tiles, tries: int = 300,
                 val_buffer_tiles: int | None = None):
    """Try consecutive seeds until train, val and test all hold peat. Uses labels only, never model scores.

    Returns (tile_split, seed_used, peat_shares, balanced).
    """
    for s in range(seed, seed + tries):
        tiles = assign_split(tile_ok, per_block, fractions, s, buffer_tiles, val_buffer_tiles)
        sh = peat_shares(tiles, peat_tile)
        if not np.isnan(sh[0]) and sh[0] >= 0.50 and sh[1] >= 0.07 and sh[2] >= 0.07:
            return tiles, s, sh, True
    tiles = assign_split(tile_ok, per_block, fractions, seed, buffer_tiles, val_buffer_tiles)
    return tiles, seed, peat_shares(tiles, peat_tile), False
