"""Shared fixtures: tiny synthetic arrays only. Tests never download data."""

import copy

import numpy as np
import pytest
from scipy import ndimage as ndi

from peatland.config import load_config

SMALL = 256  # 256 x 256 px, tile 16, block 32 -> 8 x 8 blocks


@pytest.fixture(scope="session")
def cfg():
    c = copy.deepcopy(load_config("config.yaml"))
    c["tiling"].update(tile=16, block=32, buffer_tiles=1, min_valid_tile=0.3)
    c["rf"].update(trees=8, max_px=2000)
    return c


@pytest.fixture(scope="session")
def synth(cfg):
    """Spatially smooth fake labels, class-dependent fake spectra and a nodata strip."""
    rng = np.random.default_rng(0)
    f1 = ndi.gaussian_filter(rng.normal(size=(SMALL, SMALL)), 12)
    f2 = ndi.gaussian_filter(rng.normal(size=(SMALL, SMALL)), 8)
    lab = np.full((SMALL, SMALL), 3, np.uint8)
    lab[f1 > 0.35 * f1.std()] = 1
    lab[f1 > 1.0 * f1.std()] = 2
    lab[f2 > 0.4 * f2.std()] = 4
    lab[f2 < -1.2 * f2.std()] = 6
    lab[:10, :10] = 0
    lab[-20:, :] = 255
    means = np.array([[1500, 1400, 1300, 1200, 1600, 1700], [600, 800, 700, 4000, 2500, 1500], [300, 500, 300, 3500, 1500, 700],
                      [500, 700, 600, 2800, 2300, 1500], [400, 600, 500, 2900, 1800, 1000], [350, 550, 400, 2600, 1200, 600],
                      [200, 300, 150, 100, 50, 30]], float)
    comp = np.stack([means[np.where(lab == 255, 3, lab)][..., b] + rng.normal(0, 200, lab.shape) for b in range(6)])
    return comp.astype("float32"), lab
