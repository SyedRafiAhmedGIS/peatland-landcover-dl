"""Random Forest baseline on per-pixel spectral features."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .labels import NODATA


def sample_pixels(lab: np.ndarray, train_mask: np.ndarray, max_px: int, seed: int) -> np.ndarray:
    """Square-root-balanced random sample of training pixel indices (flat).

    The largest class gets `max_px` pixels and every other class about max_px * sqrt(size / largest) - the same effective
    class prior as the U-Net's square-root class weights, so the two models are compared fairly.
    """
    rng = np.random.default_rng(seed)
    flat_lab, flat_mask = lab.ravel(), train_mask.ravel()
    n_max = int(np.bincount(flat_lab[flat_mask], minlength=7)[:7].max()) if flat_mask.any() else 0
    picks = []
    for c in range(7):
        idx = np.flatnonzero(flat_mask & (flat_lab == c))
        if len(idx):
            k = max(1, min(len(idx), int(max_px * np.sqrt(len(idx) / n_max))))
            picks.append(rng.choice(idx, k, replace=False))
    if not picks:
        raise ValueError("no labelled training pixels")
    return np.concatenate(picks)


def train_predict_rf(xn: np.ndarray, lab: np.ndarray, pix_split: np.ndarray, cfg: dict):
    """Train on sqrt-balanced training pixels; predict every test pixel. Returns (model, test indices, predictions)."""
    seed = cfg["seed"]
    train_mask = (pix_split == 0) & (lab != NODATA)
    picks = sample_pixels(lab, train_mask, cfg["rf"]["max_px"], seed)
    flat = xn.reshape(xn.shape[0], -1)
    model = RandomForestClassifier(
        n_estimators=cfg["rf"]["trees"], max_depth=cfg["rf"]["max_depth"], min_samples_leaf=3,
        class_weight=None, n_jobs=-1, random_state=seed,
    ).fit(flat[:, picks].T, lab.ravel()[picks])
    test_idx = np.flatnonzero(((pix_split == 2) & (lab != NODATA)).ravel())
    pred = np.empty(len(test_idx), np.uint8)
    for i in range(0, len(test_idx), 400_000):  # batches keep memory flat
        pred[i:i + 400_000] = model.predict(flat[:, test_idx[i:i + 400_000]].T)
    return model, test_idx, pred
