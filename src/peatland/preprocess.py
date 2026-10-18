"""Pure-numpy preprocessing: cloud mask, features, label cleaning, normalisation."""

from __future__ import annotations

import numpy as np

from .labels import NODATA

SCL_CLEAR = (4, 5, 6)  # vegetation, bare soil, water (drops cloud, shadow, snow, defective and unclassified haze)


def scl_valid(scl):
    """Boolean mask of clear pixels from a Sentinel-2 Scene Classification Layer (numpy or xarray)."""
    return scl.isin(list(SCL_CLEAR)) if hasattr(scl, "isin") else np.isin(scl, SCL_CLEAR)


def build_features(comp: np.ndarray, bands: list[str]) -> tuple[np.ndarray, list[str]]:
    """Scale DN to reflectance and append NDVI and NDMI. Returns (features[C,H,W], feature names)."""
    refl = np.clip(comp / 10000.0, 0, 1.5).astype("float32")
    b = {name: refl[i] for i, name in enumerate(bands)}
    ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 1e-6)  # greenness
    ndmi = (b["B08"] - b["B11"]) / (b["B08"] + b["B11"] + 1e-6)  # moisture, informative for wet peat
    feats = np.concatenate([refl, ndvi[None], ndmi[None]]).astype("float32")
    return feats, [*bands, "NDVI", "NDMI"]


def clean_labels(lab_raw: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Set labels to nodata (255) wherever any feature is missing."""
    lab = lab_raw.copy()
    lab[~np.isfinite(feats).all(0)] = NODATA
    return lab


def normalise(feats: np.ndarray, lab: np.ndarray, train_pixels: np.ndarray) -> np.ndarray:
    """Standardise each feature with statistics from TRAINING pixels only (no test information leaks)."""
    sel = train_pixels & (lab != NODATA)
    mu = feats[:, sel].mean(1)
    sd = feats[:, sel].std(1) + 1e-6
    out = ((feats - mu[:, None, None]) / sd[:, None, None]).astype("float32")
    out[~np.isfinite(out)] = 0
    return out
