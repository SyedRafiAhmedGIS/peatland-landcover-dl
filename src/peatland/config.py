"""Load and validate config.yaml."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = ("seed", "study_areas", "imagery", "labels", "tiling", "rf", "unet", "paths")
NEEDED_BANDS = {"B04", "B08", "B11"}  # red, NIR, SWIR1 are needed for NDVI / NDMI


class ConfigError(ValueError):
    """Raised when config.yaml is missing something or contains an impossible value."""


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check the configuration and return it unchanged; raise ConfigError on any problem."""
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise ConfigError(f"config is missing required section '{key}'")
    tiling = cfg["tiling"]
    if tiling["block"] % tiling["tile"] != 0:
        raise ConfigError("tiling.block must be a multiple of tiling.tile so tiles nest inside blocks")
    fractions = tiling["split"]
    if set(fractions) != {"train", "val", "test"} or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ConfigError("tiling.split must contain train/val/test fractions that sum to 1")
    if not NEEDED_BANDS.issubset(cfg["imagery"]["bands"]):
        raise ConfigError(f"imagery.bands must include {sorted(NEEDED_BANDS)}")
    m0, m1 = (int(m) for m in cfg["imagery"]["months"])
    if not 1 <= m0 <= m1 <= 12:
        raise ConfigError("imagery.months must be [first, last] month numbers with 1 <= first <= last <= 12")
    if int(tiling.get("val_buffer_tiles", tiling["buffer_tiles"])) < 0 or int(tiling["buffer_tiles"]) < 0:
        raise ConfigError("tiling buffer sizes must be >= 0")
    if cfg["labels"]["source"] not in ("corine", "worldcover"):
        raise ConfigError("labels.source must be 'corine' or 'worldcover'")
    if not cfg["study_areas"]:
        raise ConfigError("at least one study area is required")
    for area in cfg["study_areas"]:
        lon0, lat0, lon1, lat1 = area["bbox"]
        if not (-180 <= lon0 < lon1 <= 180 and -90 <= lat0 < lat1 <= 90):
            raise ConfigError(f"invalid bbox for area '{area.get('name')}': {area['bbox']}")
    if not 0 < cfg["study_areas"][0].get("fraction", 1.0) <= 1:
        raise ConfigError("study area fraction must be in (0, 1]")
    return cfg


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def apply_mode(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return a copy of cfg with the named preset ("demo" / "full") merged in and cfg["mode"] set."""
    modes = cfg.get("modes", {})
    if mode not in modes:
        raise ConfigError(f"unknown mode '{mode}'; choose from {sorted(modes)}")
    out = _merge(cfg, modes[mode])
    out["mode"] = mode
    return validate_config(out)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read a YAML configuration file and validate it."""
    with open(path, encoding="utf-8") as handle:
        return validate_config(yaml.safe_load(handle))
