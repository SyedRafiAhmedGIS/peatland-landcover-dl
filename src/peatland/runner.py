"""Run the whole pipeline from one function call. Used by the CLI and the Streamlit app."""

from __future__ import annotations

import copy
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

from . import pipeline
from .config import apply_mode, validate_config
from .download import run_download

STAGE_LABELS = {
    "download": "Sentinel-2 composite + labels",
    "preprocess": "Features (NDVI, NDMI) and label cleaning",
    "split": "Spatial block split + quality gate",
    "train-rf": "Random Forest baseline",
    "train-unet": "U-Net training",
    "evaluate": "Evaluation and figures",
}
MODEL_STAGE = {"RandomForest": "train-rf", "UNet": "train-unet"}
REPORT_FILES = ["metrics.json", "metrics_RandomForest.json", "metrics_UNet.json", "data_quality.json", "run_info.json",
                "results_table.md", "01_study_area.png", "02_spatial_split.png", "03_unet_training.png", "04_results.png",
                "05_test_maps.png"]


def build_run_config(base: dict, mode: str = "demo", area: str | None = None, custom_bbox: tuple | None = None,
                     fraction: float | None = None, seed: int | None = None, imagery: dict | None = None,
                     unet: dict | None = None) -> dict:
    """Combine config.yaml, a mode preset and the user's overrides into one validated config."""
    cfg = apply_mode(base, mode)
    if custom_bbox is not None:
        cfg["study_areas"] = [{"name": "Custom area", "bbox": [float(v) for v in custom_bbox], "fraction": fraction or 1.0}]
    elif area is not None:
        chosen = [a for a in cfg["study_areas"] if a["name"] == area]
        if not chosen:
            raise ValueError(f"unknown study area '{area}'")
        cfg["study_areas"] = chosen + [a for a in cfg["study_areas"] if a["name"] != area]   # others stay as fallbacks
    if fraction is not None:
        for a in cfg["study_areas"]:
            a["fraction"] = float(fraction)
    if seed is not None:
        cfg["seed"] = int(seed)
    cfg["imagery"].update(imagery or {})
    cfg["unet"].update(unet or {})
    return validate_config(cfg)


def plan(models: tuple[str, ...] | list[str]) -> list[str]:
    """Stage order for the chosen models (at least one model is required)."""
    models = [m for m in ("RandomForest", "UNet") if m in models]
    if not models:
        raise ValueError("choose at least one model")
    return ["download", "preprocess", "split", *[MODEL_STAGE[m] for m in models], "evaluate"]


def clean_outputs(cfg: dict, models: list[str] | tuple[str, ...]) -> None:
    """Delete outputs of earlier runs so old results can never be mixed with new ones."""
    reports, processed = Path(cfg["paths"]["reports"]), Path(cfg["paths"]["processed"])
    for f in REPORT_FILES:
        (reports / f).unlink(missing_ok=True)
    for f in ("pred_RandomForest.npy", "pred_UNet.npy", "unet_run.json", "tile_split.npy", "split.json"):
        (processed / f).unlink(missing_ok=True)
    (Path(cfg["paths"]["models"]) / "unet_best.pth").unlink(missing_ok=True)


def run_pipeline(cfg: dict, *, label_file: str | None = None, allow_proxy: bool = False,
                 models: tuple[str, ...] | list[str] = ("RandomForest", "UNet"), download_fn: Callable | None = None,
                 on_stage_start: Callable[[str, int, int], None] | None = None,
                 on_stage_end: Callable[[str, float], None] | None = None,
                 on_epoch: Callable | None = None) -> dict:
    """Run all stages, record timings and write reports/run_info.json. Returns the timings dict (seconds)."""
    stages = plan(models)
    clean_outputs(cfg, models)
    download = download_fn or (lambda c: run_download(c, label_file, allow_proxy))
    table = {
        "download": download,
        "preprocess": pipeline.stage_preprocess,
        "split": pipeline.stage_split,
        "train-rf": pipeline.stage_train_rf,
        "train-unet": lambda c: pipeline.stage_train_unet(c, on_epoch=on_epoch),
        "evaluate": pipeline.stage_evaluate,
    }
    timings: dict[str, float] = {}
    for i, name in enumerate(stages):
        if on_stage_start:
            on_stage_start(name, i, len(stages))
        t0 = time.time()
        table[name](cfg)
        timings[name] = time.time() - t0
        if on_stage_end:
            on_stage_end(name, timings[name])
    write_run_info(cfg, timings)
    return timings


def write_run_info(cfg: dict, timings: dict) -> None:
    interim, processed, reports = (Path(cfg["paths"][k]) for k in ("interim", "processed", "reports"))
    meta = json.loads((interim / "meta.json").read_text()) if (interim / "meta.json").exists() else {}
    unet = json.loads((processed / "unet_run.json").read_text()) if (processed / "unet_run.json").exists() else {}
    try:
        import torch
        torch_info = {"torch": torch.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    except Exception:
        torch_info = {"torch": None, "device": "cpu"}
    info = {"timings_s": {k: round(v, 1) for k, v in timings.items()}, "python": platform.python_version(), **torch_info,
            "mode": cfg.get("mode", "custom"), "n_scenes": meta.get("s2_n_scenes"), "composite_cached": meta.get("composite_cached"),
            "s2_processing_baselines": meta.get("s2_baselines"), "s2_offset_removed_dn": meta.get("s2_offset"),
            "unet_epochs_run": unet.get("epochs_run"), "unet_time_budget_hit": unet.get("time_budget_hit"),
            "config": copy.deepcopy({k: v for k, v in cfg.items() if k != "modes"})}
    (reports / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
