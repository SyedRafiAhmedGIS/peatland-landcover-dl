"""Stage functions that connect the modules; each reads and writes files so stages can be run one at a time."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import viz
from .baseline import train_predict_rf
from .cache import load_composite
from .evaluate import normalised_confusion, results_table_md, score, write_metrics
from .grid import Grid
from .labels import NODATA, PEAT
from .preprocess import build_features, clean_labels, normalise
from .quality import run_checks, write_report
from .split import search_split
from .tiling import peat_per_tile, pixel_split_map, tile_stats


def _dirs(cfg: dict) -> tuple[Path, Path, Path]:
    interim, processed, reports = (Path(cfg["paths"][k]) for k in ("interim", "processed", "reports"))
    for d in (interim, processed, reports):
        d.mkdir(parents=True, exist_ok=True)
    return interim, processed, reports


def stage_preprocess(cfg: dict) -> None:
    """Composite + raw labels -> features, cleaned labels and the study-area figure."""
    interim, processed, reports = _dirs(cfg)
    meta = json.loads((interim / "meta.json").read_text())
    if meta.get("composite_file") and Path(meta["composite_file"]).exists():
        comp, _ = load_composite(Path(meta["composite_file"]))
    else:
        comp = np.load(interim / "composite.npy")
    lab_raw = np.load(interim / "labels_raw.npy")
    feats, names = build_features(comp, cfg["imagery"]["bands"])
    lab = clean_labels(lab_raw, feats)
    np.save(processed / "features.npy", feats)
    np.save(processed / "labels.npy", lab)
    (processed / "features.json").write_text(json.dumps(names))

    b = {n: comp[i] / 10000.0 for i, n in enumerate(cfg["imagery"]["bands"])}

    def stretch(a: np.ndarray) -> np.ndarray:
        lo, hi = np.nanpercentile(a, (2, 98))
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    rgb = np.stack([stretch(b["B04"]), stretch(b["B03"]), stretch(b["B02"])], -1)
    viz.plot_study_area(rgb, lab, f"7-class labels ({meta['label_source']})", reports / "01_study_area.png")
    counts = np.bincount(lab[lab != NODATA], minlength=7)
    print(f"Features {feats.shape}; labelled pixels {int(counts.sum()):,}; peat pixels {int(counts[PEAT]):,}")


def stage_split(cfg: dict) -> None:
    """Spatial block split + data-quality gate (raises if a critical check fails)."""
    interim, processed, reports = _dirs(cfg)
    feats, lab = np.load(processed / "features.npy"), np.load(processed / "labels.npy")
    meta = json.loads((interim / "meta.json").read_text())
    t = cfg["tiling"]
    valid_frac, tile_ok, _ = tile_stats(lab, t["tile"], t["block"], t["min_valid_tile"])
    peat_tile = peat_per_tile(lab, t["tile"], PEAT)
    tiles, seed, shares, balanced = search_split(
        tile_ok, peat_tile, t["block"] // t["tile"], t["split"], cfg["seed"], t["buffer_tiles"],
        val_buffer_tiles=t.get("val_buffer_tiles", t["buffer_tiles"]))
    if not balanced:
        print("WARNING: no peat-balanced split found; using the default seed.")
    np.save(processed / "tile_split.npy", tiles)
    (processed / "split.json").write_text(json.dumps({"seed": seed, "peat_shares": [None if np.isnan(s) else s for s in shares]}))
    viz.plot_split(tiles, seed, reports / "02_spatial_split.png")
    grid = Grid.from_dict(meta["grid"])
    report = run_checks(grid, feats, lab, tiles, valid_frac, cfg)
    write_report(report, reports / "data_quality.json",
                 {"label_source": meta["label_source"], "area": meta["area"], "split_seed": seed})
    kept = {n: int((tiles == k).sum()) for n, k in (("train", 0), ("val", 1), ("test", 2), ("buffer", 3))}
    print(f"Split seed {seed}; tiles {kept}; quality gate: {report['status']}")


def load_context(cfg: dict) -> dict:
    """Load features, labels, split and the training-only normalised features."""
    interim, processed, _ = _dirs(cfg)
    feats, lab = np.load(processed / "features.npy"), np.load(processed / "labels.npy")
    tiles = np.load(processed / "tile_split.npy")
    pix = pixel_split_map(tiles, cfg["tiling"]["tile"])
    xn = normalise(feats, lab, pix == 0)
    meta = json.loads((interim / "meta.json").read_text())
    return {"lab": lab, "tiles": tiles, "pix": pix, "xn": xn, "meta": meta}


def stage_train_rf(cfg: dict) -> None:
    _, processed, _ = _dirs(cfg)
    ctx = load_context(cfg)
    _, test_idx, pred = train_predict_rf(ctx["xn"], ctx["lab"], ctx["pix"], cfg)
    full = np.full(ctx["lab"].size, NODATA, np.uint8)
    full[test_idx] = pred
    np.save(processed / "pred_RandomForest.npy", full.reshape(ctx["lab"].shape))
    print(f"Random Forest predicted {len(test_idx):,} test pixels")


def stage_train_unet(cfg: dict, on_epoch=None) -> None:
    from .unet import train_unet

    _, processed, reports = _dirs(cfg)
    ctx = load_context(cfg)
    pred, hist, info = train_unet(ctx["xn"], ctx["lab"], ctx["tiles"], cfg, cfg["paths"]["models"], on_epoch=on_epoch)
    np.save(processed / "pred_UNet.npy", pred)
    (processed / "unet_run.json").write_text(json.dumps({**info, "history": hist}))
    viz.plot_training(hist, reports / "03_unet_training.png")
    if info["epochs_run"] < info["min_epochs"]:
        print(f"WARNING: U-Net ran only {info['epochs_run']} epochs (minimum {info['min_epochs']}) - under-trained.")


def stage_evaluate(cfg: dict) -> dict:
    """Score every saved prediction on the same test pixels and write metrics, figures and a README table."""
    interim, processed, reports = _dirs(cfg)
    lab = np.load(processed / "labels.npy")
    pix = pixel_split_map(np.load(processed / "tile_split.npy"), cfg["tiling"]["tile"])
    meta = json.loads((interim / "meta.json").read_text())
    test_mask = (pix == 2) & (lab != NODATA)
    maps = {n: np.load(processed / f"pred_{n}.npy") for n in ("RandomForest", "UNet") if (processed / f"pred_{n}.npy").exists()}
    if not maps:
        raise SystemExit("No predictions found. Run train-rf and/or train-unet first.")
    y_true = lab[test_mask]
    preds = {n: m[test_mask] for n, m in maps.items()}
    results = {n: score(y_true, p) for n, p in preds.items()}
    for n, p in preds.items():
        results[n]["confusion_normalised"] = normalised_confusion(y_true, p).round(4).tolist()
    split_info = json.loads((processed / "split.json").read_text()) if (processed / "split.json").exists() else {}
    unet_run = json.loads((processed / "unet_run.json").read_text()) if (processed / "unet_run.json").exists() else {}
    note = ("Labels are CORINE (100 m, 25 ha minimum mapping unit)." if meta["label_source"] == "corine"
            else "PROXY labels: ESA WorldCover has no peat class (herbaceous wetland substituted).")
    write_metrics(results, {"label_source": meta["label_source"], "area": meta["area"], "test_pixels": int(len(y_true)),
                            "seed": cfg["seed"], "note": note,
                            "split_seed": split_info.get("seed"), "mode": cfg.get("mode", "custom"),
                            "unet_epochs_run": unet_run.get("epochs_run") if "UNet" in results else None,
                            "unet_time_budget_hit": unet_run.get("time_budget_hit") if "UNet" in results else None,
                            "unet_converged": bool(results["UNet"]["kappa"] >= 0.10) if "UNet" in results else None}, reports)
    best = max(results, key=lambda k: results[k]["peat_f1"] or -1)
    viz.plot_results(results, y_true, preds, reports / "04_results.png")
    viz.plot_test_maps(lab, maps, test_mask, best, reports / "05_test_maps.png")
    (reports / "results_table.md").write_text(results_table_md(results) + "\n")
    print(results_table_md(results))
    return results
