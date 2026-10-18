"""Regression tests for the sampling, split-buffer, date-range, verdict and config fixes (synthetic data only)."""

import copy

import numpy as np
import pytest
from scipy import ndimage as ndi

from peatland import summary
from peatland.config import ConfigError, load_config, validate_config
from peatland.download import date_range
from peatland.grid import Grid, make_grid
from peatland.quality import run_checks
from peatland.split import BUFFER, TEST, TRAIN, VAL, assign_split
from peatland.tiling import epoch_indices, tile_stats


def test_epoch_indices_returns_exactly_the_requested_number_of_tiles():
    rng = np.random.default_rng(0)
    idx = epoch_indices(168, 640, rng)                      # the old code returned only 168 here
    assert len(idx) == 640 and idx.min() >= 0 and idx.max() < 168
    counts = np.bincount(idx, minlength=168)
    assert counts.min() >= 3 and counts.max() <= 4          # every tile is used about equally often


def test_epoch_indices_with_more_tiles_than_requested_has_no_repeats():
    idx = epoch_indices(100, 40, np.random.default_rng(1))
    assert len(idx) == 40 and len(set(idx.tolist())) == 40


def test_epoch_indices_rejects_empty_training_set():
    with pytest.raises(ValueError):
        epoch_indices(0, 10, np.random.default_rng(0))


def _tile_ok(cfg, lab):
    t = cfg["tiling"]
    valid_frac, tile_ok, _ = tile_stats(lab, t["tile"], t["block"], t["min_valid_tile"])
    return valid_frac, tile_ok, t["block"] // t["tile"]


def test_zero_val_buffer_keeps_more_training_tiles_but_test_stays_isolated(cfg, synth):
    _, lab = synth
    _, tile_ok, per = _tile_ok(cfg, lab)
    fr = cfg["tiling"]["split"]
    strict = assign_split(tile_ok, per, fr, 42, 1, 1)
    relaxed = assign_split(tile_ok, per, fr, 42, 1, 0)
    assert (relaxed == TRAIN).sum() > (strict == TRAIN).sum()
    assert (relaxed == TEST).sum() == (strict == TEST).sum()
    gap = ndi.distance_transform_cdt(relaxed != TEST, metric="chessboard")[(relaxed == TRAIN) | (relaxed == VAL)].min()
    assert gap >= 2                                          # test is still separated from train AND validation
    assert (relaxed == BUFFER).any()


def test_default_val_buffer_equals_test_buffer(cfg, synth):
    _, lab = synth
    _, tile_ok, per = _tile_ok(cfg, lab)
    fr = cfg["tiling"]["split"]
    assert (assign_split(tile_ok, per, fr, 5, 1) == assign_split(tile_ok, per, fr, 5, 1, 1)).all()


def test_quality_gate_checks_all_three_spacings_and_flags_a_violation(cfg, synth):
    comp, lab = synth
    feats = np.nan_to_num(np.concatenate([comp, comp[:2]]) / 1e4).astype("float32")
    valid_frac, tile_ok, per = _tile_ok(cfg, lab)
    grid = Grid(lab.shape, make_grid((-7.9, 53.1, -7.5, 53.4), 1.0, 10, 32629, 32).affine, 32629)
    c = copy.deepcopy(cfg)
    c["tiling"]["val_buffer_tiles"] = 0
    ts = assign_split(tile_ok, per, c["tiling"]["split"], 42, 1, 0)
    names = [x["name"] for x in run_checks(grid, feats, lab, ts, valid_frac, c)["checks"]]
    assert sum("keep the buffer distance" in n for n in names) == 3
    c["tiling"]["val_buffer_tiles"] = 2                      # demand a 2-tile train-validation ring that the split does not have
    report = run_checks(grid, feats, lab, ts, valid_frac, c)
    assert any(x["status"] == "fail" and "from validation" in x["name"] for x in report["checks"])


def test_date_range_uses_the_real_last_day_of_the_month():
    assert date_range(2018, 5, 9) == "2018-05-01/2018-09-30"
    assert date_range(2018, 5, 8) == "2018-05-01/2018-08-31"      # old code stopped at the 30th
    assert date_range(2020, 2, 2) == "2020-02-01/2020-02-29"
    assert date_range(2018, 2, 2) == "2018-02-01/2018-02-28"


def test_config_rejects_bad_months_and_negative_buffers():
    base = load_config("config.yaml")
    bad = copy.deepcopy(base)
    bad["imagery"]["months"] = [9, 5]
    with pytest.raises(ConfigError):
        validate_config(bad)
    bad = copy.deepcopy(base)
    bad["tiling"]["val_buffer_tiles"] = -1
    with pytest.raises(ConfigError):
        validate_config(bad)


def _r(macro, peat, kappa):
    return {"accuracy": 0.8, "macro_f1": macro, "kappa": kappa, "peat_f1": peat, "peat_precision": 0.5, "peat_recall": 0.5,
            "peat_support_px": 20_000}


def test_headline_reports_mixed_results_instead_of_one_metric():
    results = {"RandomForest": _r(0.60, 0.33, 0.58), "UNet": _r(0.56, 0.39, 0.59)}
    msgs = summary.headline(results, {"label_source": "corine", "mode": "full"})
    text = " | ".join(m for _, m in msgs)
    assert "mixed result" in text and "peat F1" in text and "noise" in text


def test_headline_unet_clearly_better_and_clear_tie():
    better = summary.headline({"RandomForest": _r(0.5, 0.3, 0.5), "UNet": _r(0.6, 0.4, 0.6)}, {"label_source": "corine", "mode": "full"})
    assert any("U-Net outperforms the Random Forest on macro F1, peat F1, kappa" in m for _, m in better)
    tie = summary.headline({"RandomForest": _r(0.5, 0.3, 0.5), "UNet": _r(0.51, 0.31, 0.51)}, {"label_source": "corine", "mode": "full"})
    assert any("no clear difference" in m for _, m in tie)
