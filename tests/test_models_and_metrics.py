import numpy as np
import pytest

from peatland.baseline import sample_pixels, train_predict_rf
from peatland.evaluate import normalised_confusion, results_table_md, score
from peatland.preprocess import build_features, normalise
from peatland.split import assign_split
from peatland.tiling import pixel_split_map, tile_stats


def test_score_handles_absent_classes_correctly():
    y = np.array([4, 4, 3, 3, 1, 1])
    p = np.array([4, 3, 3, 3, 1, 4])
    s = score(y, p)
    assert s["per_class"]["Peat bogs"]["support"] == 2
    assert s["peat_recall"] == pytest.approx(0.5) and s["peat_precision"] == pytest.approx(0.5)
    assert s["per_class"]["Water"]["support"] == 0               # absent class must not shift the peat index
    assert 0 < s["macro_f1"] <= 1


def test_score_without_peat_returns_none():
    assert score(np.array([1, 1, 2]), np.array([1, 2, 2]))["peat_f1"] is None


def test_confusion_rows_sum_to_one_for_present_classes():
    m = normalised_confusion(np.array([0, 0, 1]), np.array([0, 1, 1]))
    assert m[0].sum() == pytest.approx(1) and m[1].sum() == pytest.approx(1) and m[2].sum() == 0


def test_sampling_follows_sqrt_rule_and_is_seeded():
    lab = np.concatenate([np.zeros(10_000), np.ones(2_500), np.full(100, 2)]).astype(np.uint8).reshape(1, -1)
    mask = np.ones(lab.shape, bool)
    a = sample_pixels(lab, mask, 400, seed=1)
    counts = np.bincount(lab.ravel()[a], minlength=3)
    assert counts.tolist() == [400, 200, 40]                      # 400 * sqrt(1), sqrt(1/4), sqrt(1/100)
    assert (a == sample_pixels(lab, mask, 400, seed=1)).all()      # deterministic


def test_random_forest_learns_separable_synthetic_classes(cfg, synth):
    comp, lab = synth
    t = cfg["tiling"]
    feats, _ = build_features(comp, ["B02", "B03", "B04", "B08", "B11", "B12"])
    _, tile_ok, _ = tile_stats(lab, t["tile"], t["block"], t["min_valid_tile"])
    pix = pixel_split_map(assign_split(tile_ok, 2, t["split"], 42, 1), t["tile"])
    xn = normalise(feats, lab, pix == 0)
    _, idx, pred = train_predict_rf(xn, lab, pix, cfg)
    assert len(idx) == len(pred) > 0
    assert score(lab.ravel()[idx], pred)["accuracy"] > 0.8        # sanity check, synthetic spectra are separable


def test_results_table_uses_only_given_numbers():
    md = results_table_md({"M": {"accuracy": 0.5, "macro_f1": 0.25, "peat_f1": None, "peat_precision": None, "peat_recall": None}})
    assert "0.500" in md and "0.250" in md and "n/a" in md
