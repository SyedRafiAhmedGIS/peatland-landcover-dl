"""End-to-end smoke test of preprocess -> split -> RF -> evaluate on tiny synthetic data (no network, no torch)."""

import copy
import json

import numpy as np

from peatland import pipeline
from peatland.grid import make_grid


def test_stages_run_end_to_end(cfg, synth, tmp_path):
    comp, lab = synth
    c = copy.deepcopy(cfg)
    for k in ("interim", "processed", "models", "reports"):
        c["paths"][k] = str(tmp_path / k)
    (tmp_path / "interim").mkdir()
    grid = make_grid((-7.9, 53.1, -7.5, 53.4), 1.0, 10, 32629, 32)
    grid = type(grid)(lab.shape, grid.affine, 32629)
    np.save(tmp_path / "interim/composite.npy", comp)
    np.save(tmp_path / "interim/labels_raw.npy", lab)
    (tmp_path / "interim/meta.json").write_text(json.dumps(
        {"area": "synthetic", "grid": grid.to_dict(), "label_source": "corine", "label_scheme": "test"}))
    pipeline.stage_preprocess(c)
    pipeline.stage_split(c)
    pipeline.stage_train_rf(c)
    results = pipeline.stage_evaluate(c)
    assert results["RandomForest"]["accuracy"] > 0.7
    assert json.loads((tmp_path / "reports/data_quality.json").read_text())["status"] in ("pass", "warn")
    assert (tmp_path / "reports/metrics.json").exists() and (tmp_path / "reports/05_test_maps.png").exists()
