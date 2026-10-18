import json

import numpy as np
import pytest

from peatland import runner, summary
from peatland.config import ConfigError, apply_mode, load_config
from peatland.grid import make_grid


def _res(kappa=0.5, peat_px=20_000):
    return {"accuracy": 0.8, "macro_f1": 0.6, "kappa": kappa, "peat_f1": 0.5, "peat_precision": 0.5, "peat_recall": 0.5,
            "peat_support_px": peat_px}


def test_headline_flags_proxy_demo_nonconvergence_and_few_peat_pixels():
    msgs = summary.headline({"RandomForest": _res(peat_px=100), "UNet": _res(kappa=0.02, peat_px=100)},
                            {"label_source": "worldcover_2021", "mode": "demo", "unet_time_budget_hit": True})
    text = " | ".join(m for _, m in msgs)
    assert "PROXY" in text and "DEMO" in text and "did not converge" in text and "time cap" in text and "Only 100 peat pixels" in text
    assert "comparison is invalid" in text


def test_headline_clean_corine_full_run_has_no_warnings():
    msgs = summary.headline({"RandomForest": _res(), "UNet": _res()}, {"label_source": "corine", "mode": "full"})
    assert {lvl for lvl, _ in msgs} == {"info"}


def test_apply_mode_merges_presets_without_mutating_base():
    base = load_config("config.yaml")
    demo = apply_mode(base, "demo")
    assert demo["imagery"]["max_scenes_per_tile"] == 4 and demo["rf"]["max_px"] == 80000 and demo["mode"] == "demo"
    assert base["imagery"]["max_scenes_per_tile"] == 6 and "mode" not in base
    assert demo["imagery"]["bands"] == base["imagery"]["bands"]          # untouched keys survive the merge
    assert apply_mode(base, "full")["rf"]["max_px"] == 150000
    with pytest.raises(ConfigError):
        apply_mode(base, "turbo")


def test_build_run_config_applies_user_choices():
    base = load_config("config.yaml")
    c = runner.build_run_config(base, mode="demo", area="Connemara blanket bog", fraction=0.5, seed=7,
                                imagery={"year": 2019}, unet={"epochs": 4})
    assert c["study_areas"][0]["name"] == "Connemara blanket bog" and len(c["study_areas"]) == 2   # other area kept as fallback
    assert all(a["fraction"] == 0.5 for a in c["study_areas"]) and c["seed"] == 7
    assert c["imagery"]["year"] == 2019 and c["unet"]["epochs"] == 4
    custom = runner.build_run_config(base, custom_bbox=(-8, 53, -7.5, 53.3))
    assert [a["name"] for a in custom["study_areas"]] == ["Custom area"]
    with pytest.raises(ConfigError):
        runner.build_run_config(base, custom_bbox=(-7, 53, -8, 54))       # min > max
    with pytest.raises(ValueError):
        runner.build_run_config(base, area="Atlantis")


def test_plan_orders_stages_and_requires_a_model():
    assert runner.plan(["UNet", "RandomForest"]) == ["download", "preprocess", "split", "train-rf", "train-unet", "evaluate"]
    assert runner.plan(["RandomForest"])[-2:] == ["train-rf", "evaluate"]
    with pytest.raises(ValueError):
        runner.plan([])


def _fake_download(cfg, synth, tmp_path):
    comp, lab = synth

    def fake(c):
        grid = make_grid((-7.9, 53.1, -7.5, 53.4), 1.0, 10, 32629, 32)
        grid = type(grid)(lab.shape, grid.affine, 32629)
        interim = tmp_path / "interim"
        interim.mkdir(exist_ok=True)
        np.save(interim / "composite.npy", comp)
        np.save(interim / "labels_raw.npy", lab)
        (interim / "meta.json").write_text(json.dumps({"area": "synthetic", "grid": grid.to_dict(), "label_source": "corine",
                                                       "label_scheme": "test", "s2_n_scenes": 3, "composite_cached": False}))
    return fake


def test_run_pipeline_end_to_end_clears_stale_outputs_and_writes_run_info(cfg, synth, tmp_path):
    import copy
    c = copy.deepcopy(cfg)
    c["mode"] = "demo"
    for k in ("interim", "processed", "models", "reports"):
        c["paths"][k] = str(tmp_path / k)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "metrics_UNet.json").write_text("{}")             # stale file from an earlier run
    (tmp_path / "processed").mkdir()
    np.save(tmp_path / "processed" / "pred_UNet.npy", np.zeros((2, 2), np.uint8))
    events = []
    timings = runner.run_pipeline(c, models=["RandomForest"], download_fn=_fake_download(c, synth, tmp_path),
                                  on_stage_start=lambda n, i, t: events.append(("start", n, i, t)),
                                  on_stage_end=lambda n, s: events.append(("end", n)))
    assert list(timings) == ["download", "preprocess", "split", "train-rf", "evaluate"]
    assert events[0] == ("start", "download", 0, 5) and events[-1] == ("end", "evaluate")
    assert not (tmp_path / "reports" / "metrics_UNet.json").exists()           # stale results can never be mixed in
    assert not (tmp_path / "processed" / "pred_UNet.npy").exists()
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert list(metrics["models"]) == ["RandomForest"] and metrics["mode"] == "demo" and metrics["unet_converged"] is None
    assert len(metrics["models"]["RandomForest"]["confusion_normalised"]) == 7
    info = json.loads((tmp_path / "reports" / "run_info.json").read_text())
    assert info["mode"] == "demo" and info["n_scenes"] == 3 and set(info["timings_s"]) == set(timings)
