"""Headless UI tests with Streamlit's AppTest. No network, no torch needed."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from peatland import runner  # noqa: E402
from peatland.config import load_config  # noqa: E402
from tests.test_summary_and_runner import _fake_download  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _config_file(tmp_path, cfg):
    c = copy.deepcopy(load_config("config.yaml"))
    for k in ("interim", "processed", "models", "reports"):
        c["paths"][k] = str(tmp_path / k)
    c["tiling"].update(tile=16, block=32)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c))
    return path, c


def test_app_loads_without_results(tmp_path, cfg, monkeypatch):
    path, _ = _config_file(tmp_path, cfg)
    monkeypatch.setenv("PEATLAND_CONFIG", str(path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("No results yet" in i.value for i in at.info)
    run_btn = [b for b in at.button if "Run pipeline" in b.label]
    assert run_btn and run_btn[0].disabled                      # no CORINE and no proxy selected -> cannot run


def test_proxy_choice_enables_run_and_warns(tmp_path, cfg, monkeypatch):
    path, _ = _config_file(tmp_path, cfg)
    monkeypatch.setenv("PEATLAND_CONFIG", str(path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    label_radio = next(r for r in at.radio if "WorldCover" in " ".join(r.options))
    label_radio.set_value(label_radio.options[1]).run()
    assert not at.exception
    assert any("PROXY" in w.value or "proxy" in w.value for w in at.warning)
    assert not [b for b in at.button if "Run pipeline" in b.label][0].disabled


def test_app_renders_results_from_a_real_run(tmp_path, cfg, synth, monkeypatch):
    path, c = _config_file(tmp_path, cfg)
    c["rf"].update(trees=6, max_px=2000)
    c["mode"] = "demo"
    runner.run_pipeline(c, models=["RandomForest"], download_fn=_fake_download(c, synth, tmp_path))
    monkeypatch.setenv("PEATLAND_CONFIG", str(path))
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "🟤 Peat F1" in labels and "Macro F1" in labels
    assert any("DEMO" in w.value for w in at.warning)            # demo-mode warning comes from the saved metrics
    assert any("Quality gate" in (s.value or "") for s in list(at.success) + list(at.warning))
    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    shown = [m.value for m in at.metric if m.label == "Macro F1"]
    assert shown == [f"{saved['models']['RandomForest']['macro_f1']:.3f}"]   # UI shows exactly what the file says
    assert np.isfinite(saved["models"]["RandomForest"]["kappa"])


def _proxy_app(tmp_path, cfg, monkeypatch):
    path, _ = _config_file(tmp_path, cfg)
    monkeypatch.setenv("PEATLAND_CONFIG", str(path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    radio = next(r for r in at.radio if "WorldCover" in " ".join(r.options))
    radio.set_value(radio.options[1]).run()
    return at


def _click_run(at):
    next(b for b in at.button if "Run pipeline" in b.label).click().run()


def test_run_button_success_path_shows_live_log_and_done_message(tmp_path, cfg, monkeypatch):
    seen = {}

    def fake_run(c, *, label_file, allow_proxy, models, on_stage_start, on_stage_end, on_epoch, **kw):
        seen.update(label_file=label_file, allow_proxy=allow_proxy, models=list(models), mode=c["mode"])
        stages = runner.plan(models)
        for i, s in enumerate(stages):
            on_stage_start(s, i, len(stages))
            print(f"working on {s}")
            if s == "train-unet":
                on_epoch(1, 0.9, 1.0, 1.0)
                on_epoch(2, 0.7, 0.8, 2.0)
            on_stage_end(s, 0.1)
        return {}

    monkeypatch.setattr(runner, "run_pipeline", fake_run)
    at = _proxy_app(tmp_path, cfg, monkeypatch)
    _click_run(at)
    assert not at.exception
    assert any("Done." in s.value for s in at.success)
    assert seen["allow_proxy"] is True and seen["label_file"] is None and seen["mode"] == "demo"
    assert seen["models"] == ["RandomForest", "UNet"]


def test_run_button_reports_stops_and_quality_failures(tmp_path, cfg, monkeypatch):
    from peatland.quality import QualityError

    def stop(*a, **k):
        raise SystemExit("CORINE Land Cover file not found.")

    monkeypatch.setattr(runner, "run_pipeline", stop)
    at = _proxy_app(tmp_path, cfg, monkeypatch)
    _click_run(at)
    assert not at.exception and any("CORINE Land Cover file not found" in e.value for e in at.error)

    def fail(*a, **k):
        raise QualityError("Data-quality gate FAILED: ['x']")

    monkeypatch.setattr(runner, "run_pipeline", fail)
    at = _proxy_app(tmp_path, cfg, monkeypatch)
    _click_run(at)
    assert not at.exception and any("Data-quality gate FAILED" in e.value for e in at.error)
