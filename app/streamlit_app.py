"""Streamlit front-end for the peatland land-cover pipeline.

Run:  streamlit run app/streamlit_app.py
Set PEATLAND_CONFIG to use a different config file (default: config.yaml).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from peatland import __version__, runner, summary  # noqa: E402
from peatland.config import ConfigError, apply_mode, load_config  # noqa: E402
from peatland.grid import make_grid  # noqa: E402
from peatland.labels import CLASS_COLORS, CLASS_NAMES, NODATA, PEAT, find_label_file  # noqa: E402
from peatland.quality import QualityError  # noqa: E402
from peatland.viz import CMAP  # noqa: E402

st.set_page_config(page_title="Peatland land-cover pipeline", page_icon="🌿", layout="wide")
CONFIG_PATH = os.environ.get("PEATLAND_CONFIG", "config.yaml")
MODEL_NAMES = {"RandomForest": "Random Forest", "UNet": "U-Net"}

st.markdown(
    """<style>
    .hero{padding:18px 24px;border-radius:14px;color:#fff;margin-bottom:10px;
          background:linear-gradient(120deg,#0f2027 0%,#203a43 45%,#2c5364 100%)}
    .hero h1{margin:0;font-size:30px;color:#fff} .hero p{margin:4px 0 0;opacity:.85}
    .chip{display:inline-block;padding:2px 10px;margin:6px 6px 0 0;border-radius:99px;background:rgba(255,255,255,.16);font-size:13px}
    </style>""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False)
def compute_info() -> dict:
    """Is PyTorch installed, and is a GPU available? (imported lazily so the results viewer works without torch)"""
    if importlib.util.find_spec("torch") is None:
        return {"torch": False, "gpu": None}
    try:
        import torch

        return {"torch": True, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception:
        return {"torch": False, "gpu": None}


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def resolve_label(path: str, mtime: float) -> str | None:
    found = find_label_file(path)
    return str(found) if found else None


def zip_reports(reports: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(reports.glob("*")):
            if f.is_file() and f.name != ".gitkeep":
                z.write(f, f.name)
    return buf.getvalue()


class LogSink(io.TextIOBase):
    """Captures print()/progress-bar output of the pipeline and shows the last lines live."""

    def __init__(self, box, keep: int = 18):
        self.box, self.keep, self.lines, self._last = box, keep, [""], 0.0

    def write(self, s: str) -> int:
        for ch in s:
            if ch == "\n":
                self.lines.append("")
            elif ch == "\r":
                self.lines[-1] = ""
            else:
                self.lines[-1] += ch
        if time.time() - self._last > 0.4:
            self.flush()
        return len(s)

    def flush(self) -> None:
        self._last = time.time()
        self.box.code("\n".join(self.lines[-self.keep:]) or " ", language="text")


try:
    BASE = load_config(CONFIG_PATH)
except (OSError, ConfigError) as exc:
    st.error(f"Could not load {CONFIG_PATH}: {exc}")
    st.stop()

CI = compute_info()
if "label_state" not in st.session_state:
    st.session_state["label_state"] = {}

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Run settings")
    mode = st.radio("Mode", ["demo", "full"], horizontal=True,
                    help="demo = fast preset for a live demonstration. full = the complete run whose numbers you report.")
    preset = apply_mode(BASE, mode)
    area_names = [a["name"] for a in BASE["study_areas"]]
    area_choice = st.selectbox("Study area", [*area_names, "Custom bounding box"])
    custom_bbox = None
    if area_choice == "Custom bounding box":
        b0 = BASE["study_areas"][0]["bbox"]
        c1, c2 = st.columns(2)
        lon0 = c1.number_input("min lon", value=float(b0[0]), format="%.3f")
        lat0 = c2.number_input("min lat", value=float(b0[1]), format="%.3f")
        lon1 = c1.number_input("max lon", value=float(b0[2]), format="%.3f")
        lat1 = c2.number_input("max lat", value=float(b0[3]), format="%.3f")
        custom_bbox = (lon0, lat0, lon1, lat1)
        st.caption("UTM zone 29N is configured (west / central Ireland). Other regions need a different EPSG in config.yaml.")
    fraction = st.slider("Share of the area to use", 0.3, 1.0, 1.0, 0.05,
                         help="Uses the central part of the box. Smaller = much faster (a good rehearsal setting).")
    models = st.multiselect("Models", list(MODEL_NAMES), default=list(MODEL_NAMES), format_func=MODEL_NAMES.get)
    if "UNet" in models and not CI["torch"]:
        st.warning("PyTorch is not installed here - the U-Net cannot run. Install requirements.txt or untick U-Net.")

    st.subheader("🏷️ Labels")
    label_choice = st.radio("Label source", ["CORINE 2018 (real peat class)", "WorldCover (PROXY, no peat class)"],
                            label_visibility="collapsed")
    corine_path: str | None = None
    if label_choice.startswith("CORINE"):
        up = st.file_uploader("Upload the CORINE raster (.tif or .zip)", type=["tif", "tiff", "zip"],
                              help="Free Copernicus login: land.copernicus.eu > CORINE Land Cover 2018 > raster 100 m. Clip to Ireland to keep it small.")
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        if up is not None:
            dest = raw_dir / Path(up.name).name
            sig = (up.name, up.size)
            if st.session_state["label_state"].get("sig") != sig:
                with st.spinner("Saving upload..."):
                    dest.write_bytes(up.getbuffer())
                st.session_state["label_state"]["sig"] = sig
            corine_path = resolve_label(str(dest), dest.stat().st_mtime)
        else:
            existing = Path(BASE["labels"]["corine_path"])
            if existing.exists():
                found = [f for f in ([existing] if existing.is_file() else sorted(existing.rglob("*")))
                         if f.suffix.lower() in (".tif", ".tiff", ".zip")]
                if found:
                    corine_path = resolve_label(str(found[0]), found[0].stat().st_mtime)
        if corine_path:
            st.success(f"CORINE ready: {Path(corine_path).name}")
        else:
            st.info("No CORINE file yet. Upload one, or switch to the clearly-labelled proxy.")
    else:
        st.warning("WorldCover has no peat class. 'Herbaceous wetland' is only a proxy, and every report will say so.")

    with st.expander("Advanced"):
        im, un = preset["imagery"], preset["unet"]
        year = st.number_input("Imagery year", 2016, 2025, int(im["year"]), key=f"year_{mode}")
        months = st.slider("Months", 3, 11, tuple(int(m) for m in im["months"]), key=f"months_{mode}")
        max_cloud = st.slider("Max scene cloud cover (%)", 5, 80, int(im["max_cloud"]), key=f"cloud_{mode}")
        scenes = st.slider("Scenes per Sentinel-2 tile", 1, 10, int(im["max_scenes_per_tile"]), key=f"scenes_{mode}")
        epochs = st.slider("U-Net epochs (max)", 3, 40, int(un["epochs"]), key=f"ep_{mode}")
        min_epochs = st.slider("U-Net minimum epochs", 1, 15, int(un["min_epochs"]), key=f"minep_{mode}")
        seed = st.number_input("Random seed", 0, 99999, int(BASE["seed"]), key="seed")

    st.divider()
    st.caption(f"Compute: **{CI['gpu'] or 'CPU'}** · peatland v{__version__}")
    if st.button("🗑️ Clear composite cache", help="Forces the next run to download Sentinel-2 again."):
        n = 0
        for f in Path(BASE["paths"]["interim"]).glob("composite_*.npz"):
            f.unlink()
            n += 1
        st.toast(f"Removed {n} cached composite(s).")

ready = bool(models) and (label_choice.startswith("WorldCover") or corine_path is not None)
try:
    cfg = runner.build_run_config(
        BASE, mode=mode, area=None if custom_bbox else area_choice, custom_bbox=custom_bbox, fraction=fraction, seed=int(seed),
        imagery={"year": int(year), "months": [int(months[0]), int(months[1])], "max_cloud": int(max_cloud),
                 "max_scenes_per_tile": int(scenes)},
        unet={"epochs": int(epochs), "min_epochs": int(min_epochs)})
    cfg_error = None
except (ConfigError, ValueError) as exc:
    cfg, cfg_error = None, str(exc)
reports_dir = Path(BASE["paths"]["reports"])

# ----------------------------------------------------------------------------- header
st.markdown(
    """<div class="hero"><h1>🌿 Peatland land-cover pipeline</h1>
    <p>Sentinel-2 + CORINE · Random Forest vs U-Net · spatially honest evaluation · peat bogs as the focus class</p>
    <span class="chip">🛰️ composite</span><span class="chip">🏷️ labels</span><span class="chip">🧩 tiles</span>
    <span class="chip">🗺️ block split</span><span class="chip">🔎 quality gate</span><span class="chip">🌲 RF</span>
    <span class="chip">🧠 U-Net</span><span class="chip">📊 report</span></div>""",
    unsafe_allow_html=True,
)
tab_run, tab_res, tab_qc, tab_map, tab_about = st.tabs(["▶️ Run", "📊 Results", "🔎 Data quality", "🗺️ Explore maps", "ℹ️ About"])

# ----------------------------------------------------------------------------- run tab
with tab_run:
    if cfg_error:
        st.error(f"Invalid settings: {cfg_error}")
    else:
        a = cfg["study_areas"][0]
        g = make_grid(tuple(a["bbox"]), a.get("fraction", 1.0), cfg["imagery"]["resolution"], cfg["imagery"]["epsg"], cfg["tiling"]["block"])
        km_w, km_h = g.shape[1] * g.resolution[0] / 1000, g.shape[0] * g.resolution[1] / 1000
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mode", mode.upper())
        c2.metric("Study area", a["name"])
        c3.metric("Size", f"{km_w:.1f} × {km_h:.1f} km", f"{g.shape[0] * g.shape[1] / 1e6:.1f} M pixels", delta_color="off")
        c4.metric("Models", " + ".join(MODEL_NAMES[m] for m in models) or "none")
        st.caption("Stages: " + " → ".join(runner.STAGE_LABELS[s] for s in runner.plan(models or ["RandomForest"])) +
                   ". The Sentinel-2 composite is cached, so repeating a run with the same area and dates skips the download. "
                   "Real timings are shown after every run and saved in reports/run_info.json.")
        if mode == "demo":
            st.warning("Demo mode is a fast preset - do not quote its numbers as final results.")
        if not ready:
            st.info("Choose at least one model and either upload CORINE or select the WorldCover proxy to enable the run button.")
        run_clicked = st.button("🚀 Run pipeline", type="primary", disabled=not ready, use_container_width=True)

        if run_clicked:
            status = st.status("Starting...", expanded=True)
            bar = st.progress(0.0, text="Starting")
            log_box = st.empty()
            curve = st.empty()
            losses: list[dict] = []
            sink = LogSink(log_box)

            def on_start(name: str, i: int, n: int) -> None:
                status.update(label=f"[{i + 1}/{n}] {runner.STAGE_LABELS[name]}")
                bar.progress(i / n, text=runner.STAGE_LABELS[name])

            def on_end(name: str, seconds: float) -> None:
                status.write(f"✅ {runner.STAGE_LABELS[name]} - {seconds:.0f} s")

            def on_epoch(epoch: int, tl: float, vl: float, secs: float) -> None:
                losses.append({"epoch": epoch, "train": tl, "validation": vl})
                curve.line_chart(pd.DataFrame(losses).set_index("epoch"))

            t_start = time.time()
            try:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    runner.run_pipeline(cfg, label_file=corine_path, allow_proxy=not corine_path, models=models,
                                        on_stage_start=on_start, on_stage_end=on_end, on_epoch=on_epoch)
                sink.flush()
                bar.progress(1.0, text="Finished")
                status.update(label=f"Pipeline finished in {(time.time() - t_start) / 60:.1f} min", state="complete", expanded=False)
                st.success("Done. Open the **Results** tab for metrics, maps and downloads.")
            except SystemExit as exc:
                sink.flush()
                status.update(label="Stopped", state="error")
                st.error(str(exc.code) if exc.code else "The pipeline stopped.")
            except QualityError as exc:
                sink.flush()
                status.update(label="Data-quality gate failed", state="error")
                st.error(f"{exc} - see the Data quality tab.")
            except Exception as exc:  # noqa: BLE001 - show anything unexpected instead of a blank page
                sink.flush()
                status.update(label="Failed", state="error")
                st.exception(exc)

# ----------------------------------------------------------------------------- results tab
with tab_res:
    metrics = read_json(reports_dir / "metrics.json")
    if not metrics:
        st.info("No results yet. Run the pipeline first (▶️ Run tab) - or put an earlier run's `reports/` folder in place.")
    else:
        res = metrics["models"]
        meta = {k: v for k, v in metrics.items() if k != "models"}
        for level, msg in summary.headline(res, meta):
            getattr(st, level)(msg)

        cols = st.columns(len(res))
        for col, (name, r) in zip(cols, res.items(), strict=True):
            with col:
                st.subheader(MODEL_NAMES.get(name, name))
                m1, m2 = st.columns(2)
                m1.metric("🟤 Peat F1", "n/a" if r["peat_f1"] is None else f"{r['peat_f1']:.3f}")
                m2.metric("Macro F1", f"{r['macro_f1']:.3f}")
                m3, m4 = st.columns(2)
                m3.metric("Accuracy", f"{r['accuracy']:.3f}")
                m4.metric("Kappa", f"{r['kappa']:.3f}")
                if r["peat_f1"] is not None:
                    st.caption(f"Peat precision {r['peat_precision']:.3f} · recall {r['peat_recall']:.3f}")

        st.markdown("#### Per-class F1 on unseen test blocks")
        f1 = pd.DataFrame({MODEL_NAMES.get(n, n): [r["per_class"][c]["f1"] for c in CLASS_NAMES] for n, r in res.items()}, index=CLASS_NAMES)
        st.bar_chart(f1)
        sup = pd.DataFrame({"test pixels": [next(iter(res.values()))["per_class"][c]["support"] for c in CLASS_NAMES]}, index=CLASS_NAMES)
        with st.expander("Per-class precision / recall / F1 / support"):
            for n, r in res.items():
                st.markdown(f"**{MODEL_NAMES.get(n, n)}**")
                st.dataframe(pd.DataFrame(r["per_class"]).T.assign(support=lambda d: d["support"].astype(int)), use_container_width=True)
            st.markdown("**Reference pixel counts in the test blocks**")
            st.dataframe(sup, use_container_width=True)

        st.markdown("#### Confusion matrices (row-normalised: share of each reference class)")
        cm_cols = st.columns(len(res))
        for col, (name, r) in zip(cm_cols, res.items(), strict=True):
            cm = r.get("confusion_normalised")
            if cm:
                fig, ax = plt.subplots(figsize=(5.2, 4.6))
                ax.imshow(np.array(cm), cmap="Blues", vmin=0, vmax=1)
                for i in range(7):
                    for j in range(7):
                        ax.text(j, i, f"{cm[i][j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cm[i][j] > 0.5 else "black")
                ax.set_xticks(range(7), CLASS_NAMES, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(7), CLASS_NAMES, fontsize=7)
                ax.set(xlabel="predicted", ylabel="reference", title=MODEL_NAMES.get(name, name))
                fig.tight_layout()
                col.pyplot(fig)
                plt.close(fig)

        st.markdown("#### Figures")
        for fname, cap in [("05_test_maps.png", "Reference vs predictions on the test blocks, and where peat is correct / missed / a false alarm"),
                           ("03_unet_training.png", "U-Net training curve (log scale)"),
                           ("01_study_area.png", "Study area: composite and labels")]:
            if (reports_dir / fname).exists():
                st.image(str(reports_dir / fname), caption=cap, use_container_width=True)

        run_info = read_json(reports_dir / "run_info.json")
        if run_info:
            with st.expander("Run details and timings"):
                t = run_info.get("timings_s", {})
                if t:
                    st.bar_chart(pd.Series(t, name="seconds"))
                    st.caption(f"Total {sum(t.values()) / 60:.1f} min · composite cached: {run_info.get('composite_cached')} · "
                               f"device: {run_info.get('device')} · U-Net epochs: {run_info.get('unet_epochs_run')}")
                st.json({k: v for k, v in run_info.items() if k != "config"})

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download all reports (zip)", zip_reports(reports_dir), "peatland_reports.zip", "application/zip",
                           use_container_width=True)
        d2.download_button("⬇️ metrics.json", (reports_dir / "metrics.json").read_bytes(), "metrics.json", "application/json",
                           use_container_width=True)

# ----------------------------------------------------------------------------- data quality tab
with tab_qc:
    qc = read_json(reports_dir / "data_quality.json")
    if not qc:
        st.info("The quality report appears here after the split stage.")
    else:
        {"pass": st.success, "warn": st.warning, "fail": st.error}[qc["status"]](
            f"Quality gate: {qc['status'].upper()} · area {qc.get('area')} · labels {qc.get('label_source')} · split seed {qc.get('split_seed')}")
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
        st.dataframe(pd.DataFrame([{"": icon[c["status"]], "check": c["name"], "result": c["detail"]} for c in qc["checks"]]),
                     use_container_width=True, hide_index=True)
        st.markdown("#### Labelled pixels per split")
        st.dataframe(pd.DataFrame(qc["class_counts_per_split"]), use_container_width=True)
        if (reports_dir / "02_spatial_split.png").exists():
            st.image(str(reports_dir / "02_spatial_split.png"), caption="Spatial block split (grey = exclusion buffer)", width=600)

# ----------------------------------------------------------------------------- explore tab
with tab_map:
    processed = Path(BASE["paths"]["processed"])
    lab_f, split_f = processed / "labels.npy", processed / "tile_split.npy"
    if not (lab_f.exists() and split_f.exists()):
        st.info("Maps appear here after a run (they are drawn from data/processed).")
    else:
        lab = np.load(lab_f, mmap_mode="r")
        tiles = np.load(split_f)
        tile = cfg["tiling"]["tile"] if cfg else BASE["tiling"]["tile"]
        test_mask = np.kron(tiles, np.ones((tile, tile), np.int8)) == 2
        test_mask &= np.asarray(lab) != NODATA
        preds = {n: np.load(processed / f"pred_{n}.npy", mmap_mode="r") for n in MODEL_NAMES if (processed / f"pred_{n}.npy").exists()}
        layers = ["Labels (whole area)", "Reference (test blocks only)", *[f"{MODEL_NAMES[n]} prediction" for n in preds],
                  *[f"Peat hit / miss - {MODEL_NAMES[n]}" for n in preds]]
        c1, c2 = st.columns([3, 1])
        layer = c1.selectbox("Layer", layers)
        step = c2.slider("Detail (1 = full resolution)", 1, 8, 4)
        L = np.asarray(lab[::step, ::step])
        T = test_mask[::step, ::step]
        fig, ax = plt.subplots(figsize=(9, 8))
        if layer.startswith("Labels"):
            ax.imshow(np.where(L == NODATA, 7, L), cmap=CMAP, vmin=0, vmax=7, interpolation="nearest")
        elif layer.startswith("Reference"):
            ax.imshow(np.where(T, np.where(L == NODATA, 7, L), 7), cmap=CMAP, vmin=0, vmax=7, interpolation="nearest")
        elif layer.startswith("Peat hit"):
            key = next(n for n in preds if MODEL_NAMES[n] in layer)
            P = np.asarray(preds[key][::step, ::step])
            agree = np.zeros(L.shape, np.uint8)
            agree[T] = 1
            agree[T & (L == PEAT) & (P == PEAT)] = 2
            agree[T & (L == PEAT) & (P != PEAT)] = 3
            agree[T & (L != PEAT) & (P == PEAT)] = 4
            ax.imshow(agree, cmap=ListedColormap(["#ffffff", "#e5e7eb", "#16a34a", "#dc2626", "#f59e0b"]), vmin=0, vmax=4, interpolation="nearest")
            ax.legend(handles=[mpatches.Patch(color=c, label=n) for c, n in zip(["#16a34a", "#dc2626", "#f59e0b"],
                      ["correct peat", "missed peat", "false alarm"], strict=True)], loc="lower left")
        else:
            key = next(n for n in preds if MODEL_NAMES[n] in layer)
            P = np.asarray(preds[key][::step, ::step])
            ax.imshow(np.where(T, np.where(P == NODATA, 7, P), 7), cmap=CMAP, vmin=0, vmax=7, interpolation="nearest")
        if not layer.startswith("Peat hit"):
            ax.legend(handles=[mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(7)], loc="lower left", fontsize=8)
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Predictions exist only for the held-out test blocks - the model never saw them during training.")

# ----------------------------------------------------------------------------- about tab
with tab_about:
    st.markdown(f"""
### What this app does
It runs the same code as the command line (`python -m peatland.cli`) and the Colab notebook, with a form instead of a config file.

1. **Composite** - clearest Sentinel-2 L2A scenes per tile from Microsoft Planetary Computer, per-scene SCL cloud masking
   (clear classes 4-6), the +1000 DN processing-baseline offset removed when all scenes carry it, then a median over time.
2. **Labels** - CORINE 2018 warped onto the same 10 m grid and mapped to 7 classes (peat bogs = code 412). Without CORINE, a
   WorldCover **proxy** with no real peat class - always flagged.
3. **Tiles and split** - 128 px tiles inside 2.56 km blocks. Whole blocks go to train / validation / test with an exclusion
   buffer around the test blocks. Feature scaling uses training pixels only.
4. **Quality gate** - the run stops if a critical check fails.
5. **Models** - Random Forest (square-root-balanced sampling) vs a small U-Net (weighted loss, early stopping not before the
   minimum number of epochs). Both are scored on identical test pixels.

### Limitations
{summary.LIMITATIONS}

### Data
Sentinel-2 L2A (Copernicus, via Planetary Computer) · CORINE Land Cover 2018 (Copernicus Land Monitoring Service, manual download) ·
ESA WorldCover (CC BY 4.0, proxy only). See `docs/data_sources.md`.
""")
