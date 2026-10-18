# Changelog

## [0.1.0] - unreleased
### Fixed
- **U-Net sampling**: `tiles_per_epoch` was silently capped at the number of training tiles (640 requested, 168 available -> ~11 gradient
  steps per epoch). Epochs now draw exactly `tiles_per_epoch` tiles (`tiling.epoch_indices`). `full` mode trains up to 30 epochs.
- **Split buffer**: new `tiling.val_buffer_tiles` (default 0). The old validation ring discarded ~60% of the nominal training tiles;
  test blocks keep the full exclusion ring.
- **Quality gate** now also checks validation-test and train-validation spacing.
- **Headline verdict** compares macro F1, peat F1 and kappa (reports mixed results) instead of macro F1 alone, with a noise note.
- **STAC date range** ends on the real last day of the end month (was always the 30th).
- Config validation: `imagery.months` and buffer sizes.
### Added
- **Streamlit app** (`app/streamlit_app.py`): run settings form, CORINE upload, live log and U-Net loss curve, results with computed
  warnings, quality-gate view, interactive test-block maps, report download. Shares all logic with the CLI via `peatland.runner`.
- `demo` / `full` mode presets (`modes:` in config.yaml; `--mode` on the CLI).
- Composite cache (uint16, keyed by area + dates + settings) so repeat runs skip the download.
- Sentinel-2 processing-baseline check: the +1000 DN offset is removed only when all scenes carry it; mixed baselines are flagged.
- U-Net minimum epochs, time-cap flag, convergence flag (kappa < 0.10) and `reports/run_info.json`.
- `peatland.summary.headline`: one computed source of warnings/limitations for the CLI and the app.

### Changed
- Cloud mask keeps SCL classes 4-6 only (unclassified haze dropped).
- Random Forest samples pixels per class in proportion to sqrt(class size) and uses no class weights, matching the U-Net's
  square-root class prior.
- A split with no peat is no longer treated as "peat-balanced".
- Earlier changes:
- Config-driven pipeline: download, preprocess, split, train-rf, train-unet, evaluate (`python -m peatland.cli`).
- Sentinel-2 L2A median composite with per-scene SCL cloud masking (Microsoft Planetary Computer).
- CORINE label remapping to 7 classes (handles index 1-44 and 3-digit code rasters); WorldCover PROXY fallback.
- Spatial block split (2.56 km blocks) with a one-tile exclusion buffer and a peat-aware seed search.
- Data-quality gate writing `reports/data_quality.json`.
- Random Forest baseline and small U-Net (ResNet-18 encoder), evaluated on identical test pixels.
- 36 pytest tests on synthetic arrays, ruff linting, Dockerfile and GitHub Actions CI.
