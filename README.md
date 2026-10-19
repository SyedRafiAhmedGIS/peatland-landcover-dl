# peatland-landcover-dl

[![CI](https://github.com/SyedRafiAhmedGIS/peatland-landcover-dl/actions/workflows/ci.yml/badge.svg)](https://github.com/SyedRafiAhmedGIS/peatland-landcover-dl/actions)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![Open the app in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SyedRafiAhmedGIS/peatland-landcover-dl/blob/main/notebooks/run_app_in_colab.ipynb) [![Open the pipeline notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SyedRafiAhmedGIS/peatland-landcover-dl/blob/main/notebooks/peatland_landcover_pipeline.ipynb)

Reproducible, open-data land-cover classification for a peat-rich area of Ireland, with the **peat-bog class as the focus**.
Sentinel-2 imagery and CORINE labels feed a Random Forest baseline and a small U-Net, both evaluated on **spatially separated
test blocks** and packaged with tests, a data-quality gate, CI and Docker.

> **Status:** v0.1.0 (work in progress). Results are only shown below once they have been produced by a real run
> (`reports/`). Nothing in this README is typed in by hand.

## Study area and data

| Item | Source | Licence / access |
|---|---|---|
| Imagery | Sentinel-2 L2A via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) (STAC) | Copernicus open data, no login |
| Labels | [CORINE Land Cover](https://land.copernicus.eu/en/products/corine-land-cover) 2018, 100 m raster | Copernicus open data, free login, **manual download** |
| Fallback labels | ESA WorldCover 10 m (Planetary Computer) | CC BY 4.0. **No peat class** - herbaceous wetland is a proxy only |

Default area: Irish Midlands bogs, lon/lat `[-7.90, 53.10, -7.50, 53.40]` (automatic fallback: Connemara blanket bog). The code
checks that the area really contains CORINE peat-bog (412) pixels before downloading imagery. Imagery: May-September 2018,
scenes under 30 % cloud, per-scene SCL cloud masking, median composite, bands B02 B03 B04 B08 B11 B12 at 10 m plus NDVI and NDMI.

## Method

1. **Composite** - clearest scenes per Sentinel-2 tile, SCL mask per scene (clear classes 4-6), +1000 DN processing-baseline offset
   removed when all scenes carry it, median over time. Cached so repeat runs skip the download.
2. **Labels** - CORINE warped onto the same 10 m grid and remapped to 7 classes (artificial, agriculture, forest, open semi-natural,
   **peat bogs**, other wetlands, water). Both CORINE raster encodings (index 1-44 or 3-digit codes) are detected automatically.
3. **Tiles and spatial split** - 128 x 128 tiles nested in 2.56 km blocks. Whole blocks go to train / validation / test
   (70/15/15) with a one-tile (1.28 km) exclusion ring between the **test** blocks and everything else (`tiling.buffer_tiles`). Validation blocks
   (used only for early stopping) are not ringed from training by default (`tiling.val_buffer_tiles: 0`), which keeps ~45% of the area for
   training instead of ~27%. Feature scaling uses training pixels only.
4. **Quality gate** - CRS, resolution, shapes, NaNs, block purity, buffer distance, class presence. A failure stops the run.
5. **Models** - Random Forest (square-root-balanced sampling) vs U-Net (ResNet-18 encoder, from scratch, `tiles_per_epoch` tiles drawn per epoch, square-root-weighted
   cross-entropy ignoring nodata, flips and 90-degree rotations, early stopping not before a minimum number of epochs). Both use
   the same effective class prior. A U-Net with kappa < 0.10 is flagged as not converged.
6. **Evaluation** - both models scored on identical test pixels: accuracy, macro F1, kappa, per-class P/R/F1, confusion matrices,
   test maps. Peat bogs are reported separately.

Why spatial blocks? Neighbouring pixels are strongly correlated, so a random pixel split lets the model "see" its test
areas during training and overstates accuracy. See [docs/methods.md](docs/methods.md).

## Streamlit app

A web front-end for the same pipeline (no config editing needed):

```bash
pip install -r requirements.txt && pip install --no-deps -e .
streamlit run app/streamlit_app.py
```

- **Sidebar:** mode (demo / full), study area or custom bounding box, share of the area, models, CORINE **upload** (or the
  clearly-labelled WorldCover proxy), advanced imagery and U-Net settings.
- **Run tab:** one button; live stage progress, log and U-Net loss curve. Errors and failed quality checks are shown, never hidden.
- **Results tab:** metrics per model, per-class F1, confusion matrices, figures, computed warnings (proxy labels, demo mode,
  U-Net not converged, few peat pixels) and a zip download of `reports/`.
- **Data quality / Explore maps tabs:** the quality-gate table and class counts per split, and test-block maps with a peat
  hit / miss / false-alarm layer.

The app can also just *view* results: put an earlier run's `reports/` (and `data/processed/` for maps) in place and open it - PyTorch
is only needed to train the U-Net. To use it from Colab, open `notebooks/run_app_in_colab.ipynb` with the badge above (it clones this repository; a zip upload is the fallback). In Docker:

```bash
docker run -p 8501:8501 -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" --entrypoint streamlit \
  peatland-landcover-dl run app/streamlit_app.py --server.address 0.0.0.0
```

## Reproduce

```bash
git clone https://github.com/SyedRafiAhmedGIS/peatland-landcover-dl.git && cd peatland-landcover-dl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install --no-deps -e .

# manual step: save the CORINE 2018 raster (.zip or .tif) from land.copernicus.eu into data/raw/
python -m peatland.cli run-all --mode full   # download -> preprocess -> split -> train-rf -> train-unet -> evaluate
```

Run stages one at a time with `download`, `preprocess`, `split`, `train-rf`, `train-unet`, `evaluate`.
Use `--mode demo` for a fast preset (do not quote demo numbers), `--skip-unet` for a baseline-only run, and `fraction` in `config.yaml` for a smaller area.
Without CORINE the pipeline stops with instructions; `--allow-proxy` switches to WorldCover **proxy** labels (clearly flagged in every report).

**Docker**

```bash
docker build -t peatland-landcover-dl .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" -v "$PWD/models:/app/models" \
  peatland-landcover-dl run-all
```

A GPU is optional (the U-Net is time-boxed and works on CPU). A Colab notebook version is in [`notebooks/`](notebooks/).

## Results

Generated by `python -m peatland.cli evaluate` into `reports/` (table in `reports/results_table.md`).

| Model | Overall accuracy | Macro F1 | Peat-bog F1 | Peat precision | Peat recall |
|---|---|---|---|---|---|
| RandomForest | 0.852 | 0.684 | 0.404 | 0.289 | 0.671 |
| UNet | 0.844 | 0.677 | 0.400 | 0.265 | 0.815 |

Figures after a run: `reports/04_results.png` (confusion matrices, per-class F1), `reports/05_test_maps.png` (test maps and
peat hit/miss map). Data-quality report: `reports/data_quality.json`.

## Limitations (please read)

- CORINE is 100 m with a 25 ha minimum mapping unit - coarse labels against 10 m imagery. Peat bogs are hard because they are
  spectrally close to wet grassland and heath, and small peat patches are generalised away in the labels.
- Label year (2018) and imagery year can differ slightly; one area, one season; one random block split (no repeats yet).
- No field validation. The small U-Net is trained from scratch on CPU under a time budget.
- Results are land-cover agreement with CORINE, **not a peatland condition assessment**.
- WorldCover fallback has no peat class; results using it are not comparable.

More detail: [docs/limitations.md](docs/limitations.md).

## Project structure

```
src/peatland/   config, grid, labels, download, cache, preprocess, tiling, split, quality, baseline, unet, evaluate, viz,
                pipeline, runner, summary, cli
app/            Streamlit app (streamlit_app.py)
tests/          pytest tests on tiny synthetic arrays + headless UI tests (never download data)
reports/        metrics, data-quality report and figures from real runs (small files, committed)
docs/           data_sources.md, methods.md, limitations.md
notebooks/      single-file Colab version of the pipeline (prototype)
```

Tests: `pytest` (includes headless Streamlit UI tests) · Lint: `ruff check .` · CI runs both and builds the Docker image on every push.

## Author

Syed Rafi Ahmed - MIT licence. Data are open (Copernicus, ESA, Microsoft Planetary Computer); no dissertation data is used.

## Results (v0.1.0)

Preset: `--mode full --allow-proxy` (WorldCover proxy labels, Irish Midlands bogs study area, 12 clearest Sentinel-2 scenes).

| Model | Overall accuracy | Macro F1 | Peat-bog F1 | Peat precision | Peat recall |
|---|---|---|---|---|---|
| RandomForest | 0.852 | 0.684 | 0.404 | 0.289 | 0.671 |
| UNet | 0.844 | 0.677 | 0.400 | 0.265 | 0.815 |

Stages: download (125s) → preprocess (10s) → split (1s) → train-rf (32s) → train-unet (859s, early-stopped epoch 12) → evaluate (3s).
