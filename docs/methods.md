# Methods

## Composite
Scenes (May-September of the configured year, scene cloud cover below the limit) are grouped by MGRS tile and the
least-cloudy ones kept. Each scene is masked with the Scene Classification Layer (classes 4-6 kept; unclassified haze is dropped)
**before** a per-pixel median over time. Processing baseline >= 04.00 adds a +1000 DN offset to L2A reflectance; it is removed
only if every selected scene carries it, and mixed baselines are flagged. The composite is cached (uint16) under `data/interim`. Bands are resampled to the 10 m grid
(bilinear; SCL nearest).

## Features
Reflectance = DN / 10000. NDVI = (NIR - Red) / (NIR + Red); NDMI = (NIR - SWIR1) / (NIR + SWIR1). Eight channels in total.
Features are standardised with statistics computed on **training pixels only**.

## Labels
CORINE is warped onto the study grid with nearest neighbour resampling and remapped to seven classes (see `labels.py`).
Pixels without valid imagery become nodata (255) and are ignored in training and evaluation.

## Spatial split
Tiles (128 px) nest exactly inside spatial blocks (256 px = 2.56 km). Whole blocks are shuffled with a fixed seed and assigned to
train / validation / test (70/15/15). Tiles within one tile (1.28 km) of a test tile are removed from train and validation (`buffer_tiles`). Train tiles next to
validation tiles are removed only if `val_buffer_tiles` > 0 (default 0): validation only drives early stopping, so this keeps ~45% instead
of ~27% of the area for training while the reported test blocks stay fully isolated. The quality gate checks all three spacings. Seeds are tried in order until train, validation and test all contain peat;
this decision uses labels only and never model performance. The quality gate independently re-checks that no block is shared
between splits and that the buffer distance holds.

## Models
- **Random Forest**: pixels sampled per class in proportion to sqrt(class size) (the largest class gets `rf.max_px`), no class
  weights, depth 18 - the same effective class prior as the U-Net, so the comparison is fair.
- **U-Net**: `segmentation_models_pytorch`, ResNet-18 encoder, no pretrained weights, slim decoder, 8 input channels,
  weighted cross-entropy (square-root inverse frequency) ignoring nodata, AdamW, `tiles_per_epoch` tiles per epoch (whole shuffled passes over the training tiles, repeated when fewer tiles exist), flips and 90-degree rotations, early stopping
  on validation loss (not before `min_epochs`), a safety time cap (a capped run is flagged), best checkpoint restored.
  Kappa < 0.10 on the test blocks marks the U-Net as not converged and its scores are not interpreted.

## Evaluation
Both models are scored on the same test pixels. Macro F1 averages only classes present in the test reference. Peat precision,
recall and F1 are reported separately.
