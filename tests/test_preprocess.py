import numpy as np

from peatland.preprocess import build_features, clean_labels, normalise, scl_valid


def test_cloud_mask_keeps_only_clear_classes():
    scl = np.array([[3, 4, 5], [6, 7, 9]])           # 3 shadow, 7 unclassified (haze), 9 cloud
    assert scl_valid(scl).tolist() == [[False, True, True], [True, False, False]]


def test_features_have_8_channels_and_indices_in_range(synth):
    comp, _ = synth
    feats, names = build_features(comp, ["B02", "B03", "B04", "B08", "B11", "B12"])
    assert feats.shape == (8, *comp.shape[1:]) and names[-2:] == ["NDVI", "NDMI"]
    assert feats[6].min() >= -1 and feats[6].max() <= 1


def test_nan_features_become_nodata_labels(synth):
    comp, lab = synth
    comp = comp.copy()
    comp[:, 50, 50] = np.nan
    feats, _ = build_features(comp, ["B02", "B03", "B04", "B08", "B11", "B12"])
    assert clean_labels(lab, feats)[50, 50] == 255


def test_normalise_uses_training_pixels_only(synth):
    comp, lab = synth
    feats, _ = build_features(comp, ["B02", "B03", "B04", "B08", "B11", "B12"])
    train = np.zeros(lab.shape, bool)
    train[:100] = True
    out = normalise(feats, lab, train)
    sel = train & (lab != 255)
    assert abs(out[:, sel].mean()) < 1e-3 and np.isfinite(out).all()
