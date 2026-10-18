import numpy as np
import pytest

from peatland.grid import Grid, make_grid
from peatland.labels import PEAT
from peatland.quality import QualityError, run_checks, write_report
from peatland.split import BUFFER, TEST, TRAIN, VAL, assign_split, search_split
from peatland.tiling import peat_per_tile, pixel_split_map, tile_stats


def _prep(cfg, lab):
    t = cfg["tiling"]
    valid_frac, tile_ok, _ = tile_stats(lab, t["tile"], t["block"], t["min_valid_tile"])
    return valid_frac, tile_ok, t["block"] // t["tile"]


def test_tile_shapes_and_nodata_strip(cfg, synth):
    _, lab = synth
    valid_frac, tile_ok, per = _prep(cfg, lab)
    assert valid_frac.shape == (16, 16) and per == 2
    assert not tile_ok[-1].any()                      # bottom rows are nodata


def test_tiling_rejects_ragged_rasters(cfg):
    with pytest.raises(ValueError):
        tile_stats(np.zeros((100, 100), np.uint8), 16, 32, 0.3)


def test_no_block_appears_in_two_splits(cfg, synth):
    _, lab = synth
    _, tile_ok, per = _prep(cfg, lab)
    for seed in range(5):
        ts = assign_split(tile_ok, per, cfg["tiling"]["split"], seed, 1)
        v = ts.reshape(8, per, 8, per).transpose(0, 2, 1, 3).reshape(8, 8, -1)
        for block in v.reshape(-1, per * per):
            kept = set(block[(block >= TRAIN) & (block <= TEST)].tolist())
            assert len(kept) <= 1


def test_buffer_separates_test_from_train(cfg, synth):
    from scipy import ndimage as ndi
    _, lab = synth
    _, tile_ok, per = _prep(cfg, lab)
    ts = assign_split(tile_ok, per, cfg["tiling"]["split"], 42, 1)
    assert (ts == BUFFER).any() and (ts == TEST).any() and (ts == VAL).any()
    gap = ndi.distance_transform_cdt(ts != TEST, metric="chessboard")[ts == TRAIN].min()
    assert gap >= 2


def test_split_is_deterministic(cfg, synth):
    _, lab = synth
    _, tile_ok, per = _prep(cfg, lab)
    a = assign_split(tile_ok, per, cfg["tiling"]["split"], 7, 1)
    b = assign_split(tile_ok, per, cfg["tiling"]["split"], 7, 1)
    assert (a == b).all()


def test_search_split_puts_peat_in_every_part(cfg, synth):
    _, lab = synth
    _, tile_ok, per = _prep(cfg, lab)
    peat = peat_per_tile(lab, cfg["tiling"]["tile"], PEAT)
    ts, _, shares, balanced = search_split(tile_ok, peat, per, cfg["tiling"]["split"], 42, 1)
    assert balanced and all(s > 0 for s in shares)


def _grid(shape):
    return Grid(shape, make_grid((-7.9, 53.1, -7.5, 53.4), 1.0, 10, 32629, 32).affine, 32629)


def test_quality_gate_passes_on_good_data(cfg, synth):
    comp, lab = synth
    feats = np.nan_to_num(np.concatenate([comp, comp[:2]]) / 1e4).astype("float32")
    valid_frac, tile_ok, per = _prep(cfg, lab)
    ts = assign_split(tile_ok, per, cfg["tiling"]["split"], 42, 1)
    report = run_checks(_grid(lab.shape), feats, lab, ts, valid_frac, cfg)
    assert report["status"] in ("pass", "warn")


def test_quality_gate_catches_bad_data(cfg, synth, tmp_path):
    comp, lab = synth
    feats = np.nan_to_num(np.concatenate([comp, comp[:2]]) / 1e4).astype("float32")
    feats[:, 100, 100] = np.nan                       # a labelled pixel with a NaN feature
    lab = lab.copy()
    lab[100, 100] = 3
    valid_frac, tile_ok, per = _prep(cfg, lab)
    ts = assign_split(tile_ok, per, cfg["tiling"]["split"], 42, 1)
    report = run_checks(_grid(lab.shape), feats, lab, ts, valid_frac, cfg)
    assert report["status"] == "fail"
    with pytest.raises(QualityError):
        write_report(report, tmp_path / "dq.json")


def test_quality_gate_catches_shape_mismatch(cfg, synth):
    comp, lab = synth
    feats = np.zeros((8, 100, 100), np.float32)
    valid_frac, tile_ok, per = _prep(cfg, lab)
    ts = assign_split(tile_ok, per, cfg["tiling"]["split"], 42, 1)
    assert run_checks(_grid(lab.shape), feats, lab, ts, valid_frac, cfg)["status"] == "fail"


def test_pixel_split_map_expands_tiles():
    assert pixel_split_map(np.array([[0, 2]], np.int8), 4).shape == (4, 8)
