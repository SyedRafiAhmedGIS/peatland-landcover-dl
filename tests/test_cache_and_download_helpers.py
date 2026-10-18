import numpy as np

from peatland import cache
from peatland.download import baseline_offset


def test_baseline_offset_removed_only_when_all_scenes_have_it():
    assert baseline_offset(["05.00", "04.00"]) == (1000.0, False)
    assert baseline_offset(["02.08", "02.09"]) == (0.0, False)
    off, mixed = baseline_offset(["02.08", "05.00"])
    assert off == 0.0 and mixed                                   # mixed -> no blind correction, flagged
    assert baseline_offset(["?"]) == (0.0, False)


def test_cache_roundtrip_is_identical_for_fresh_and_cached_runs(tmp_path):
    comp = np.random.default_rng(0).uniform(0, 6000, (6, 40, 40)).astype("float32")
    comp[:, 3, 3] = np.nan
    path = tmp_path / "c.npz"
    fresh = cache.save_composite(path, comp, {"n_scenes": 5})
    loaded, meta = cache.load_composite(path)
    assert np.array_equal(np.isnan(fresh), np.isnan(loaded)) and np.nanmax(np.abs(fresh - loaded)) == 0
    assert meta["n_scenes"] == 5 and np.nanmax(np.abs(fresh - comp)) <= 0.5


def test_cache_key_changes_with_settings(cfg):
    img = cfg["imagery"]
    a = cache.cache_key([-7.9, 53.1, -7.5, 53.4], (512, 512), [10, 0, 5e5, 0, -10, 6e6], img)
    assert a == cache.cache_key([-7.9, 53.1, -7.5, 53.4], (512, 512), [10, 0, 5e5, 0, -10, 6e6], img)
    assert a != cache.cache_key([-7.9, 53.1, -7.5, 53.4], (512, 512), [10, 0, 5e5, 0, -10, 6e6], {**img, "max_cloud": 10})
    assert a != cache.cache_key([-7.9, 53.1, -7.5, 53.4], (256, 512), [10, 0, 5e5, 0, -10, 6e6], img)
