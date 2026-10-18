import pytest

from peatland.grid import Grid, grid_lonlat, make_grid


def test_grid_is_whole_blocks_on_10m_lattice():
    g = make_grid((-7.9, 53.1, -7.5, 53.4), 1.0, 10, 32629, 256)
    assert g.shape[0] % 256 == 0 and g.shape[1] % 256 == 0
    assert g.resolution == (10.0, 10.0) and g.affine.c % 10 == 0 and g.affine.f % 10 == 0


def test_fraction_shrinks_area():
    full = make_grid((-7.9, 53.1, -7.5, 53.4), 1.0, 10, 32629, 256)
    half = make_grid((-7.9, 53.1, -7.5, 53.4), 0.5, 10, 32629, 256)
    assert half.shape[0] < full.shape[0] and half.shape[1] < full.shape[1]


def test_lonlat_bounds_inside_requested_box():
    g = make_grid((-7.9, 53.1, -7.5, 53.4), 1.0, 10, 32629, 256)
    lo, la, lo2, la2 = grid_lonlat(g)
    assert -8.0 < lo < lo2 < -7.4 and 53.0 < la < la2 < 53.5


def test_grid_roundtrip_and_too_small_area():
    g = make_grid((-7.9, 53.1, -7.5, 53.4), 1.0, 10, 32629, 256)
    assert Grid.from_dict(g.to_dict()) == g
    with pytest.raises(ValueError):
        make_grid((-7.9, 53.1, -7.899, 53.101), 1.0, 10, 32629, 256)
