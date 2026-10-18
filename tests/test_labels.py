import numpy as np

from peatland.labels import CLC_CODES, IDX_LUT, PEAT, corine_code_to_class, remap_corine, remap_worldcover


def test_peat_code_maps_to_peat_class():
    assert corine_code_to_class(412) == PEAT
    assert corine_code_to_class(411) == 5
    assert corine_code_to_class(111) == 0 and corine_code_to_class(211) == 1 and corine_code_to_class(311) == 2
    assert corine_code_to_class(512) == 6 and corine_code_to_class(999) == 255


def test_remap_code_scheme():
    raw = np.array([[412, 311], [111, 0]], np.int16)
    lab, scheme = remap_corine(raw)
    assert scheme == "code(3-digit)"
    assert lab.tolist() == [[4, 2], [0, 255]]


def test_remap_index_scheme_matches_code_scheme():
    idx = CLC_CODES.index(412) + 1                      # Copernicus index of peat bogs
    raw = np.array([[idx, CLC_CODES.index(311) + 1], [CLC_CODES.index(111) + 1, 48]], np.int16)
    lab, scheme = remap_corine(raw)
    assert scheme == "index(1-44)"
    assert lab.tolist() == [[4, 2], [0, 255]]
    assert IDX_LUT[idx] == PEAT


def test_worldcover_proxy_mapping():
    lab = remap_worldcover(np.array([10, 40, 80, 90, 70], np.uint8))
    assert lab.tolist() == [2, 1, 6, 4, 255]
