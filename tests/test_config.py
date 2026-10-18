import copy

import pytest

from peatland.config import ConfigError, load_config, validate_config


def test_default_config_is_valid():
    assert validate_config(load_config("config.yaml"))["seed"] == 42


@pytest.mark.parametrize("mutate", [
    lambda c: c["tiling"].update(block=100),                       # block not a multiple of tile
    lambda c: c["tiling"].update(split={"train": 0.5, "val": 0.2, "test": 0.2}),
    lambda c: c["imagery"].update(bands=["B02", "B03"]),           # missing red / NIR / SWIR
    lambda c: c["labels"].update(source="landsat"),
    lambda c: c["study_areas"][0].update(bbox=[-7.5, 53.4, -7.9, 53.1]),   # min > max
    lambda c: c.pop("unet"),
])
def test_bad_configs_are_rejected(mutate):
    cfg = copy.deepcopy(load_config("config.yaml"))
    mutate(cfg)
    with pytest.raises(ConfigError):
        validate_config(cfg)
