"""Tests for custom yaml tags."""

import pytest
import yaml

from fritz_ds_lib.utils.yaml import configure_yaml_loader, constractor_flatten_list


@pytest.fixture
def cfg_path(resource_path):
    """Yaml config for testing custom tags."""
    return resource_path / "utils/test_yaml_cfg.yaml"


def test_constractor_flatten_list(cfg_path):
    """Test list_flattening_constructor function for pyaml loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!flatten", constractor_flatten_list)
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, loader)
    assert cfg["flattened_list"]["elements"] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_yaml_loader(cfg_path):
    """Test works as expected."""
    loader = configure_yaml_loader()
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, loader)
    assert cfg["flattened_list"]["elements"] == [1, 2, 3, 4, 5, 6, 7, 8]
