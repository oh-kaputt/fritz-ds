"""Module for tests learners' yaml config files."""

from importlib import resources

import pytest

from fritz_ds_lib.utils.utils import load_from_file

MODULES_WITH_MODELS = [
    "fritz_ds_example.housing.models",
    "fritz_ds_example.titanic.models",
]


def get_model_configs():
    """Get all yaml files in a module at the given path."""
    return [
        path
        for module in MODULES_WITH_MODELS
        for path in resources.files(module).iterdir()
        if path.suffix == ".yaml"
    ]


class TestYamlFiles:
    """Test learner's yaml files."""

    @pytest.mark.parametrize("path_to_file", get_model_configs())
    def test_model_configs_valid(self, path_to_file):
        """Test learners features coincide with its child's features."""
        load_from_file(path_to_file)
