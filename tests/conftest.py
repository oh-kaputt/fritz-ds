from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parents[1]
RESOURCE_PATH = SRC_ROOT / "tests/resources"


@pytest.fixture(scope="session")
def resource_path() -> Path:
    """Resource path fixture."""
    return RESOURCE_PATH
