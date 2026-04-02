import random
from itertools import chain
from typing import Callable, get_args

import pytest
from pandas import DataFrame

from fritz_ds_lib.core.names import DatasetType
from fritz_ds_lib.model_selection.split import SklearnTrainTestSpliter

DATASET_TYPES = get_args(DatasetType)


@pytest.fixture
def stratified_df() -> DataFrame:
    n = 100
    return DataFrame(
        {"val": range(n), "cls": random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2], k=n)}
    )


@pytest.fixture
def simple_df() -> DataFrame:
    return DataFrame({"val": range(100)})


class TestSklearnTrainTestSpliter:
    @pytest.fixture
    def get_spliter(self, request) -> Callable[..., SklearnTrainTestSpliter]:
        return lambda: SklearnTrainTestSpliter(**request.param)

    @pytest.mark.parametrize(
        "get_spliter",
        [
            {"train": 0.7, "validation": 0.15, "test": 0.15, "random_state": 42},
            {"train": 0.8, "validation": 0.2, "random_state": 43},
            {"train": 0.75, "test": 0.25, "random_state": 44},
            {
                "train": 0.7,
                "validation": 0.15,
                "test": 0.15,
                "stratify": "cls",
                "random_state": 45,
            },
            {"train": 0.8, "validation": 0.2, "stratify": "cls", "random_state": 46},
            {"train": 0.75, "test": 0.25, "stratify": "cls", "random_state": 47},
        ],
        indirect=["get_spliter"],
    )
    def test_sets_disjoint(self, stratified_df, get_spliter):
        df = stratified_df
        sets = []
        for ds in DATASET_TYPES:
            # new spliter instance for each dataset type
            spliter = get_spliter()
            if getattr(spliter, ds) is not None:
                sets.append(spliter.split(df, ds)["val"])
        assert len(set(chain(*sets))) == sum([len(s) for s in sets])
