import pytest

from fritz_ds_lib.core.base import validate_model
from fritz_ds_lib.estimator.adapter import (
    LgbmClassifier,
    LgbmRegressor,
    SklearnClassifier,
    SklearnRegressor,
)


class TestLgbmEstimator:
    @pytest.mark.parametrize(
        "estimator, expected",
        [("LgbmClassifier", LgbmClassifier), ("LgbmRegressor", LgbmRegressor)],
    )
    def test_from_dict(self, estimator, expected):
        cfg = {
            "class": f"fritz_ds_lib.estimator.adapter.{estimator}",
            "cols_categorical": ["cat_feature"],
            "params": {"num_leaves": 31},
        }
        est = validate_model(cfg)
        assert isinstance(est, expected)


class TestSklearnEstimator:
    @pytest.mark.parametrize(
        "estimator, expected",
        [
            ("SklearnClassifier", SklearnClassifier),
            ("SklearnRegressor", SklearnRegressor),
        ],
    )
    def test_from_dict(self, estimator, expected):
        cfg = {
            "class": f"fritz_ds_lib.estimator.adapter.{estimator}",
            "model_type": "sklearn.ensemble.RandomForestClassifier",
            "params": {"max_depth": 31},
        }
        est = validate_model(cfg)
        assert isinstance(est, expected)
