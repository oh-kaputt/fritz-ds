from abc import abstractmethod
from typing import Any, Optional, Type, TypeVar

import lightgbm as lgbm
import numpy as np
from lightgbm import Booster, Dataset
from pydantic import field_validator
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from fritz_ds_lib.core.base import ProjectBaseModel
from fritz_ds_lib.core.cereal import import_object

T = TypeVar("T", bound=BaseEstimator)


class AbstractEstimator(ProjectBaseModel):
    params: dict[str, Any]
    model_: Optional[T] = None

    # TODO: use pydantic's third party type validation
    @field_validator("params")
    def _validate_sklearn_classes(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        for k, v in values.items():
            if isinstance(v, dict) and "sklearn_class" in v:
                kls = v.pop("sklearn_class")
                values[k] = import_object(kls)(**v)
        return values

    def __sklearn_clone__(self):
        values = {k: v for (k, v) in dict(vars(self)).items() if k != "model_"}
        return self.__class__(**values)

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.model_.predict(X)

    @abstractmethod
    def predict_proba(self, X):
        pass

    @property
    @abstractmethod
    def classes_(self):
        pass

    def set_params(self, **params):
        self.params.update(**params)


class SklearnEstimator(AbstractEstimator):
    model_type: Type[T]

    @field_validator("model_type", mode="before")
    def _validate_model_type(cls, value: Any) -> Type[T]:
        if isinstance(value, str):
            value = import_object(value)
        return value

    def fit(self, X, y):
        if self.model_ is not None:
            raise ValueError("Model has already been fitted.")
        self.model_ = self.model_type(**self.params)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def classes_(self):
        return self.model_.classes_


class SklearnClassifier(SklearnEstimator, ClassifierMixin, BaseEstimator): ...


class SklearnRegressor(SklearnEstimator, RegressorMixin, BaseEstimator): ...


class LgbmEstimator(AbstractEstimator):
    model_: Optional[Booster] = None
    cols_categorical: list[str]

    def fit(self, X, y):
        if self.model_ is not None:
            raise ValueError("Model has already been fitted.")
        dataset = Dataset(X, y, categorical_feature=self.cols_categorical)
        self.model_ = lgbm.train(params=self.params, train_set=dataset)
        return self

    def predict_proba(self, X):
        y_pred = self.model_.predict(X)
        if len(y_pred.shape) == 1:
            y_pred = np.c_[1 - y_pred, y_pred]
        return y_pred

    @property
    def classes_(self):
        # TODO: find class labels in the booster
        return np.array([0, 1])


class LgbmClassifier(LgbmEstimator, ClassifierMixin, BaseEstimator): ...


class LgbmRegressor(LgbmEstimator, RegressorMixin, BaseEstimator): ...
