import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from fritz_ds_lib.core.base import ProjectBaseModel
from fritz_ds_lib.estimator.adapter import AbstractEstimator


class AverageValuePredictor(ProjectBaseModel, BaseEstimator):
    average_value: float

    def predict(self, X):
        return self.average_value * np.ones(shape=(len(X),))


class AverageValueRegressor(AbstractEstimator, RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.model_ = AverageValuePredictor(average_value=np.mean(y))
        return self

    def predict_proba(self, X):
        raise NotImplementedError

    @property
    def classes_(self):
        raise NotImplementedError
