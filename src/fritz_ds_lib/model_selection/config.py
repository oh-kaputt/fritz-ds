from typing import Annotated, Any, Type

from pydantic import BeforeValidator, SerializeAsAny, SkipValidation, field_validator

from fritz_ds_lib.core.base import ProjectBaseModel, validate_model
from fritz_ds_lib.core.cereal import import_object
from fritz_ds_lib.core.names import ArrayLike
from fritz_ds_lib.model_selection.cv_split import SklearnCvSplitProtocol


class SklearnCvSplit(ProjectBaseModel):
    sklearn_cls: Type[SklearnCvSplitProtocol]
    params: dict[str, Any]

    @field_validator("sklearn_cls", mode="before")
    def _validate_model_type(cls, value: Any) -> SklearnCvSplitProtocol:
        if isinstance(value, str):
            value = import_object(value)
        return value

    def split(self, X: ArrayLike, y: ArrayLike = None, groups: ArrayLike = None):
        """Generate indices to split data into training and test set."""
        return self.sklearn_cls(**self.params).split(X, y, groups)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.sklearn_cls(**self.params).get_n_splits()


class CvConfig(ProjectBaseModel):
    param_grid: dict[str, Any]
    scoring: str
    spliter: Annotated[
        SkipValidation[SerializeAsAny[SklearnCvSplit]], BeforeValidator(validate_model)
    ]
