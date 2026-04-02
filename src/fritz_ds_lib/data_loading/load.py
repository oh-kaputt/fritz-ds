from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, final

import pandas as pd
from pandas import DataFrame
from pydantic import (
    BaseModel,
    SerializeAsAny,
    SkipValidation,
    field_validator,
    model_validator,
)
from sklearn.datasets import fetch_openml
from typing_extensions import Self

from fritz_ds_lib.core.base import ProjectBaseModel
from fritz_ds_lib.core.names import DatasetType
from fritz_ds_lib.data_loading.utils import (
    change_types,
    edit_column_names,
    rename_columns,
)
from fritz_ds_lib.model_selection.split import AbstractTrainTestSpliter
from fritz_ds_lib.utils.utils import load_from_dict


class AbstractRawDataLoader(BaseModel):
    @abstractmethod
    def _load(self) -> DataFrame:
        """Load data."""

    def _preprocess(self, df: DataFrame) -> DataFrame:
        """Do strictly required data cleanup before train_test split.

        A function that should be used only for cases when the raw data
        required cleaning before the train/test split.
        For example: only nan values in the attributes frame with existing target.
        """
        return df

    @final
    def load(self) -> DataFrame:
        """Load data."""
        df = self._load()
        df = self._preprocess(df)
        return df


class OpenMlRawDataLoader(AbstractRawDataLoader):
    name: str
    version: int

    @final
    def _load(self) -> DataFrame:
        """Load data."""
        X, y = fetch_openml(name=self.name, version=self.version, return_X_y=True)
        return pd.concat([X, y], axis=1)


class AbstractLocalRawDataLoader(AbstractRawDataLoader):
    path: Union[str, Path]

    @abstractmethod
    def _load_data_from_file(self) -> DataFrame:
        """Load data from local file."""

    @final
    def _load(self) -> DataFrame:
        """Load data."""
        df = self._load_data_from_file()
        return df


class ExcelRawDataLoader(AbstractLocalRawDataLoader):
    sheet_name: str

    @final
    def _load_data_from_file(self) -> DataFrame:
        """Load data from local file."""
        return pd.read_excel(
            self.path,
            sheet_name=self.sheet_name,
        )


class CsvRawDataLoader(AbstractLocalRawDataLoader):
    @final
    def _load_data_from_file(self) -> DataFrame:
        """Load data from local file."""
        return pd.read_csv(self.path)


RawDataLoader = TypeVar("RawDataLoader", bound="AbstractRawDataLoader")


class DataLoader(ProjectBaseModel):
    train: SkipValidation[SerializeAsAny[RawDataLoader]]
    validation: Optional[SkipValidation[SerializeAsAny[RawDataLoader]]] = None
    test: Optional[SkipValidation[SerializeAsAny[RawDataLoader]]] = None
    spliter: Optional[SkipValidation[SerializeAsAny[AbstractTrainTestSpliter]]] = None
    target_name: Optional[str] = None
    renaming_mapping: dict[str, str]
    parse_dates: list[str]
    dtype: dict[str, str]

    @model_validator(mode="after")
    def _validate_loaders_and_spliter(self) -> Self:
        if self.spliter is None:
            return self
        for dset in ["validation", "test"]:
            if (getattr(self, dset) is not None) and (
                getattr(self.spliter, dset) is not None
            ):
                raise ValueError(f"Dataset {dset} is provided. No work for the spliter.")
        if self.test is None and self.spliter.test is None:
            raise ValueError(
                "Either test dataset or its size in the spliter must be provided."
            )
        return self

    @field_validator("train", "validation", "test", mode="before")
    @classmethod
    def _validate_raw_loaders(
        cls, value: dict[str, Any] | RawDataLoader
    ) -> RawDataLoader:
        if isinstance(value, dict):
            value = load_from_dict(value)
        return value

    @field_validator("spliter", mode="before")
    @classmethod
    def _validate_spliter(cls, value: dict[str, Any] | spliter) -> spliter:
        if isinstance(value, dict):
            value = load_from_dict(value)
        return value

    def _get_loader(self, dataset: DatasetType) -> RawDataLoader:
        """Find the right loader for this dataset."""
        loader = getattr(self, dataset)
        if loader is None:
            loader = self.train
        return loader

    def _split_needed(self, dataset: DatasetType) -> bool:
        """Return true if we need to split the train dataset."""
        if getattr(self, dataset) is None:
            return True
        if dataset == "train" and self.spliter is not None:
            return True
        return False

    def _validate_request(self, dataset: DatasetType) -> None:
        """Raise error if a dataset is requested that is against the config."""
        if getattr(self, dataset) is None and (
            self.spliter is None or getattr(self.spliter, f"{dataset}") is None
        ):
            raise ValueError(
                f"No {dataset} dataset is provided "
                f"and {dataset} of the spliter is set to zero."
            )

    def load(
        self, dataset: DatasetType, return_x_y: bool = False
    ) -> tuple[DataFrame, Optional[DataFrame]]:

        self._validate_request(dataset)
        loader = self._get_loader(dataset)
        df = loader.load()
        if self._split_needed(dataset):
            df = self.spliter.split(df, dataset)

        df = change_types(df, self.dtype, self.parse_dates)
        df = rename_columns(df, self.renaming_mapping)
        df = edit_column_names(df)

        # test data may include no target col
        if not return_x_y or (dataset == "test" and self.target_name not in df.columns):
            return df, None

        if self.target_name not in df.columns:
            raise ValueError(f"Column {self.target_name} is not in the dataframe.")
        y = df[[self.target_name]]
        df.drop(columns=self.target_name, inplace=True)
        return df, y
