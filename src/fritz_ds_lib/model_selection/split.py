from abc import abstractmethod
from datetime import datetime
from typing import Annotated, Optional, Union

from pandas import DataFrame
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from fritz_ds_lib.core.names import DatasetType

BTW_0_1 = Annotated[float, Field(gt=0.0, lt=1.0)]


class AbstractTrainTestSpliter(BaseModel):
    train: Union[BTW_0_1, datetime]
    validation: Union[BTW_0_1, datetime] = None
    test: Union[BTW_0_1, datetime] = None

    @abstractmethod
    def split(self, df: DataFrame, dataset_type: DatasetType) -> DataFrame:
        pass


class SklearnTrainTestSpliter(AbstractTrainTestSpliter):
    train: BTW_0_1 = None
    validation: BTW_0_1 = None
    test: BTW_0_1 = None

    random_state: int
    shuffle: bool = True
    stratify: Optional[str] = None  # col name in X dataframe

    def split(self, df: DataFrame, dataset_type: DatasetType) -> DataFrame:
        train_valid = self.train
        if self.validation is not None:
            train_valid += self.validation

        arrays = [df] if self.stratify is None else [df, df[self.stratify]]
        if self.test is not None:
            X, X_test, *rest = train_test_split(
                *arrays,
                train_size=train_valid,
                test_size=self.test,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=df.get(self.stratify),
            )
        else:
            X, rest = df, [df.get(self.stratify)]

        if self.validation is not None:
            X_train, X_validation = train_test_split(
                X,
                # round to avoid computer arithmetic problems
                train_size=round(self.train / train_valid, 4),
                test_size=round(self.validation / train_valid, 4),
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=rest[0] if self.stratify is not None else None,
            )
        else:
            X_train = X
        return locals()[f"X_{dataset_type}"]


class TimeBasedTrainTestSpliter(AbstractTrainTestSpliter):
    train: datetime = None
    validation: datetime = None
    test: datetime = None
    time_col: str

    def split(self, df: DataFrame, dataset_type: DatasetType) -> DataFrame:
        valid_test = self.validation if self.validation is not None else self.test
        match dataset_type:
            case "test":
                if self.test is None:
                    raise ValueError("Test is not in the spliter config.")
                return df[df[self.time_col] >= self.test]
            case "validation":
                if self.validation is None:
                    raise ValueError("Validation is not in the spliter config.")
                return df[
                    df[self.time_col].between_time(
                        start_time=self.validation, end_time=self.test
                    )
                ]
            case "train":
                return df[df[self.time_col] < valid_test]
