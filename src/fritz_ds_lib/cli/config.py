import copy
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SkipValidation, field_validator, model_validator

from fritz_ds_lib.core.model import ModelConfig
from fritz_ds_lib.data_loading.load import DataLoader
from fritz_ds_lib.utils.utils import load_from_file


class AppConfig(BaseModel):
    loader: SkipValidation[DataLoader]
    model_cfg: SkipValidation[ModelConfig]
    model_cfg_folder: str
    output_folder: str
    cv_model_file_name: str
    trained_model_file_name: str
    pred_df_file_name: str

    @model_validator(mode="before")
    @classmethod
    def _validate_model_cfg(cls, values: Any) -> Any:
        if isinstance(values["model_cfg"], str):
            path = Path(values["model_cfg_folder"]) / values["model_cfg"]
            values["model_cfg"] = load_from_file(path)
        return values

    @field_validator("loader", mode="before")
    @classmethod
    def _validate_loader(cls, value: Any) -> Any:
        if isinstance(value, (str, Path)):
            return load_from_file(value)
        return value

    def _output_folder(self, artifact: str) -> Path:
        folder = Path(self.output_folder) / self.model_cfg.col_target / artifact
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def train_model_path(self):
        folder = self._output_folder("models")
        filename = self.trained_model_file_name.format(model_name=self.model_cfg.name)
        return folder / filename

    @property
    def cv_model_path(self):
        folder = self._output_folder("models")
        filename = self.cv_model_file_name.format(model_name=self.model_cfg.name)
        return folder / filename

    @property
    def predictions_path(self):
        folder = self._output_folder("predictions")
        filename = self.pred_df_file_name.format(model_name=self.model_cfg.name)
        return folder / filename

    @property
    def iter_models(self):
        for filepath in Path(self.model_cfg_folder).glob("*.yaml"):
            params = copy.deepcopy(vars(self))
            params["model_cfg"] = filepath.name
            yield AppConfig(**params)
