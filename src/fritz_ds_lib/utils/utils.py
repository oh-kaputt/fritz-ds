import logging
from importlib import import_module, resources
from pathlib import Path
from typing import Any, Union

import pandas as pd
import yaml
from pydantic import BaseModel, TypeAdapter

from fritz_ds_lib.core.cereal import import_object
from fritz_ds_lib.utils.yaml import configure_yaml_loader


def read_config_file(package_path, filename):
    """Read yml document.

    Args:
        package_path (str): path to package containing the config file
        filename (str):     name of the file to read

    Returns:
        dict: dict with data read in the config file
    """
    module = import_module(package_path)
    cfg = yaml.safe_load(resources.open_text(module, filename))
    return cfg


def init_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(name)s %(asctime)s %(levelname)s:%(message)s")
        )
        logger.addHandler(handler)
    return logger


def clip_negative_values(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df[col_name] = df[col_name].where(df[col_name] >= 0, 0)
    return df


def do_postpocessing(df: pd.DataFrame, col_target: str) -> pd.DataFrame:
    df = clip_negative_values(df, col_target)
    return df


def load_from_dict(cfg: dict[str, Any]) -> BaseModel:
    dotted_path = cfg.get("class")
    if dotted_path is None:
        raise ValueError("Could not find dotted import path.")
    cls = import_object(dotted_path)
    return TypeAdapter(cls).validate_python(cfg)


def load_from_package(package_path: str, filename: str) -> BaseModel:
    cfg = read_config_file(package_path, filename)
    return load_from_dict(cfg)


def load_from_file(path: Union[str, Path]) -> BaseModel:
    loader = configure_yaml_loader()
    with open(path, "r") as f:
        cfg = yaml.load(f, loader)
    return load_from_dict(cfg)
