import pathlib

from lib import logging
from lib.config.formatter import ModelFormatter
from lib.config.io import load_raw_config
from lib.config.schema import BinarizerConfig, RootConfig
from preprocessing.binarizer_base import BaseBinarizer

__all__ = [
    "load_config_for_binarization",
    "binarize_datasets",
]


def load_config_for_binarization(
        config_path: pathlib.Path,
        scope: int = 0,
        overrides: list[str] = None
) -> BinarizerConfig:
    config = load_raw_config(config_path, overrides)
    config = RootConfig.model_validate(config, scope=scope)
    config.resolve(scope_mask=scope)
    config.check(scope_mask=scope)
    formatter = ModelFormatter()
    print(formatter.format(config.binarizer))
    return config.binarizer


def binarize_datasets(
        binarizer: BaseBinarizer
):
    logging.info(f"Starting binarizer: {binarizer.__class__.__name__}.")
    binarizer.process()
    logging.success("Binarization completed.")
