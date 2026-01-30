import pathlib

import click
import dask

from lib import logging
from lib.config.formatter import ModelFormatter
from lib.config.io import load_raw_config
from lib.config.schema import RootConfig, BinarizerConfig, ConfigurationScope

__all__ = [
    "binarize_datasets",
]

dask.config.set(scheduler="synchronous")


def _load_and_log_config(config_path: pathlib.Path, scope: int = 0, overrides: list[str] = None) -> RootConfig:
    config = load_raw_config(config_path, overrides)
    config = RootConfig.model_validate(config, scope=scope)
    config.resolve(scope_mask=scope)
    config.check(scope_mask=scope)
    formatter = ModelFormatter()
    print(formatter.format(config.binarizer))
    return config


def binarize_datasets(
        binarizer_cls, binarizer_config: BinarizerConfig
):
    from preprocessing.binarizer_base import BaseBinarizer
    if not issubclass(binarizer_cls, BaseBinarizer):
        raise ValueError(f"binarizer_cls must be a subclass of {BaseBinarizer.__name__}")
    logging.info(f"Starting binarizer: {binarizer_cls.__name__}.")
    binarizer = binarizer_cls(binarizer_config)
    binarizer.process()
    logging.success("Binarization completed.")


def shared_options(func):
    options = [
        click.option(
            "--config", type=click.Path(
                exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
            ),
            required=True,
            help="Path to the configuration file."
        ),
        click.option(
            "--override", multiple=True,
            type=click.STRING, required=False,
            help="Override configuration values in dotlist format."
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


@click.command(help="Binarize raw notes datasets.")
@shared_options
def main(config: pathlib.Path, override: list[str]):
    config = _load_and_log_config(config, overrides=override, scope=ConfigurationScope.ESTIMATION)
    from preprocessing.notes_binarizer import NotesBinarizer
    binarize_datasets(
        NotesBinarizer, config.binarizer
    )


if __name__ == "__main__":
    main()
