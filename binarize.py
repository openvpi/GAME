import pathlib

import click

from lib import logging


@click.command(help="Binarize raw notes datasets.")
@click.option(
    "--config", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the configuration file."
)
@click.option(
    "--override", multiple=True,
    type=click.STRING, required=False,
    help="Override configuration values in dotlist format."
)
@click.option(
    "--eval", "eval_mode", is_flag=True, default=False, show_default=True,
    help="Evaluation mode: the whole dataset will be processed as validation set."
)
def main(config: pathlib.Path, override: list[str], eval_mode: bool):
    from preprocessing.api import (
        load_config_for_binarization,
        binarize_datasets,
    )
    from preprocessing.notes_binarizer import NotesBinarizer

    config = load_config_for_binarization(config, overrides=override)
    if eval_mode:
        logging.debug("Using evaluation mode.")
    binarizer = NotesBinarizer(config, eval_mode=eval_mode)
    binarize_datasets(binarizer=binarizer)


if __name__ == "__main__":
    main()
