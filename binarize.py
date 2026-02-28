import pathlib

import click

from lib import logging


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
        click.option(
            "--eval", "eval_mode", is_flag=True, default=False, show_default=True,
            help="Evaluation mode: the whole dataset will be processed as validation set."
        )
    ]
    for option in options[::-1]:
        func = option(func)
    return func


@click.command(help="Binarize raw notes datasets.")
@shared_options
def main(config: pathlib.Path, override: list[str], eval_mode: bool):
    import dask
    from preprocessing.api import (
        load_config_for_binarization,
        binarize_datasets,
    )
    from preprocessing.notes_binarizer import NotesBinarizer
    dask.config.set(scheduler="synchronous")

    config = load_config_for_binarization(config, overrides=override)
    if eval_mode:
        logging.debug("Using evaluation mode.")
    binarizer = NotesBinarizer(config, eval_mode=eval_mode)
    binarize_datasets(binarizer=binarizer)


if __name__ == "__main__":
    main()
