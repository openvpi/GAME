import pathlib

import click

from lib.config.core import ConfigBaseModel
from lib.config.io import load_raw_config
from lib import logging


@click.command(help="Evaluate a model on a binary dataset.")
@click.option(
    "-d", "--dataset", type=click.Path(
        exists=True, dir_okay=True, file_okay=False, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the binary dataset directory."
)
@click.option(
    "-m", "--model", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the model."
)
@click.option(
    "-c", "--config", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the configuration file to read validation arguments from."
)
@click.option(
    "--override", multiple=True,
    type=click.STRING, required=False,
    help=(
            "Override configuration values in dotlist format. "
            "This only affects keys under training.validation "
            "and this prefix should not be included in the dotlist keys."
    ),
)
@click.option(
    "-o", "--save-dir", type=click.Path(
        exists=False, dir_okay=True, file_okay=False, writable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Directory to save evaluation results."
)
@click.option(
    "--plot", is_flag=True, default=False, show_default=True,
    help="Whether to save comparison plots for each sample."
)
@click.option(
    "--batch-size", type=click.IntRange(min=1), show_default=True,
    default=4,
    help="Batch size for inference."
)
@click.option(
    "--num-workers", type=click.IntRange(min=0), show_default=True,
    default=0,
    help="Number of worker processes for dataloader."
)
@click.option(
    "--precision", type=click.STRING, default="32-true", show_default=True,
    help="Precision for evaluation."
)
def main(
        dataset: pathlib.Path,
        model: pathlib.Path,
        config: pathlib.Path,
        override: list[str],
        save_dir: pathlib.Path,
        plot: bool,
        batch_size: int,
        num_workers: int,
        precision: str,
):
    from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
    from inference.api import (
        load_config_for_evaluation,
        load_inference_model,
        infer_model,
    )
    from training.me_module import MIDIExtractionDataset
    from inference.callbacks import VisualizeNoteComparisonCallback, ExportMetricSummaryCallback

    @rank_zero_only
    def _check_file_and_config(file: pathlib.Path, cfg: ConfigBaseModel):
        cfg_load = load_raw_config(file, inherit=False, overrides=None)
        if cfg_load != cfg.model_dump():
            raise RuntimeError(
                f"Contents of '{file}' do not match the configuration. "
                f"If you edited the configuration file, please re-binarize the dataset."
            )

    model, _ = load_inference_model(model)
    _check_file_and_config(dataset / "feature.yaml", model.inference_config.features)
    validation_config = load_config_for_evaluation(config, overrides=override)

    dataset = MIDIExtractionDataset(
        data_dir=dataset,
        prefix="valid",
    )
    num_digits = len(str(len(dataset)))
    callbacks = [
        ExportMetricSummaryCallback(save_path=save_dir / "summary.json"),
    ]
    if plot:
        callbacks.append(
            VisualizeNoteComparisonCallback(
                save_dir=save_dir / "plots",
                num_digits=num_digits,
            )
        )

    infer_model(
        model=model,
        dataset=dataset,
        config=validation_config,
        callbacks=callbacks,
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
        mode="evaluate",
    )
    logging.success("Evaluation completed.", callback=rank_zero_info)


if __name__ == '__main__':
    main()
