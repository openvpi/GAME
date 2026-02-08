import pathlib

import click

from lib import logging


# noinspection PyUnusedLocal
def _validate_d3pm_ts(ctx, param, value) -> list[float] | None:
    if value is None:
        return None
    try:
        ts = [float(t.strip()) for t in value.split(",")]
        if not ts:
            raise ValueError("At least one T value must be provided.")
        if any(t < 0 or t >= 1 for t in ts):
            raise ValueError("All T values must be in the range (0, 1).")
        return ts
    except Exception as e:
        raise click.BadParameter(f"Invalid T values: {e}")


# noinspection PyUnusedLocal
def _validate_exts(ctx, param, value) -> set[str]:
    try:
        exts = {"." + ext.strip().lower() for ext in value.split(",")}
        if not exts:
            raise ValueError("At least one extension must be provided.")
        return exts
    except Exception as e:
        raise click.BadParameter(f"Invalid extensions: {e}")


# noinspection PyUnusedLocal
def _validate_output_formats(ctx, param, value) -> set[str]:
    try:
        formats = {fmt.strip().lower() for fmt in value.split(",")}
        supported_formats = {"mid", "txt", "csv"}
        if not formats.issubset(supported_formats):
            raise ValueError(
                f"Unsupported formats: {formats - supported_formats}. "
                f"Supported formats: {supported_formats}"
            )
        return formats
    except Exception as e:
        raise click.BadParameter(f"Invalid output formats: {e}")


def _t0_nstep_to_ts(t0: float, nsteps: int) -> list[float]:
    if nsteps == 1:
        return [t0]
    step = (1 - t0) / (nsteps - 1)
    return [t0 + i * step for i in range(nsteps)]


def _get_language_id(language: str, lang_map: dict[str, int]) -> int:
    if language and lang_map:
        if language not in lang_map:
            raise ValueError(
                f"Language '{language}' not supported by the segmentation model. "
                f"Supported languages: {', '.join(lang_map.keys())}"
            )
        language_id = lang_map[language]
    else:
        language_id = 0
    return language_id


def _parse_filemap(path: pathlib.Path, exts: set[str], glob: str | None) -> dict[str, pathlib.Path]:
    if path.is_file():
        return {path.name: path}
    elif path.is_dir():
        if glob:
            files = [f for f in path.rglob(glob) if f.is_file()]
        else:
            files = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in exts]
        filemap = {
            f.relative_to(path).as_posix(): f
            for f in files
        }
        if not filemap:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return filemap
    else:
        raise ValueError(f"Invalid path: {path}")


@click.group()
def main():
    pass


def shared_options(func):
    options = [
        click.option(
            "--seg", type=click.Path(
                exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
            ),
            required=True,
            help="Path to the segmentation model."
        ),
        click.option(
            "--est", type=click.Path(
                exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
            ),
            required=True,
            help="Path to the estimation model."
        ),
        click.option(
            "--language", type=str, default=None, show_default=False,
            help="Language code for better segmentation if supported."
        ),
        click.option(
            "--batch-size", type=click.IntRange(min=1), default=4, show_default=True,
            help="Batch size for inference."
        ),
        click.option(
            "--num-workers", type=click.IntRange(min=0), default=0, show_default=True,
            help="Number of worker processes for dataloader."
        ),
        click.option(
            "--seg-threshold", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), default=0.3, show_default=True,
            help="Boundary decoding threshold for segmentation model."
        ),
        click.option(
            "--seg-radius", type=click.FloatRange(min=0.01), default=0.02, show_default=True,
            help="Boundary decoding radius for segmentation model."
        ),
        click.option(
            "--t0", "--seg-d3pm-t0", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), default=0.0, show_default=True,
            help="Starting T value (t0) for segmentation model."
        ),
        click.option(
            "--nsteps", "--seg-d3pm-nsteps", type=click.IntRange(min=1), default=4, show_default=True,
            help="Number of D3PM sampling steps for segmentation model."
        ),
        click.option(
            "--ts", "--seg-d3pm-ts", type=str, default=None, show_default=False,
            callback=_validate_d3pm_ts,
            help=(
                "Custom T values for D3PM sampling in segmentation model, separated by commas. "
                "Overrides --t0 and --nsteps if provided."
            )
        ),
        click.option(
            "--est-threshold", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), default=0.2, show_default=True,
            help="Presence detecting threshold for estimation model."
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


@main.command(name="extract", help="Extract MIDI from single or multiple audio files.")
@click.argument(
    "path", type=click.Path(
        exists=True, dir_okay=True, file_okay=True, readable=True, path_type=pathlib.Path
    ),
)
@shared_options
@click.option(
    "--exts", type=str, default="wav,flac,mp3,aac,ogg", show_default=True,
    callback=_validate_exts,
    help=(
            "List of audio file extensions to process, separated by commas. "
            "Ignored if a single audio file is provided."
    )
)
@click.option(
    "--glob", type=str, default=None, show_default=False,
    help=(
            "Glob pattern to filter audio files (i.e. *.wav). "
            "Overrides --exts if provided. Ignored if a single audio file is provided."
    )
)
@click.option(
    "--output-formats", type=str, default="mid", show_default=True,
    callback=_validate_output_formats,
    help=(
            "List of output formats to save the extracted results, separated by commas. "
            "Supported formats: mid, txt, csv."
    )
)
@click.option(
    "--tempo", type=click.FloatRange(min=0, min_open=True), default=120, show_default=True,
    help="Tempo (in BPM) to save MIDI files with. Ignored for non-MIDI output formats."
)
@click.option(
    "--pitch-format", type=click.Choice(["number", "name"]), default="name", show_default=True,
    help=(
            "Format to save pitch in text-based output formats. "
            "Ignored for MIDI output. 'number' saves MIDI pitch numbers, while 'name' saves note names (e.g. C4+10)."
    )
)
@click.option(
    "--round-pitch", is_flag=True, default=False, show_default=True,
    help=(
            "Whether to round pitch values to the nearest integer in text-based output formats. "
            "Ignored for MIDI output. If not set, pitch values will be saved with decimals or cents."
    )
)
@click.option(
    "--output-dir", type=click.Path(
        exists=False, dir_okay=True, file_okay=False, writable=True, path_type=pathlib.Path
    ),
    default=None, show_default=False,
    help=(
            "Directory to save the extracted results. "
            "If not provided, results will be saved in the same directory as the input."
    )
)
def extract(
        path: pathlib.Path,
        seg: pathlib.Path,
        est: pathlib.Path,
        language: str,
        batch_size: int,
        num_workers: int,
        seg_threshold: float,
        seg_radius: float,
        t0: float,
        nsteps: int,
        ts: list[float],
        est_threshold: float,
        exts: set[str],
        glob: str,
        output_formats: set[str],
        pitch_format: str,
        round_pitch: bool,
        tempo: float,
        output_dir: pathlib.Path,
):
    if ts is None:
        ts = _t0_nstep_to_ts(t0, nsteps)
    filemap = _parse_filemap(path, exts, glob)
    if output_dir is None:
        output_dir = path if path.is_dir() else path.parent

    from lightning_utilities.core.rank_zero import rank_zero_info
    from inference.api import (
        load_segmentation_inference_model,
        load_estimation_inference_model,
        run_inference,
    )
    from inference.slicer2 import Slicer
    from inference.data import SlicedAudioFileIterableDataset
    from inference.callbacks import SaveMidiCallback, SaveTextCallback

    segmentation_model, lang_map = load_segmentation_inference_model(seg)
    estimation_model = load_estimation_inference_model(est)
    language_id = _get_language_id(language, lang_map)

    seg_sr = segmentation_model.inference_config.features.audio_sample_rate
    if seg_sr != (est_sr := estimation_model.inference_config.features.audio_sample_rate):
        logging.warning(
            f"Samplerate of segmentation model ({seg_sr}) differs from segmentation model ({est_sr}). "
            "Audio will be resampled during inference, which may affect performance.",
            callback=rank_zero_info
        )
    dataset = SlicedAudioFileIterableDataset(
        filemap=filemap,
        samplerate=seg_sr,
        slicer=Slicer(
            sr=seg_sr,
        ),
        language=language_id,
    )
    callbacks = []
    if "mid" in output_formats:
        callbacks.append(SaveMidiCallback(
            output_dir=output_dir,
            tempo=tempo,
        ))
    if "txt" in output_formats:
        callbacks.append(SaveTextCallback(
            output_dir=output_dir,
            file_format="txt",
            pitch_format=pitch_format,
            round_pitch=round_pitch,
        ))
    if "csv" in output_formats:
        callbacks.append(SaveTextCallback(
            output_dir=output_dir,
            file_format="csv",
            pitch_format=pitch_format,
            round_pitch=round_pitch,
        ))
    run_inference(
        segmentation_model=segmentation_model,
        estimation_model=estimation_model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        callbacks=callbacks,
        segmentation_threshold=seg_threshold,
        segmentation_radius=seg_radius,
        segmentation_d3pm_ts=ts,
        estimation_threshold=est_threshold,
    )
    logging.success("Inference completed.", callback=rank_zero_info)


if __name__ == '__main__':
    main()
