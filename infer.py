import glob
import pathlib
from typing import Any

import click

from lib import logging
from lib.config.schema import ValidationConfig

_OPT_KEY_BATCH_SIZE = "batch_size"
_OPT_KEY_NUM_WORKERS = "num_workers"
_OPT_KEY_SEG_THRESHOLD = "seg_threshold"
_OPT_KEY_SEG_RADIUS = "seg_radius"
_OPT_KEY_SEG_D3PM_T0 = "t0"
_OPT_KEY_SEG_D3PM_NSTEPS = "nsteps"
_OPT_KEY_EST_THRESHOLD = "est_threshold"


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


# noinspection PyUnusedLocal
def _validate_uv_vocab(ctx, param, value) -> set[str] | None:
    if value is None:
        return None
    try:
        vocab = {v.strip() for v in value.split(",") if v.strip()}
        return vocab
    except Exception as e:
        raise click.BadParameter(f"Invalid UV vocab: {e}")


# noinspection PyUnusedLocal
def _validate_path_or_glob(ctx, param, value) -> str:
    try:
        paths = []
        for v in value:
            if glob.has_magic(v):
                paths.extend(glob.glob(v, recursive=True))
            else:
                paths.append(v)
        if not paths:
            raise FileNotFoundError(f"No files found for paths: {value}")
        paths = [
            pathlib.Path(p)
            for p in paths
        ]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Path does not exist: {p}")
            if not p.is_file():
                raise FileNotFoundError(f"Path is not a file: {p}")
        return paths
    except Exception as e:
        raise click.BadParameter(f"Invalid path or glob: {e}")


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


def _parse_filemap(path: pathlib.Path, exts: set[str], glb: str | None) -> dict[str, pathlib.Path]:
    if path.is_file():
        return {path.name: path}
    elif path.is_dir():
        if glb:
            files = [f for f in path.rglob(glb) if f.is_file()]
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


def shared_options(func=None, *, defaults: dict[str, Any] = None):
    if defaults is None:
        defaults = {}

    options = [
        click.option(
            "-m", "--model", type=click.Path(
                exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
            ),
            required=True,
            help="Path to the model."
        ),
        click.option(
            "-l", "--language", type=str, default=None, show_default=False,
            help="Language code for better segmentation if supported."
        ),
        click.option(
            "--batch-size", type=click.IntRange(min=1), show_default=True,
            default=defaults.get(_OPT_KEY_BATCH_SIZE, 4),
            help="Batch size for inference."
        ),
        click.option(
            "--num-workers", type=click.IntRange(min=0), show_default=True,
            default=defaults.get(_OPT_KEY_NUM_WORKERS, 0),
            help="Number of worker processes for dataloader."
        ),
        click.option(
            "--seg-threshold", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_THRESHOLD, 0.2),
            help="Boundary decoding threshold for segmentation model."
        ),
        click.option(
            "--seg-radius", type=click.FloatRange(min=0.01), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_RADIUS, 0.02),
            help="Boundary decoding radius for segmentation model."
        ),
        click.option(
            "--t0", "--seg-d3pm-t0", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_D3PM_T0, 0.0),
            help="Starting T value (t0) of D3PM for segmentation model."
        ),
        click.option(
            "--nsteps", "--seg-d3pm-nsteps", type=click.IntRange(min=1), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_D3PM_NSTEPS, 8),
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
            ), show_default=True,
            default=defaults.get(_OPT_KEY_EST_THRESHOLD, 0.2),
            help="Presence detecting threshold for estimation model."
        ),
    ]

    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)


@main.command(
    name="extract",
    help="Extract MIDI from single or multiple audio files."
)
@click.argument(
    "path", type=click.Path(
        exists=True, dir_okay=True, file_okay=True, readable=True, path_type=pathlib.Path
    ),
)
@shared_options(defaults={
    _OPT_KEY_SEG_D3PM_T0: 0.0,
    _OPT_KEY_SEG_D3PM_NSTEPS: 8,
})
@click.option(
    "--input-formats", type=str, default="wav,flac,mp3,aac,ogg", show_default=True,
    callback=_validate_exts,
    help=(
            "List of audio file extensions to process, separated by commas. "
            "Ignored if a single audio file is provided."
    )
)
@click.option(
    "--glob", "glb", type=str, default=None, show_default=False,
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
        model: pathlib.Path,
        language: str,
        batch_size: int,
        num_workers: int,
        seg_threshold: float,
        seg_radius: float,
        t0: float,
        nsteps: int,
        ts: list[float],
        est_threshold: float,
        input_formats: set[str],
        glb: str,
        output_formats: set[str],
        pitch_format: str,
        round_pitch: bool,
        tempo: float,
        output_dir: pathlib.Path,
):
    filemap = _parse_filemap(path, input_formats, glb)
    if output_dir is None:
        output_dir = path if path.is_dir() else path.parent

    from lightning_utilities.core.rank_zero import rank_zero_info
    from inference.api import (
        load_inference_model,
        infer_model,
    )
    from inference.slicer2 import Slicer
    from inference.data import SlicedAudioFileIterableDataset
    from inference.callbacks import SaveCombinedMidiFileCallback, SaveCombinedTextFileCallback

    model, lang_map = load_inference_model(model)
    language_id = _get_language_id(language, lang_map)

    sr = model.inference_config.features.audio_sample_rate
    dataset = SlicedAudioFileIterableDataset(
        filemap=filemap,
        samplerate=sr,
        slicer=Slicer(
            sr=sr,
            threshold=-40.,
            min_length=1000,
            min_interval=200,
            max_sil_kept=100,
        ),
        language=language_id,
    )
    callbacks = []
    if "mid" in output_formats:
        callbacks.append(SaveCombinedMidiFileCallback(
            output_dir=output_dir,
            tempo=tempo,
        ))
    if "txt" in output_formats:
        callbacks.append(SaveCombinedTextFileCallback(
            output_dir=output_dir,
            file_format="txt",
            pitch_format=pitch_format,
            round_pitch=round_pitch,
        ))
    if "csv" in output_formats:
        callbacks.append(SaveCombinedTextFileCallback(
            output_dir=output_dir,
            file_format="csv",
            pitch_format=pitch_format,
            round_pitch=round_pitch,
        ))
    # noinspection PyArgumentList
    infer_model(
        model=model,
        dataset=dataset,
        config=ValidationConfig(
            d3pm_sample_t0=t0,
            d3pm_sample_steps=nsteps,
            d3pm_sample_ts=ts,
            boundary_decoding_threshold=seg_threshold,
            boundary_decoding_radius=round(seg_radius / model.timestep),
            note_presence_threshold=est_threshold,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        callbacks=callbacks,
    )
    logging.success("Inference completed.", callback=rank_zero_info)


@main.command(
    name="align",
    help="Generate aligned note labels with word boundaries in DiffSinger transcriptions."
)
@click.argument(
    "paths", type=str, nargs=-1, metavar="PATH_OR_GLOB",
    callback=_validate_path_or_glob,
)
@shared_options(defaults={
    _OPT_KEY_SEG_D3PM_T0: 0.5,
    _OPT_KEY_SEG_D3PM_NSTEPS: 4,
})
@click.option(
    "--save-path", type=click.Path(
        file_okay=True, dir_okay=False, writable=True, path_type=pathlib.Path
    ),
    help=(
        "Path of the output file to save the updated transcriptions. "
        "Ignored if multiple input files are provided. Overrides --save-name if provided."
    )
)
@click.option(
    "--save-name", type=str, default=None, show_default=False,
    help=(
            "Name of the updated transcription files to save. "
            "Ignored if input is a single file and --save-path is provided. "
            "If both --save-path and --save-name are not provided, the original files will be overwritten."
    )
)
@click.option(
    "--overwrite", is_flag=True, default=False, show_default=True,
    help=(
        "Whether to overwrite existing files when saving updated transcriptions."
    )
)
@click.option(
    "--uv-vocab", type=str, default="AP,SP,br,sil", show_default=True,
    callback=_validate_uv_vocab,
    help=(
        "Comma-separated list of unvoiced phoneme names."
    )
)
@click.option(
    "--uv-vocab-path", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    default=None, show_default=False,
    help=(
        "Path to a file containing unvoiced phoneme names separated by whitespace. "
        "Overrides --uv-vocab if provided."
    )
)
@click.option(
    "--uv-word-cond", type=click.Choice(["lead", "all"]),
    default="lead", show_default=True,
    help=(
        "Condition for determining whether a word is unvoiced. "
        "'lead' checks only the leading phoneme; "
        "'all' requires all phonemes to be unvoiced."
    )
)
@click.option(
    "--uv-note-cond", type=click.Choice(["predict", "follow"]),
    default="predict", show_default=True,
    help=(
        "Condition for determining whether a note is unvoiced. "
        "'predict' uses the raw model output; "
        "'follow' aligns note v/uv with the word it belongs to."
    )
)
@click.option(
    "--no-wb", is_flag=True, default=False, show_default=True,
    help=(
            "Whether to disable word-note alignment. Not recommended."
    )
)
def align(
        paths: list[pathlib.Path],
        model: pathlib.Path,
        language: str,
        batch_size: int,
        num_workers: int,
        seg_threshold: float,
        seg_radius: float,
        t0: float,
        nsteps: int,
        ts: list[float],
        est_threshold: float,
        save_path: pathlib.Path,
        save_name: str,
        overwrite: bool,
        no_wb: bool,
        uv_vocab: set[str],
        uv_vocab_path: pathlib.Path,
        uv_word_cond: str,
        uv_note_cond: str,
):
    if len(paths) > 1:
        save_path = None
    if save_path is not None:
        save_name = save_path.name
    inplace = (save_path is None) and (save_name is None)
    if inplace and not overwrite:
        raise ValueError("In-place saving requires --overwrite flag to be set.")
    if not overwrite:
        for p in paths:
            if save_path is not None and save_path.exists():
                raise FileExistsError(f"Output file already exists: {save_path}")
            if save_name is not None:
                output_file = p.parent / save_name
                if output_file.exists():
                    raise FileExistsError(f"Output file already exists: {output_file}")
    save_dir = save_path.parent if save_path else None
    use_wb = not no_wb
    # --uv-vocab-path overrides --uv-vocab; suppress UV logic when --no-wb
    if uv_vocab_path is not None:
        uv_vocab = set(uv_vocab_path.read_text(encoding="utf8").split())
    resolved_uv_vocab = uv_vocab if use_wb else None
    if use_wb and uv_note_cond == "follow" and not resolved_uv_vocab:
        logging.warning(
            "--uv-note-cond follow is set but the UV vocab is empty, so all words are "
            "treated as voiced and all notes will be forced voiced. This is not expected "
            "in most cases. Please check --uv-vocab and --uv-vocab-path options."
        )

    from lightning_utilities.core.rank_zero import rank_zero_info
    from inference.api import (
        load_inference_model,
        infer_model,
    )
    from inference.data import DiffSingerTranscriptionsDataset
    from inference.callbacks import UpdateDiffSingerTranscriptionsCallback

    model, lang_map = load_inference_model(model)
    language_id = _get_language_id(language, lang_map)

    sr = model.inference_config.features.audio_sample_rate
    dataset = DiffSingerTranscriptionsDataset(
        filelist=paths,
        samplerate=sr,
        language=language_id,
        use_wb=use_wb,
        uv_vocab=resolved_uv_vocab,
        uv_word_cond=uv_word_cond,
    )
    callbacks = [
        UpdateDiffSingerTranscriptionsCallback(
            filelist=paths,
            overwrite=inplace,
            save_dir=save_dir,
            save_filename=save_name,
            use_wb=use_wb,
            uv_vocab=resolved_uv_vocab,
            uv_word_cond=uv_word_cond,
            uv_note_cond=uv_note_cond,
        )
    ]
    # noinspection PyArgumentList
    infer_model(
        model=model,
        dataset=dataset,
        config=ValidationConfig(
            d3pm_sample_t0=t0,
            d3pm_sample_steps=nsteps,
            d3pm_sample_ts=ts,
            boundary_decoding_threshold=seg_threshold,
            boundary_decoding_radius=round(seg_radius / model.timestep),
            note_presence_threshold=est_threshold,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        callbacks=callbacks,
    )
    logging.success("Inference completed.", callback=rank_zero_info)


if __name__ == '__main__':
    main()
