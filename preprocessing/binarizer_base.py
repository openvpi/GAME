import abc
import json
import pathlib
import random
from dataclasses import dataclass

import numpy
import torch
import tqdm
import yaml

from lib import logging
from lib.config.schema import BinarizerConfig
from lib.indexed_dataset import IndexedDatasetBuilder
from lib.multiprocess import chunked_multiprocess_run
from modules.commons.tts_modules import LengthRegulator

ACCEPTED_AUDIO_FORMATS = {".wav", ".flac"}


@dataclass
class MetadataItem(abc.ABC):
    name: str
    language: str
    waveform_fn: pathlib.Path
    estimated_duration: float


@dataclass
class DataSample:
    path: str
    name: str
    length: int
    data: dict[str, int | float | numpy.ndarray]
    error: str = None


class BaseBinarizer(abc.ABC):
    __data_attrs__: list[str] = None

    def __init__(self, config: BinarizerConfig, eval_mode=False):
        self.config = config
        self.eval_mode = eval_mode
        self.data_dir: pathlib.Path = config.data_dir_resolved
        self.lang_map: dict[str, int] = {}
        self.timestep = config.features.timestep

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = LengthRegulator()
        # Lazy-initialized modules
        self.mel_spec = None

        self.valid_items: list[MetadataItem] = []
        self.train_items: list[MetadataItem] = []

    @abc.abstractmethod
    def load_metadata(self, subset_dir: pathlib.Path) -> list[MetadataItem]:
        pass

    @abc.abstractmethod
    def process_item(self, item: MetadataItem) -> DataSample:
        pass

    def split_dataset(self, metadata_list: list[MetadataItem]):
        if self.eval_mode:
            # Put all items into validation set and leave training set empty.
            self.valid_items.extend(metadata_list)
            self.valid_items.sort(key=lambda itm: itm.estimated_duration, reverse=True)
        else:
            validation_indices = sorted(random.sample(
                range(len(metadata_list)),
                k=min(self.config.validation_count, len(metadata_list))
            ))
            validation_indices_set = set(validation_indices)
            for i, item in enumerate(metadata_list):
                if i in validation_indices_set:
                    self.valid_items.append(item)
                else:
                    self.train_items.append(item)
            self.train_items.sort(key=lambda itm: itm.estimated_duration, reverse=True)
        if not self.eval_mode and not self.train_items:
            raise RuntimeError("Training set is empty.")
        if not self.valid_items:
            raise RuntimeError("Validation set is empty.")

    def free_lazy_modules(self):
        """
        The lazy-initialized PyTorch modules should be freed before multiprocessing,
        because of CUDA IPC (shared memory) issues on Windows platforms.
        Reference:
          - https://github.com/pytorch/pytorch/issues/100358
          - https://github.com/Xiao-Chenguang/FedMind/issues/61
        """
        self.mel_spec = None

    def process_items(self, items: list[MetadataItem], prefix: str, multiprocessing=True):
        builder = IndexedDatasetBuilder(
            path=self.data_dir, prefix=prefix, allowed_attr=self.__data_attrs__
        )
        if multiprocessing and self.config.num_workers > 0:
            logging.debug(f"Processing {prefix} items with {self.config.num_workers} worker(s).")
            self.free_lazy_modules()
            iterable = chunked_multiprocess_run(
                self.process_item, [(item,) for item in items], num_workers=self.config.num_workers
            )
        else:
            logging.debug(f"Processing {prefix} items in main process.")
            iterable = (self.process_item(item) for item in items)
        item_paths = []
        lengths = []
        attr_lengths = {}
        total_duration = 0
        with tqdm.tqdm(iterable, total=len(items), desc=f"Processing {prefix} items") as progress:
            for sample in progress:
                sample: DataSample
                if sample.error:
                    logging.error(
                        f"Error encountered in sample '{sample.name}': {sample.error}",
                        callback=progress.write
                    )
                    continue
                builder.add_item(sample.data)
                item_paths.append(sample.path)
                lengths.append(sample.length)
                for k, v in sample.data.items():
                    if isinstance(v, numpy.ndarray) and v.ndim > 0:
                        if k not in attr_lengths:
                            attr_lengths[k] = []
                        attr_lengths[k].append(v.shape[0])
                duration = sample.length * self.timestep
                total_duration += duration
        builder.finalize()
        metadata = {
            "item_paths": item_paths,
            "lengths": lengths,
            **attr_lengths
        }
        metadata = {
            k: numpy.array(v)
            for k, v in metadata.items()
        }
        with open(self.data_dir / f"{prefix}.info.npz", "wb") as f:
            numpy.savez(f, **metadata)

        logging.info(f"Total duration of {prefix}: {format_duration(total_duration)}.")
        logging.debug(f"Processing {prefix} items done.")

    def process(self):
        # Collect metadata from all subsets
        index_file_paths = list(self.data_dir.rglob("index.csv"))
        metadata_list = []
        for index_file_path in index_file_paths:
            subset_dir = index_file_path.parent
            subset_metadata_list = self.load_metadata(subset_dir)
            metadata_list.extend(subset_metadata_list)
            logging.debug(f"Loaded {len(subset_metadata_list)} metadata items from '{subset_dir.as_posix()}'.")

        # Collect language codes
        lang_set = set()
        for item in metadata_list:
            if item.language:
                lang_set.add(item.language)
        if lang_set:
            logging.info(f"Languages found: {', '.join(sorted(lang_set))}.")
        else:
            logging.warning("No language information found.")
        self.lang_map = {lang: i for i, lang in enumerate(sorted(lang_set), start=1)}

        # Split training and validation sets
        self.split_dataset(metadata_list)
        logging.info(f"Training set total size: {len(self.train_items)}.")
        logging.info(f"Validation set total size: {len(self.valid_items)}.")

        # Copy description files
        with open(self.data_dir / "lang_map.json", "w", encoding="utf8") as f:
            json.dump(self.lang_map, f, ensure_ascii=False)
        with open(self.data_dir / "feature.yaml", "w", encoding="utf8") as f:
            yaml.safe_dump(self.config.features.model_dump(), f, allow_unicode=True, sort_keys=False)

        # Process datasets
        if self.eval_mode:
            self.process_items(self.valid_items, prefix="valid", multiprocessing=True)
        else:
            self.process_items(self.valid_items, prefix="valid", multiprocessing=False)
            self.process_items(self.train_items, prefix="train", multiprocessing=True)


def find_waveform_file(subset_dir: pathlib.Path, item_name: str) -> pathlib.Path:
    for ext in ACCEPTED_AUDIO_FORMATS:
        searched_wav_fn = subset_dir / "waveforms" / f"{item_name}{ext}"
        if searched_wav_fn.exists():
            return searched_wav_fn
    logging.error(
        f"Waveform file missing in raw dataset \'{subset_dir.as_posix()}\': "
        f"item {item_name}, searched extensions: {', '.join(ACCEPTED_AUDIO_FORMATS)}."
    )
    return None


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds to a 'XhYmZs' string."""
    if seconds < 0:
        raise ValueError("Duration cannot be negative.")

    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")

    # Always include seconds, formatted to two decimal places if it is less than 1 minute
    if sec > 0:
        if parts:
            parts.append(f"{int(sec)}s")
        else:
            parts.append(f"{sec:.2f}s")
    elif not parts:  # If total duration is 0
        return "0s"

    return "".join(parts)
