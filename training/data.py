import dataclasses
import math
import pathlib

import librosa
import numpy
import torch

from lib.config.schema import AugmentationConfig
from lib.feature.mel import StretchableMelSpectrogram
from lib.indexed_dataset import IndexedDataset
from .augmentation import *

__all__ = [
    "collate_nd",
    "BaseDataset",
    "DynamicBatchSampler",
]


def collate_nd(values, pad_value=0, max_len=None):
    """
    Pad a list of Nd tensors on their first dimension and stack them into a (N+1)d tensor.
    """
    size = ((max(v.size(0) for v in values) if max_len is None else max_len), *values[0].shape[1:])
    res = torch.full((len(values), *size), fill_value=pad_value, dtype=values[0].dtype, device=values[0].device)

    for i, v in enumerate(values):
        res[i, :len(v), ...] = v
    return res


class BaseDataset(torch.utils.data.Dataset):
    __non_zero_paddings__ = {
        "spectrogram": math.log(1e-5),
    }

    def __init__(
            self,
            data_dir: pathlib.Path,
            prefix: str,
            augmentation_config: AugmentationConfig = None,
            augmentation_deterministic: bool = False,
            augmentation_skip_transforms: bool = False,
            augmentation_return_dirty: bool = False,
    ):
        super().__init__()
        self.info = {
            k: v
            for k, v in numpy.load(data_dir / f"{prefix}.info.npz").items()
        }
        self.data_dir = data_dir
        self.data = IndexedDataset(data_dir, prefix)
        self.epoch = torch.multiprocessing.Value("i", 0)
        self.augmentation_config = augmentation_config
        self.augmentation_deterministic = augmentation_deterministic
        self.augmentation_skip_transforms = augmentation_skip_transforms
        self.augmentation_return_dirty = augmentation_return_dirty
        self.augmentation_args_map: dict[int, AugmentationArgs] = {}
        self.setup = False

    def _setup(self):
        if self.augmentation_config is None:
            return
        self.mel_spectrogram = StretchableMelSpectrogram(
            sample_rate=self.augmentation_config.features.audio_sample_rate,
            n_mels=self.augmentation_config.features.spectrogram.num_bins,
            n_fft=self.augmentation_config.features.fft_size,
            win_length=self.augmentation_config.features.win_size,
            hop_length=self.augmentation_config.features.hop_size,
            fmin=self.augmentation_config.features.spectrogram.fmin,
            fmax=self.augmentation_config.features.spectrogram.fmax,
            clip_val=1e-9,
        ).eval()
        if self.augmentation_deterministic:
            # Pre-generate augmentation args for each sample
            seed = generate_seed(sorted(self.info.keys()))
            generator = numpy.random.default_rng(seed)
            for index in range(len(self)):
                self.augmentation_args_map[index] = generate_augmentation_args(
                    self.augmentation_config, generator=generator,
                    skip_transforms=self.augmentation_skip_transforms,
                )

    def _get_waveform(self, index):
        # Load original waveform
        waveform_fn = self.data_dir / self.info["item_paths"][index]
        waveform, _ = librosa.load(
            waveform_fn, sr=self.augmentation_config.features.audio_sample_rate, mono=True
        )
        return waveform

    def _get_augmented_spectrogram(self, index, spectrogram: torch.Tensor, args: AugmentationArgs) -> torch.Tensor:
        waveform = None
        # RIR reverb (wav -> wav)
        if args.rir_kernel_path is not None:
            if waveform is None:
                waveform = self._get_waveform(index)
            waveform = rir_reverb(
                waveform,
                sr=self.augmentation_config.features.audio_sample_rate,
                kernel_path=args.rir_kernel_path,
            )
        # Colored noise (wav -> wav)
        if args.colored_noise_exponent is not None:
            if waveform is None:
                waveform = self._get_waveform(index)
            waveform = colored_noise(
                waveform,
                exponent=args.colored_noise_exponent,
                factor=args.colored_noise_factor,
                seed=index if self.augmentation_deterministic else None,
            )
        # Natural noise (wav -> wav)
        if args.natural_noise_path is not None:
            if waveform is None:
                waveform = self._get_waveform(index)
            waveform = natural_noise(
                waveform,
                sr=self.augmentation_config.features.audio_sample_rate,
                noise_path=args.natural_noise_path,
                zoom=args.natural_noise_zoom,
                offset=args.natural_noise_offset,
                db=args.natural_noise_db,
            )
        # Pitch shifting (wav -> mel)
        if args.pitch_shift is not None:
            if waveform is None:
                waveform = self._get_waveform(index)
            spectrogram = pitch_shifting(
                waveform,
                self.mel_spectrogram,
                shift=args.pitch_shift,
            )
        elif waveform is not None:
            # Recompute spectrogram if waveform was modified by augmentations above
            spectrogram = self.mel_spectrogram(
                torch.from_numpy(waveform).unsqueeze(0)
            ).squeeze(0).T
        # Loudness scaling (mel -> mel)
        if args.loudness_scale is not None:
            spectrogram = loudness_scaling(
                spectrogram, scale=args.loudness_scale
            )
        # Spectrogram masking (mel -> mel)
        if args.time_mask_width is not None or args.freq_mask_width is not None:
            spectrogram = spectrogram_masking(
                spectrogram,
                time_mask_offset=args.time_mask_offset,
                time_mask_width=args.time_mask_width,
                time_mask_std=args.time_mask_std,
                freq_mask_offset=args.freq_mask_offset,
                freq_mask_width=args.freq_mask_width,
                freq_mask_mean=args.freq_mask_mean,
                freq_mask_std=args.freq_mask_std,
                intersect=args.spec_mask_intersect,
                seed=index if self.augmentation_deterministic else None,
            )
        return spectrogram

    def __getitem__(self, index):
        if not self.setup:
            self._setup()
            self.setup = True
        sample = self.data[index]
        spectrogram = sample["spectrogram"]
        augmentation = {}
        if self.augmentation_config is not None:
            augmentation_args = self.augmentation_args_map.get(index)
            if augmentation_args is None:
                # Must be indeterministic if augmentation args are not pre-generated
                augmentation_args = generate_augmentation_args(
                    self.augmentation_config,
                    skip_transforms=self.augmentation_skip_transforms,
                )
            # Apply augmentations to spectrogram
            spectrogram = self._get_augmented_spectrogram(index, spectrogram, augmentation_args)
            # Convert augmentation args to dict for pickling
            augmentation = dataclasses.asdict(augmentation_args)
        spectrogram = torch.clamp(spectrogram, min=math.log(1e-5))
        if self.augmentation_return_dirty:
            sample["spectrogram"] = torch.clamp(sample["spectrogram"], min=math.log(1e-5))
            sample["spectrogram_dirty"] = spectrogram
        else:
            sample["spectrogram"] = spectrogram
        return {
            "_idx": index,
            "_augmentation": augmentation,
            **sample
        }

    def __len__(self):
        return self.info["lengths"].shape[0]

    def set_epoch(self, epoch: int):
        self.epoch.value = epoch

    def num_frames(self, index: int) -> int:
        return self.info["lengths"][index]

    @classmethod
    def collate(cls, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        for s in samples:
            s.pop("_augmentation", None)
        batch = {
            "size": len(samples),
            "indices": torch.LongTensor([s.pop("_idx") for s in samples]),
        }
        if len(samples) == 0:
            return batch
        for key, value in samples[0].items():
            if value.ndim == 0:
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                pad_value = cls.__non_zero_paddings__.get(key, 0)
                batch[key] = collate_nd([s[key] for s in samples], pad_value=pad_value)
        return batch


class DynamicBatchSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
            self,
            dataset: BaseDataset,
            max_batch_size: int,
            max_batch_frames: int,
            sort_by_len: bool = True,
            frame_count_grid: int = 1,
            batch_count_multiple_of: int = 1,
            reassign_batches: bool = True,
            shuffle_batches: bool = True,
            seed: int = 0,
    ):
        if torch.distributed.is_initialized():
            num_replicas = None
            rank = None
        else:
            num_replicas = 1
            rank = 0
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_batch_frames = max_batch_frames
        self.sort_by_len = sort_by_len
        self.frame_count_grid = frame_count_grid
        self.batch_count_multiple_of = batch_count_multiple_of
        self.reassign_batches = reassign_batches
        self.shuffle_batches = shuffle_batches
        self.generator: torch.Generator = torch.Generator().manual_seed(seed)
        self.batches: list[list[int]] = None
        self.formed = None

    def __iter__(self):
        self.form_batches()
        return iter(self.batches)

    def __len__(self):
        self.form_batches()
        return len(self.batches)

    def set_epoch(self, epoch: int):
        super().set_epoch(epoch)
        self.generator = torch.Generator().manual_seed(self.seed + epoch)

    def permutation(self, n: int) -> list[int]:
        perm = torch.randperm(n, generator=self.generator).tolist()
        return perm

    def form_batches(self):
        if self.formed == self.epoch + self.seed:
            return

        self.dataset.set_epoch(self.epoch)
        lengths = [self.dataset.num_frames(i) for i in range(len(self.dataset))]
        if self.sort_by_len:
            sorted_indices = sorted(
                self.permutation(len(lengths)),
                key=lambda x: lengths[x] // self.frame_count_grid, reverse=True
            )
        else:
            sorted_indices = list(range(len(lengths)))

        def batch_full(batch_: list[int], new_index_: int):
            if len(batch_) >= self.max_batch_size:
                return True
            max_len = max(lengths[new_index_], max((lengths[i] for i in batch_), default=0))
            if max_len * (len(batch_) + 1) > self.max_batch_frames:
                return True
            return False

        batches: list[list[int]] = []

        current_batch = []
        for idx in sorted_indices:
            sample_length = lengths[idx]
            if sample_length > self.max_batch_frames:
                raise ValueError(
                    f"Sample length {sample_length} exceeds max batch frames {self.max_batch_frames}."
                )
            if batch_full(current_batch, idx):
                batches.append(current_batch)
                current_batch = []
            current_batch.append(idx)
        if current_batch:
            batches.append(current_batch)

        multiple_of = self.num_replicas * self.batch_count_multiple_of
        remainder = (multiple_of - (len(batches) % multiple_of)) % multiple_of
        if self.reassign_batches:
            new_batch = []
            while remainder > 0:
                num_batches = len(batches)
                perm = self.permutation(num_batches)
                batches = [batches[i] for i in perm]
                modified = False
                idx = 0
                while remainder > 0 and idx < num_batches:
                    batch = batches[idx]
                    if len(batch) > 1:
                        item = batch[-1]
                        if batch_full(new_batch, item):
                            batches.append(new_batch)
                            new_batch = []
                            modified = True
                            remainder -= 1
                            if remainder == 0:
                                break
                        batch.pop()
                        new_batch.append(item)
                    idx += 1
                if not modified:
                    if len(new_batch) > 0:
                        batches.append(new_batch)
                        new_batch = []
                        remainder -= 1
                    if remainder > 0:
                        raise RuntimeError(
                            f"Unable to reassign batches to meet the required multiple count of {multiple_of}."
                        )
        else:
            batches += [[]] * remainder

        if self.shuffle_batches:
            perm = self.permutation(len(batches))
            batches = [batches[i] for i in perm]
        elif self.sort_by_len:
            batches = sorted(
                batches,
                key=lambda b: len(b) * max(lengths[i] for i in b),
                reverse=True
            )

        batches = [b for i, b in enumerate(batches) if i % self.num_replicas == self.rank]

        self.batches = batches
        self.formed = self.epoch + self.seed
