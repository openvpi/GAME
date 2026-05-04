import abc
import hashlib
import math
from dataclasses import dataclass

import colorednoise
import librosa
import numpy as np
import scipy.signal
import torch
from torch import Tensor

from lib.config.schema import AugmentationConfig

__all__ = [
    "AugmentationArgs",
    "generate_seed",
    "generate_augmentation_args",
    "Augmentation",
    "Compose",
    "ColoredNoise",
    "NaturalNoise",
    "RIRReverb",
    "LoudnessScaling",
    "SpectrogramMasking",
    "WaveformAugmentationContext",
    "SpectrogramAugmentationContext",
    "build_wav_transforms",
    "build_spec_transforms",
]


@dataclass
class _NaturalNoiseArgs:
    path: str
    zoom: float
    offset: float
    scale: float


@dataclass
class AugmentationArgs:
    colored_noise_exponent: float = None
    colored_noise_factor: float = None
    natural_noise_args: list[_NaturalNoiseArgs] = None
    natural_noise_db: float = None
    rir_kernel_path: str = None
    pitch_shift: float = None
    loudness_scale: float = None
    time_mask_offset: float = None
    time_mask_width: int = None
    time_mask_std: float = None
    freq_mask_offset: int = None
    freq_mask_width: int = None
    freq_mask_mean: float = None
    freq_mask_std: float = None
    spec_mask_intersect: bool = None


@dataclass
class WaveformAugmentationContext:
    waveform: np.ndarray
    sr: int
    args: AugmentationArgs
    deterministic: bool
    index: int


@dataclass
class SpectrogramAugmentationContext:
    spectrogram: Tensor
    args: AugmentationArgs
    deterministic: bool
    index: int


def generate_seed(strings: list[str]) -> int:
    text = "|".join(strings)
    hash_obj = hashlib.sha256(text.encode("utf8"))
    hex_digest = hash_obj.hexdigest()
    return int(hex_digest[:8], 16)


def generate_augmentation_args(
        config: AugmentationConfig, generator: np.random.Generator = None,
        destructive_only: bool = False,
) -> AugmentationArgs:
    if generator is None:
        generator = np.random.default_rng()

    args = AugmentationArgs()

    if (
            config.colored_noise.enabled
            and generator.random() < config.colored_noise.prob
    ):
        args.colored_noise_exponent = generator.uniform(
            config.colored_noise.min_exponent,
            config.colored_noise.max_exponent,
        )
        args.colored_noise_factor = generator.uniform(-6, -1)

    if (
            config.natural_noise.enabled
            and generator.random() < config.natural_noise.prob
    ):
        natural_noise_args_list = []
        repeats = generator.integers(1, config.natural_noise.max_repeats + 1)
        for _ in range(repeats):
            noise_path = generator.choice(config.natural_noise.noise_file_list)
            noise_zoom = 2 ** generator.uniform(-1, 1)
            noise_offset = generator.uniform(0, 1)
            noise_scale = 10 ** (generator.uniform(-12, 12) / 20)
            noise_args = _NaturalNoiseArgs(
                path=noise_path,
                zoom=noise_zoom,
                offset=noise_offset,
                scale=noise_scale,
            )
            natural_noise_args_list.append(noise_args)
        args.natural_noise_args = natural_noise_args_list
        args.natural_noise_db = generator.uniform(-24, -6)

    if (
            config.rir_reverb.enabled
            and generator.random() < config.rir_reverb.prob
    ):
        args.rir_kernel_path = generator.choice(config.rir_reverb.kernel_file_list)

    if (
            not destructive_only
            and config.pitch_shifting.enabled
            and generator.random() < config.pitch_shifting.prob
    ):
        args.pitch_shift = generator.uniform(
            config.pitch_shifting.min_semitones,
            config.pitch_shifting.max_semitones,
        )

    if (
            not destructive_only
            and config.loudness_scaling.enabled
            and generator.random() < config.loudness_scaling.prob
    ):
        args.loudness_scale = generator.uniform(
            config.loudness_scaling.min_db,
            config.loudness_scaling.max_db,
        )

    if config.spectrogram_masking.enabled:
        time_masked = generator.random() < config.spectrogram_masking.time_mask_prob
        freq_masked = generator.random() < config.spectrogram_masking.freq_mask_prob
        if time_masked:
            args.time_mask_offset = generator.uniform(0, 1)
            args.time_mask_width = generator.integers(
                1,
                config.spectrogram_masking.time_mask_max_width + 1,
            )
            args.time_mask_std = generator.uniform(0, 1)
        if freq_masked:
            args.freq_mask_width = generator.integers(
                1,
                config.spectrogram_masking.freq_mask_max_width + 1,
            )
            args.freq_mask_offset = generator.integers(
                0,
                config.features.spectrogram.num_bins - args.freq_mask_width + 1,
            )
            args.freq_mask_mean = generator.uniform(math.log(1e-5), 0)
            args.freq_mask_std = generator.uniform(0, 1)
        if time_masked and freq_masked and generator.random() < config.spectrogram_masking.intersect_prob:
            args.spec_mask_intersect = True
    return args


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class Augmentation(abc.ABC):
    """Base class for a single augmentation step."""

    @abc.abstractmethod
    def should_apply(self, ctx) -> bool: ...

    @abc.abstractmethod
    def apply(self, ctx) -> None: ...

    def __call__(self, ctx):
        if self.should_apply(ctx):
            self.apply(ctx)
        return ctx


class Compose:
    """Runs a sequence of augmentations in order."""

    def __init__(self, transforms: list[Augmentation]):
        self.transforms = transforms

    def __call__(self, ctx):
        for t in self.transforms:
            t(ctx)
        return ctx


# ---------------------------------------------------------------------------
# Waveform-domain transforms (wav -> wav)
# ---------------------------------------------------------------------------

class ColoredNoise(Augmentation):
    def should_apply(self, ctx: WaveformAugmentationContext) -> bool:
        return ctx.args.colored_noise_exponent is not None

    def apply(self, ctx: WaveformAugmentationContext) -> None:
        seed = ctx.index if ctx.deterministic else None
        generator = np.random.default_rng(seed)
        noise = colorednoise.powerlaw_psd_gaussian(
            ctx.args.colored_noise_exponent,
            size=len(ctx.waveform),
            random_state=generator,
        ).astype(np.float32)
        ctx.waveform = ctx.waveform + noise * (10 ** ctx.args.colored_noise_factor)


class NaturalNoise(Augmentation):
    def should_apply(self, ctx: WaveformAugmentationContext) -> bool:
        return ctx.args.natural_noise_args is not None

    def apply(self, ctx: WaveformAugmentationContext) -> None:
        total_noise = np.zeros_like(ctx.waveform)
        for noise_args in ctx.args.natural_noise_args:
            reinterpreted_sr = round(ctx.sr * noise_args.zoom)
            noise, _ = librosa.load(noise_args.path, sr=reinterpreted_sr, mono=True)
            min_offset = -len(noise)
            max_offset = len(ctx.waveform)
            offset = int(noise_args.offset * (max_offset - min_offset) + min_offset)
            if offset < 0:
                noise = noise[-offset:]
            elif offset > 0:
                noise = np.pad(noise, (offset, 0), mode="constant")
            if len(noise) < len(ctx.waveform):
                noise = np.pad(noise, (0, len(ctx.waveform) - len(noise)), mode="constant")
            elif len(noise) > len(ctx.waveform):
                noise = noise[:len(ctx.waveform)]
            noise = noise * noise_args.scale
            total_noise += noise
        scale = np.abs(ctx.waveform).max() / (np.abs(total_noise).max() + 1e-8)
        ctx.waveform = (
            ctx.waveform + total_noise * scale * (10 ** (ctx.args.natural_noise_db / 20))
        ).astype(np.float32)


class RIRReverb(Augmentation):
    def should_apply(self, ctx: WaveformAugmentationContext) -> bool:
        return ctx.args.rir_kernel_path is not None

    def apply(self, ctx: WaveformAugmentationContext) -> None:
        rir, _ = librosa.load(ctx.args.rir_kernel_path, sr=ctx.sr, mono=True)
        convolved = scipy.signal.fftconvolve(
            ctx.waveform, rir, mode="full"
        ).astype(np.float32)
        rir_max = np.abs(rir).argmax()
        convolved = convolved[rir_max: rir_max + len(ctx.waveform)]
        scale = np.abs(ctx.waveform).max() / (np.abs(convolved).max() + 1e-8)
        ctx.waveform = (convolved * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrogram-domain transforms (mel -> mel)
# ---------------------------------------------------------------------------

class LoudnessScaling(Augmentation):
    def should_apply(self, ctx: SpectrogramAugmentationContext) -> bool:
        return ctx.args.loudness_scale is not None

    def apply(self, ctx: SpectrogramAugmentationContext) -> None:
        ctx.spectrogram = ctx.spectrogram + (ctx.args.loudness_scale / 20.0) * math.log(10)


class SpectrogramMasking(Augmentation):
    def should_apply(self, ctx: SpectrogramAugmentationContext) -> bool:
        return ctx.args.time_mask_width is not None or ctx.args.freq_mask_width is not None

    def apply(self, ctx: SpectrogramAugmentationContext) -> None:
        args = ctx.args
        seed = ctx.index if ctx.deterministic else None
        generator = np.random.default_rng(seed)
        T, C = ctx.spectrogram.shape
        spec = ctx.spectrogram.cpu().numpy()
        time_masked = args.time_mask_width is not None
        freq_masked = args.freq_mask_width is not None
        if time_masked:
            time_mask_width = min(args.time_mask_width, T)
            time_offset_max = T - time_mask_width
            time_mask_start = int(args.time_mask_offset * time_offset_max)
            time_mask_end = time_mask_start + time_mask_width
        else:
            time_mask_start = time_mask_end = None
            time_mask_width = None
        if freq_masked:
            freq_mask_start = args.freq_mask_offset
            freq_mask_end = args.freq_mask_offset + args.freq_mask_width
        else:
            freq_mask_start = freq_mask_end = None
        if time_masked and freq_masked and args.spec_mask_intersect:
            spec[time_mask_start:time_mask_end, freq_mask_start:freq_mask_end] = (
                generator.standard_normal(
                    size=(time_mask_width, args.freq_mask_width), dtype=np.float32
                ) * args.freq_mask_std + args.freq_mask_mean
            )
        else:
            if time_masked:
                spec[time_mask_start:time_mask_end, :] = (
                    generator.standard_normal(
                        size=(time_mask_width, C), dtype=np.float32
                    ) * args.time_mask_std
                )
            if freq_masked:
                spec[:, freq_mask_start:freq_mask_end] = (
                    generator.standard_normal(
                        size=(T, args.freq_mask_width), dtype=np.float32
                    ) * args.freq_mask_std + args.freq_mask_mean
                )
        ctx.spectrogram = torch.from_numpy(spec).to(ctx.spectrogram.device)


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def build_wav_transforms(config: AugmentationConfig) -> Compose:
    transforms = []
    if config.rir_reverb.enabled:
        transforms.append(RIRReverb())
    if config.colored_noise.enabled:
        transforms.append(ColoredNoise())
    if config.natural_noise.enabled:
        transforms.append(NaturalNoise())
    return Compose(transforms)


def build_spec_transforms(
        config: AugmentationConfig,
        destructive_only: bool = False,
) -> Compose:
    transforms = []
    if not destructive_only and config.loudness_scaling.enabled:
        transforms.append(LoudnessScaling())
    if config.spectrogram_masking.enabled:
        transforms.append(SpectrogramMasking())
    return Compose(transforms)
