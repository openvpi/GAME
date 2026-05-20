import hashlib
import math
from dataclasses import dataclass
from typing import ClassVar

import colorednoise
import librosa
import numpy as np
import scipy.signal
import torch
from pydantic import BaseModel, Field
from torch import Tensor

from lib.config.schema import AugmentationConfig

__all__ = [
    "generate_seed",
    "build_augmentation_chain",
    "Augmentation",
    "ComposedAugmentation",
    "ColoredNoise",
    "NaturalNoise",
    "RIRReverb",
    "LoudnessScaling",
    "SpectrogramMasking",
    "WaveformAugmentationContext",
    "SpectrogramAugmentationContext",
]


def generate_seed(strings: list[str]) -> int:
    text = "|".join(strings)
    hash_obj = hashlib.sha256(text.encode("utf8"))
    return int(hash_obj.hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Context dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WaveformAugmentationContext:
    waveform: np.ndarray
    sr: int


@dataclass
class SpectrogramAugmentationContext:
    spectrogram: Tensor


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class Augmentation(BaseModel):
    """Base class for a single augmentation step."""

    _namespace: ClassVar[str] = ""

    def should_apply(self) -> bool:
        raise NotImplementedError

    def apply(self, ctx) -> None:
        raise NotImplementedError

    def args_dict(self) -> dict:
        data = self.model_dump(exclude_none=True)
        if not data:
            return {}
        return {self._namespace: data}


class ComposedAugmentation(Augmentation):
    """Chain of augmentations applied in sequence."""

    transforms: list[Augmentation] = Field(default_factory=list)

    def should_apply(self) -> bool:
        return bool(self.transforms)

    def apply(self, ctx) -> None:
        for t in self.transforms:
            t.apply(ctx)

    def args_dict(self) -> dict:
        result = {}
        for t in self.transforms:
            result.update(t.args_dict())
        return result


# ---------------------------------------------------------------------------
# Waveform-domain transforms (wav -> wav)
# ---------------------------------------------------------------------------

class ColoredNoise(Augmentation):
    _namespace: ClassVar[str] = "colored_noise"

    exponent: float = None
    factor: float = None
    seed: int | None = None

    def __init__(self, config: AugmentationConfig, generator: np.random.Generator, **kwargs):
        super().__init__(**kwargs)
        if (
                config.colored_noise.enabled
                and generator.random() < config.colored_noise.prob
        ):
            self.exponent = generator.uniform(
                config.colored_noise.min_exponent,
                config.colored_noise.max_exponent,
            )
            self.factor = generator.uniform(-6, -1)
            self.seed = int(generator.integers(0, 2 ** 31))

    def should_apply(self) -> bool:
        return self.exponent is not None

    def apply(self, ctx: WaveformAugmentationContext) -> None:
        rng = np.random.default_rng(self.seed)
        noise = colorednoise.powerlaw_psd_gaussian(
            self.exponent, size=len(ctx.waveform), random_state=rng,
        ).astype(np.float32)
        ctx.waveform = ctx.waveform + noise * (10 ** self.factor)


class NaturalNoise(Augmentation):
    _namespace: ClassVar[str] = "natural_noise"

    class _Item(BaseModel):
        path: str
        zoom: float
        offset: float
        scale: float

    items: list[_Item] = None
    db: float = None

    def __init__(self, config: AugmentationConfig, generator: np.random.Generator, **kwargs):
        super().__init__(**kwargs)
        if (
                config.natural_noise.enabled
                and generator.random() < config.natural_noise.prob
        ):
            items = []
            repeats = generator.integers(1, config.natural_noise.max_repeats + 1)
            for _ in range(repeats):
                items.append(NaturalNoise._Item(
                    path=generator.choice(config.natural_noise.noise_file_list),
                    zoom=2 ** generator.uniform(-1, 1),
                    offset=generator.uniform(0, 1),
                    scale=10 ** (generator.uniform(-12, 12) / 20),
                ))
            self.items = items
            self.db = generator.uniform(-24, -6)

    def should_apply(self) -> bool:
        return self.items is not None

    def apply(self, ctx: WaveformAugmentationContext) -> None:
        total_noise = np.zeros_like(ctx.waveform)
        for item in self.items:
            reinterpreted_sr = round(ctx.sr * item.zoom)
            noise, _ = librosa.load(item.path, sr=reinterpreted_sr, mono=True)
            offset_range = len(ctx.waveform) + len(noise)
            offset = int(item.offset * offset_range - len(noise))
            if offset < 0:
                noise = noise[-offset:]
            elif offset > 0:
                noise = np.pad(noise, (offset, 0), mode="constant")
            if len(noise) < len(ctx.waveform):
                noise = np.pad(noise, (0, len(ctx.waveform) - len(noise)), mode="constant")
            elif len(noise) > len(ctx.waveform):
                noise = noise[:len(ctx.waveform)]
            noise = noise * item.scale
            total_noise += noise
        scale = np.abs(ctx.waveform).max() / (np.abs(total_noise).max() + 1e-8)
        ctx.waveform = (
                ctx.waveform + total_noise * scale * (10 ** (self.db / 20))
        ).astype(np.float32)


class RIRReverb(Augmentation):
    _namespace: ClassVar[str] = "rir_reverb"

    kernel_path: str = None

    def __init__(self, config: AugmentationConfig, generator: np.random.Generator, **kwargs):
        super().__init__(**kwargs)
        if (
                config.rir_reverb.enabled
                and generator.random() < config.rir_reverb.prob
        ):
            self.kernel_path = generator.choice(config.rir_reverb.kernel_file_list)

    def should_apply(self) -> bool:
        return self.kernel_path is not None

    def apply(self, ctx: WaveformAugmentationContext) -> None:
        rir, _ = librosa.load(self.kernel_path, sr=ctx.sr, mono=True)
        convolved = scipy.signal.fftconvolve(
            ctx.waveform, rir, mode="full",
        ).astype(np.float32)
        rir_max = np.abs(rir).argmax()
        convolved = convolved[rir_max:rir_max + len(ctx.waveform)]
        scale = np.abs(ctx.waveform).max() / (np.abs(convolved).max() + 1e-8)
        ctx.waveform = (convolved * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrogram-domain transforms (mel -> mel)
# ---------------------------------------------------------------------------

class LoudnessScaling(Augmentation):
    _namespace: ClassVar[str] = "loudness_scaling"

    scale: float = None

    def __init__(self, config: AugmentationConfig, generator: np.random.Generator, **kwargs):
        super().__init__(**kwargs)
        if (
                config.loudness_scaling.enabled
                and generator.random() < config.loudness_scaling.prob
        ):
            self.scale = generator.uniform(
                config.loudness_scaling.min_db,
                config.loudness_scaling.max_db,
            )

    def should_apply(self) -> bool:
        return self.scale is not None

    def apply(self, ctx: SpectrogramAugmentationContext) -> None:
        ctx.spectrogram = ctx.spectrogram + (self.scale / 20.0) * math.log(10)


class SpectrogramMasking(Augmentation):
    _namespace: ClassVar[str] = "spectrogram_masking"

    time_mask_offset: float = None
    time_mask_width: int = None
    time_mask_std: float = None
    freq_mask_offset: int = None
    freq_mask_width: int = None
    freq_mask_mean: float = None
    freq_mask_std: float = None
    intersect: bool = None
    seed: int | None = None

    def __init__(self, config: AugmentationConfig, generator: np.random.Generator, **kwargs):
        super().__init__(**kwargs)
        if config.spectrogram_masking.enabled:
            time_masked = generator.random() < config.spectrogram_masking.time_mask_prob
            freq_masked = generator.random() < config.spectrogram_masking.freq_mask_prob
            if time_masked:
                self.time_mask_offset = generator.uniform(0, 1)
                self.time_mask_width = int(generator.integers(
                    1, config.spectrogram_masking.time_mask_max_width + 1,
                ))
                self.time_mask_std = generator.uniform(0, 1)
            if freq_masked:
                self.freq_mask_width = int(generator.integers(
                    1, config.spectrogram_masking.freq_mask_max_width + 1,
                ))
                self.freq_mask_offset = int(generator.integers(
                    0, config.features.spectrogram.num_bins - self.freq_mask_width + 1,
                ))
                self.freq_mask_mean = generator.uniform(math.log(1e-5), 0)
                self.freq_mask_std = generator.uniform(0, 1)
            if time_masked and freq_masked and generator.random() < config.spectrogram_masking.intersect_prob:
                self.intersect = True
            if self.time_mask_width is not None or self.freq_mask_width is not None:
                self.seed = int(generator.integers(0, 2 ** 31))

    def should_apply(self) -> bool:
        return self.time_mask_width is not None or self.freq_mask_width is not None

    def apply(self, ctx: SpectrogramAugmentationContext) -> None:
        rng = np.random.default_rng(self.seed)
        T, C = ctx.spectrogram.shape
        spec = ctx.spectrogram.cpu().numpy()
        time_masked = self.time_mask_width is not None
        freq_masked = self.freq_mask_width is not None
        time_mask_start = time_mask_end = time_mask_width = None
        freq_mask_start = freq_mask_end = None
        if time_masked:
            time_mask_width = min(self.time_mask_width, T)
            time_mask_start = int(self.time_mask_offset * (T - time_mask_width))
            time_mask_end = time_mask_start + time_mask_width
        if freq_masked:
            freq_mask_start = self.freq_mask_offset
            freq_mask_end = self.freq_mask_offset + self.freq_mask_width
        if time_masked and freq_masked and self.intersect:
            spec[time_mask_start:time_mask_end, freq_mask_start:freq_mask_end] = (
                    rng.standard_normal(
                        size=(time_mask_width, self.freq_mask_width), dtype=np.float32,
                    ) * self.freq_mask_std + self.freq_mask_mean
            )
        else:
            if time_masked:
                spec[time_mask_start:time_mask_end, :] = (
                        rng.standard_normal(
                            size=(time_mask_width, C), dtype=np.float32,
                        ) * self.time_mask_std
                )
            if freq_masked:
                spec[:, freq_mask_start:freq_mask_end] = (
                        rng.standard_normal(
                            size=(T, self.freq_mask_width), dtype=np.float32,
                        ) * self.freq_mask_std + self.freq_mask_mean
                )
        ctx.spectrogram = torch.from_numpy(spec).to(ctx.spectrogram.device)


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def build_augmentation_chain(
        config: AugmentationConfig,
        generator: np.random.Generator,
        destructive_only: bool = False,
) -> tuple[ComposedAugmentation, float | None, ComposedAugmentation]:
    """Build (wav_chain, pitch_shift, spec_chain) for a single sample."""
    wav_transforms: list[Augmentation] = []
    for aug in [
        ColoredNoise(config=config, generator=generator),
        NaturalNoise(config=config, generator=generator),
        RIRReverb(config=config, generator=generator),
    ]:
        if aug.should_apply():
            wav_transforms.append(aug)

    pitch_shift: float | None = None
    if not destructive_only:
        if (
                config.pitch_shifting.enabled
                and generator.random() < config.pitch_shifting.prob
        ):
            pitch_shift = float(generator.uniform(
                config.pitch_shifting.min_semitones,
                config.pitch_shifting.max_semitones,
            ))

    spec_transforms: list[Augmentation] = []
    for aug in [
        LoudnessScaling(config=config, generator=generator),
        SpectrogramMasking(config=config, generator=generator),
    ]:
        if aug.should_apply():
            spec_transforms.append(aug)

    return (
        ComposedAugmentation(transforms=wav_transforms),
        pitch_shift,
        ComposedAugmentation(transforms=spec_transforms),
    )
