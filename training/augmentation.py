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
from lib.feature.mel import StretchableMelSpectrogram

__all__ = [
    "AugmentationArgs",
    "generate_seed",
    "generate_augmentation_args",
    "colored_noise",
    "natural_noise",
    "rir_reverb",
    "pitch_shifting",
    "loudness_scaling",
    "spectrogram_masking",
]


@dataclass
class AugmentationArgs:
    colored_noise_exponent: float = None
    colored_noise_factor: float = None
    natural_noise_path: str = None
    natural_noise_zoom: float = None
    natural_noise_offset: float = None
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


def generate_seed(strings: list[str]) -> int:
    text = "|".join(strings)
    hash_obj = hashlib.sha256(text.encode("utf8"))
    hex_digest = hash_obj.hexdigest()
    return int(hex_digest[:8], 16)


def generate_augmentation_args(
        config: AugmentationConfig, generator: np.random.Generator = None,
        skip_transforms: bool = False
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
        args.natural_noise_path = generator.choice(config.natural_noise.noise_file_list)
        args.natural_noise_zoom = 2 ** generator.uniform(-1, 1)
        args.natural_noise_offset = generator.uniform(0, 1)
        args.natural_noise_db = generator.uniform(-24, -6)
    if (
        config.rir_reverb.enabled
        and generator.random() < config.rir_reverb.prob
    ):
        args.rir_kernel_path = generator.choice(config.rir_reverb.kernel_file_list)
    if (
            not skip_transforms
            and config.pitch_shifting.enabled
            and generator.random() < config.pitch_shifting.prob
    ):
        args.pitch_shift = generator.uniform(
            config.pitch_shifting.min_semitones,
            config.pitch_shifting.max_semitones,
        )
    if (
            not skip_transforms
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


def colored_noise(
        waveform: np.ndarray,
        exponent: float,
        factor: float,
        seed: int = None,
):
    """
    wav -> wav
    """
    generator = np.random.default_rng(seed)
    # noinspection PyUnresolvedReferences
    noise = colorednoise.powerlaw_psd_gaussian(
        exponent, size=len(waveform), random_state=generator
    ).astype(np.float32)
    return waveform + noise * (10 ** factor)


def natural_noise(
        waveform: np.ndarray,
        sr: int,
        noise_path: str,
        zoom: float,
        offset: float,
        db: float,
):
    """
    wav -> wav
    """
    reinterpreted_sr = round(sr * zoom)
    noise, _ = librosa.load(noise_path, sr=reinterpreted_sr, mono=True)
    min_offset = -len(noise)
    max_offset = len(waveform)
    offset = int(offset * (max_offset - min_offset) + min_offset)
    if offset < 0:
        noise = noise[-offset:]
    elif offset > 0:
        noise = np.pad(noise, (offset, 0), mode="constant")
    if len(noise) < len(waveform):
        noise = np.pad(noise, (0, len(waveform) - len(noise)), mode="constant")
    elif len(noise) > len(waveform):
        noise = noise[:len(waveform)]
    scale = np.abs(waveform).max() / (np.abs(noise).max() + 1e-8)
    waveform_noisy = waveform + noise * scale * (10 ** (db / 20))
    return waveform_noisy.astype(np.float32)


def rir_reverb(
        waveform: np.ndarray,
        sr: int,
        kernel_path: str,
):
    """
    wav -> wav
    """
    rir, _ = librosa.load(kernel_path, sr=sr, mono=True)
    convolved = scipy.signal.fftconvolve(waveform, rir, mode="full").astype(np.float32)
    rir_max = np.abs(rir).argmax()
    convolved = convolved[rir_max: rir_max + len(waveform)]
    scale = np.abs(waveform).max() / (np.abs(convolved).max() + 1e-8)
    convolved = convolved * scale
    return convolved.astype(np.float32)


def pitch_shifting(
        waveform: np.ndarray,
        mel_spec_module: StretchableMelSpectrogram,
        shift: float,
) -> Tensor:
    """
    wav -> mel
    """
    shifted_spectrogram = mel_spec_module(
        torch.from_numpy(waveform).unsqueeze(0), key_shift=shift
    ).squeeze(0).T
    return shifted_spectrogram


def loudness_scaling(
        spectrogram: Tensor,
        scale: float
) -> Tensor:
    """
    mel -> mel
    """
    return spectrogram + (scale / 20.0) * math.log(10)


def spectrogram_masking(
        spectrogram: Tensor,
        time_mask_offset: float,
        time_mask_width: int,
        time_mask_std: float,
        freq_mask_offset: int,
        freq_mask_width: int,
        freq_mask_mean: float,
        freq_mask_std: float,
        intersect: bool,
        seed: int = None,
):
    """
    mel -> mel
    """
    generator = np.random.default_rng(seed)
    T, C = spectrogram.shape
    spec = spectrogram.cpu().numpy()
    time_masked = time_mask_width is not None
    freq_masked = freq_mask_width is not None
    if time_masked:
        time_mask_width = min(time_mask_width, T)
        time_offset_min = 0
        time_offset_max = T - time_mask_width
        time_mask_start = int(time_mask_offset * (time_offset_max - time_offset_min) + time_offset_min)
        time_mask_end = time_mask_start + time_mask_width
    else:
        time_mask_start = time_mask_end = None
    if freq_masked:
        freq_mask_start = freq_mask_offset
        freq_mask_end = freq_mask_offset + freq_mask_width
    else:
        freq_mask_start = freq_mask_end = None
    if time_masked and freq_masked and intersect:
        spec[time_mask_start: time_mask_end, freq_mask_start: freq_mask_end] = generator.standard_normal(
            size=(time_mask_width, freq_mask_width), dtype=np.float32
        ) * freq_mask_std + freq_mask_mean
    else:
        if time_masked:
            spec[time_mask_start: time_mask_end, :] = generator.standard_normal(
                size=(time_mask_width, C), dtype=np.float32
            ) * time_mask_std
        if freq_masked:
            spec[:, freq_mask_start: freq_mask_end] = generator.standard_normal(
                size=(T, freq_mask_width), dtype=np.float32
            ) * freq_mask_std + freq_mask_mean
    masked_spectrogram = torch.from_numpy(spec).to(spectrogram.device)
    return masked_spectrogram
