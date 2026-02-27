import torch
from torch import nn, Tensor

from lib.config.schema import ModelConfig, InferenceConfig
from lib.feature.mel import StretchableMelSpectrogram
from modules.d3pm import (
    d3pm_time_schedule,
    remove_mutable_boundaries,
)
from modules.decoding import (
    decode_soft_boundaries,
    decode_gaussian_blurred_probs,
)
from modules.functional import (
    format_boundaries,
    boundaries_to_regions,
    regions_to_durations,
)
from modules.midi_extraction import SegmentationEstimationModel


class SegmentationEstimationInferenceModel(nn.Module):
    def __init__(
            self,
            model_config: ModelConfig,
            inference_config: InferenceConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.inference_config = inference_config
        self.timestep = self.inference_config.features.timestep
        self.to_spectrogram = StretchableMelSpectrogram(
            sample_rate=inference_config.features.audio_sample_rate,
            n_mels=inference_config.features.spectrogram.num_bins,
            n_fft=inference_config.features.fft_size,
            win_length=inference_config.features.win_size,
            hop_length=inference_config.features.hop_size,
            fmin=inference_config.features.spectrogram.fmin,
            fmax=inference_config.features.spectrogram.fmax,
            clip_val=1e-5,
        )
        self.model = SegmentationEstimationModel(model_config)

    def _forward_and_decode_boundaries(
            self, x_seg, noise, mask,
            barriers, threshold, radius,
            language=None, t=None,
    ):
        logits, _ = self.model.forward_segmentation(
            x_seg, noise=noise, t=t,
            language=language, mask=mask,
        )  # [B, T]
        soft_boundaries = logits.sigmoid()
        boundaries = decode_soft_boundaries(
            boundaries=soft_boundaries,
            barriers=barriers, mask=mask,
            threshold=threshold, radius=radius,
        )  # [B, T]
        return boundaries

    def _forward_and_decode_scores(
            self, x_est, regions, max_n, t_mask, n_mask, threshold
    ):
        logits = self.model.forward_estimation(
            x_est, regions=regions, max_n=max_n,
            t_mask=t_mask, n_mask=n_mask,
        )  # [B, N, C_out]
        probs = logits.sigmoid()
        scores, presence = decode_gaussian_blurred_probs(
            probs=probs,
            min_val=self.inference_config.midi_min,
            max_val=self.inference_config.midi_max,
            deviation=self.inference_config.midi_std * 3,  # use 3 std as the decoding deviation
            threshold=threshold,
        )
        presence = presence & n_mask
        scores = scores * presence.float()
        return presence, scores

    def forward(
            self, waveform: Tensor,
            known_durations: Tensor,
            boundary_threshold: Tensor,
            boundary_radius: Tensor,
            score_threshold: Tensor,
            language: Tensor = None,
            t: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param waveform: float32 [batch_size, num_samples]
        :param known_durations: float32 [batch_size, num_known_regions]
        :param boundary_threshold: float32 scalar
        :param boundary_radius: float32 scalar
        :param score_threshold: float32 scalar
        :param language: int64 [batch_size]
        :param t: float32 [num_steps]
        :return: durations: float32; presence: bool; scores: float32 [batch_size, num_notes]
        """
        spectrogram = self.to_spectrogram(waveform).mT  # [B, T, C]
        B = waveform.size(0)
        T = spectrogram.size(1)
        Nt = t.size(0)
        known_boundaries, t_mask = format_boundaries(
            durations=known_durations, length=T, timestep=self.timestep
        )  # [B, T]
        boundary_radius = (boundary_radius / self.timestep).round().long().clamp(min=1)

        # Encoder
        x_seg, x_est = self.model.forward_encoder(spectrogram, mask=t_mask)

        # Segmentation
        if self.model_config.mode == "d3pm":
            boundaries = known_boundaries
            for i in range(Nt):
                ti = torch.full((B,), fill_value=t[i], device=waveform.device)
                p = d3pm_time_schedule(ti)
                boundaries_noise = remove_mutable_boundaries(boundaries, known_boundaries, p=p)
                regions_noise = boundaries_to_regions(boundaries_noise, mask=t_mask)  # [B, T]
                boundaries = self._forward_and_decode_boundaries(
                    x_seg, noise=regions_noise, t=ti,
                    language=language, mask=t_mask,
                    barriers=known_boundaries,
                    threshold=boundary_threshold,
                    radius=boundary_radius,
                )  # [B, T]
        elif self.model_config.mode == "completion":
            known_regions = boundaries_to_regions(known_boundaries, mask=t_mask)  # [B, T]
            boundaries = self._forward_and_decode_boundaries(
                x_seg, noise=known_regions,
                language=language, mask=t_mask,
                barriers=known_boundaries,
                threshold=boundary_threshold,
                radius=boundary_radius,
            )  # [B, T]
        else:
            raise ValueError(f"Unknown mode: {self.model_config.mode}.")
        regions = boundaries_to_regions(boundaries, mask=t_mask)  # [B, T]
        max_n = regions.max()
        durations = regions_to_durations(regions, max_n=max_n) * self.timestep  # [B, N]

        # Estimation
        idx = torch.arange(max_n, dtype=torch.long, device=regions.device).unsqueeze(0)  # [1, N]
        max_idx = regions.amax(dim=-1, keepdim=True)  # [B, 1]
        n_mask = idx < max_idx  # [B, N]
        presence, scores = self._forward_and_decode_scores(
            x_est, regions=regions, max_n=max_n,
            t_mask=t_mask, n_mask=n_mask,
            threshold=score_threshold,
        )

        return durations, presence, scores
