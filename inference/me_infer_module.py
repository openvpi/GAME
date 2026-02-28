import librosa
import lightning.pytorch
import torch
import torchmetrics
from torch import nn, Tensor

from inference.me_infer import SegmentationEstimationInferenceModel
from lib.config.schema import ValidationConfig
from modules.functional import flatten_sequences, regions_to_durations
from modules.metrics import (
    AverageChamferDistance,
    QuantityMetricCollection,
    NotePresenceMetricCollection,
    RawPitchRMSE,
    RawPitchAccuracy,
    OverallAccuracy,
    NoteOverlapMetricCollection,
)


class InferenceModule(lightning.pytorch.LightningModule):
    def __init__(
            self,
            model: SegmentationEstimationInferenceModel,
            config: ValidationConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.timestep = self.model.timestep
        self.metrics: dict[str, torchmetrics.Metric] | None = None

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if stage == "fit" or stage == "validate":
            raise ValueError(f"InferenceModule does not support stage '{stage}'.")
        if stage == "test":
            self.metrics = nn.ModuleDict({
                "average_chamfer_distance": AverageChamferDistance(),
                "quantity_metric_collection": QuantityMetricCollection(
                    tolerance=self.config.boundary_matching_tolerance,
                ),
                "presence_metric_collection": NotePresenceMetricCollection(),
                "raw_pitch_rmse": RawPitchRMSE(),
                "raw_pitch_accuracy": RawPitchAccuracy(
                    tolerance=self.config.pitch_accuracy_tolerance,
                ),
                "overall_accuracy": OverallAccuracy(
                    tolerance=self.config.pitch_accuracy_tolerance,
                ),
                "overlap_metric_collection": NoteOverlapMetricCollection(
                    pitch_width=self.config.pitch_overlap_width,
                ),
            })

    def predict_step(self, batch: dict[str, Tensor], *args, **kwargs) -> dict[str, Tensor]:
        waveform = batch["waveform"]
        samplerate = batch["samplerate"]
        if samplerate != (model_sr := self.model.inference_config.features.audio_sample_rate):
            waveform = torch.stack([
                librosa.resample(w.cpu().numpy(), orig_sr=samplerate, target_sr=model_sr)
                for w in waveform.unbind(dim=0)
            ], dim=0).to(waveform)
        known_durations = batch["known_durations"]
        language = batch["language"]
        durations, presence, scores = self.model(
            waveform=waveform,
            known_durations=known_durations,
            language=language,
            t=torch.tensor(self.config.d3pm_sample_ts_resolved).to(waveform),
            boundary_threshold=torch.tensor(self.config.boundary_decoding_threshold).to(waveform),
            boundary_radius=torch.tensor(self.config.boundary_decoding_radius).to(language),
            score_threshold=torch.tensor(self.config.note_presence_threshold).to(waveform),
        )
        return {
            "durations": durations,
            "presence": presence,
            "scores": scores,
        }

    def test_step(self, batch: dict[str, Tensor], *args, **kwargs) -> None:
        spectrogram = batch["spectrogram"]
        if self.model_config.use_languages:
            language_ids = batch["language_id"]
        else:
            language_ids = None
        regions = batch["regions"]
        boundaries = batch["boundaries"]
        scores = batch["scores"]
        presence = batch["presence"]
        mask = regions != 0

        x_seg, x_est = self.model.forward_encoder_main(spectrogram, mask=mask)
        boundaries_pred, regions_pred, max_n_pred = self.model.forward_segmenter_main(
            x_seg, known_boundaries=torch.zeros_like(boundaries), mask=mask,
            language=language_ids,
            t=torch.tensor(self.config.d3pm_sample_ts_resolved).to(spectrogram),
            threshold=torch.tensor(self.config.boundary_decoding_threshold).to(spectrogram),
            radius=torch.tensor(self.config.boundary_decoding_radius).to(regions),
        )
        durations_frame_pred = regions_to_durations(regions, max_n=max_n_pred)  # [B, N]
        durations_pred = durations_frame_pred * self.timestep
        presence_pred, scores_pred = self.model.forward_estimator(
            x_est, regions=regions_pred, mask=mask, max_n=max_n_pred,
            threshold=torch.tensor(self.config.note_presence_threshold).to(spectrogram),
        )

        self._update_metrics(
            boundaries_pred=boundaries_pred,
            regions_pred=regions_pred,
            presence_pred=presence_pred,
            scores_pred=scores_pred,
            boundaries_gt=boundaries,
            regions_gt=regions,
            presence_gt=presence,
            scores_gt=scores,
        )

        return {
            "durations": durations_pred,
            "durations_frame": durations_frame_pred,
            "presence": presence_pred,
            "scores": scores_pred,
            "N": regions_pred.amax(dim=-1),
        }

    def _update_metrics(
            self, boundaries_pred, regions_pred, presence_pred, scores_pred,
            boundaries_gt, regions_gt, presence_gt, scores_gt,
    ):
        self.metrics["average_chamfer_distance"].update(boundaries_pred, boundaries_gt)
        self.metrics["quantity_metric_collection"].update(boundaries_pred, boundaries_gt)
        presence_pred_frame = flatten_sequences(presence_pred, regions_pred)
        scores_pred_frame = flatten_sequences(scores_pred, regions_pred)
        presence_gt_frame = flatten_sequences(presence_gt, regions_gt)
        scores_gt_frame = flatten_sequences(scores_gt, regions_gt)
        mask = regions_gt != 0
        self.metrics["presence_metric_collection"].update(
            presence_pred_frame, presence_gt_frame, mask=mask,
        )
        self.metrics["raw_pitch_rmse"].update(
            pred_scores=scores_pred_frame,
            target_scores=scores_gt_frame,
            target_presence=presence_gt_frame,
            mask=mask,
        )
        self.metrics["raw_pitch_accuracy"].update(
            pred_scores=scores_pred_frame,
            target_scores=scores_gt_frame,
            target_presence=presence_gt_frame,
            mask=mask,
        )
        self.metrics["overall_accuracy"].update(
            pred_scores=scores_pred_frame,
            pred_presence=presence_pred_frame,
            target_scores=scores_gt_frame,
            target_presence=presence_gt_frame,
            mask=mask,
        )
        self.metrics["overlap_metric_collection"].update(
            pred_scores=scores_pred_frame,
            pred_presence=presence_pred_frame,
            target_scores=scores_gt_frame,
            target_presence=presence_gt_frame,
            mask=mask,
        )
