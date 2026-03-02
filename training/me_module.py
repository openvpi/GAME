import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, Tensor

from lib.plot import note_to_figure
from modules.d3pm import (
    d3pm_time_schedule,
    remove_boundaries,
)
from modules.decoding import (
    decode_soft_boundaries,
    decode_gaussian_blurred_probs,
)
from modules.functional import boundaries_to_regions, regions_to_durations, flatten_sequences
from modules.losses import (
    RegionalCosineSimilarityLoss,
    GaussianSoftBoundaryLoss,
    GaussianBlurredBinsLoss,
)
from modules.metrics import (
    AverageChamferDistance,
    QuantityMetricCollection,
    NotePresenceMetricCollection,
    RawPitchRMSE,
    RawPitchAccuracy,
    OverallAccuracy,
    NoteOverlapMetricCollection,
)
from modules.midi_extraction import SegmentationEstimationModel
from training.data import BaseDataset
from training.pl_module_base import BaseLightningModule


class MIDIExtractionDataset(BaseDataset):
    __non_zero_paddings__ = {
        **BaseDataset.__non_zero_paddings__,
        "durations": -1,
    }

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = super().__getitem__(index)
        sample["T"] = torch.tensor(sample["spectrogram"].shape[0], dtype=torch.long)
        sample["N"] = torch.tensor(sample["durations"].shape[0], dtype=torch.long)
        if (pitch_shift := sample["_augmentation"].get("pitch_shift")) is not None:
            sample["scores"] += pitch_shift
        return sample


class MIDIExtractionModule(BaseLightningModule):
    __dataset__ = MIDIExtractionDataset

    def build_model(self) -> nn.Module:
        return SegmentationEstimationModel(self.model_config)

    def register_losses_and_metrics(self) -> None:
        self.register_loss("region_loss", RegionalCosineSimilarityLoss(
            neighborhood_size=self.training_config.loss.region_loss.neighborhood_size,
            exponential_decay=self.training_config.loss.region_loss.exponential_decay,
        ))
        self.register_loss("boundary_loss", GaussianSoftBoundaryLoss(
            std=self.training_config.loss.boundary_loss.std,
        ))
        self.register_loss("note_loss", GaussianBlurredBinsLoss(
            min_val=self.training_config.loss.note_loss.midi_min,
            max_val=self.training_config.loss.note_loss.midi_max,
            num_bins=self.training_config.loss.note_loss.midi_num_bins,
            std=self.training_config.loss.note_loss.midi_std,
        ))
        self._register_metrics()
        if self.use_parallel_dirty_metrics:
            self._register_metrics(postfix="_dirty")

    def _register_metrics(self, postfix: str = "") -> None:
        self.register_metric(f"average_chamfer_distance{postfix}", AverageChamferDistance())
        self.register_metric(f"quantity_metric_collection{postfix}", QuantityMetricCollection(
            tolerance=self.training_config.validation.boundary_matching_tolerance,
            postfix=postfix,
        ))
        self.register_metric(f"presence_metric_collection{postfix}", NotePresenceMetricCollection(
            postfix=postfix,
        ))
        self.register_metric(f"raw_pitch_rmse{postfix}", RawPitchRMSE())
        self.register_metric(f"raw_pitch_accuracy{postfix}", RawPitchAccuracy(
            tolerance=self.training_config.validation.pitch_accuracy_tolerance,
        ))
        self.register_metric(f"overall_accuracy{postfix}", OverallAccuracy(
            tolerance=self.training_config.validation.pitch_accuracy_tolerance,
        ))
        self.register_metric(f"overlap_metric_collection{postfix}", NoteOverlapMetricCollection(
            pitch_width=self.training_config.validation.pitch_overlap_width, postfix=postfix
        ))

    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        spectrogram = sample["spectrogram"]
        if self.model_config.use_languages:
            language_ids = sample["language_id"]
            if not infer:
                language_ids = torch.masked_fill(
                    language_ids,
                    torch.rand(language_ids.shape, device=language_ids.device) < 0.5,
                    0
                )
        else:
            language_ids = None
        regions = sample["regions"]
        boundaries = sample["boundaries"]
        scores = sample["scores"]
        presence = sample["presence"]
        durations = sample["durations"]
        t_mask = regions != 0
        n_mask = durations >= 0

        if infer:
            boundaries_pred, presence_pred, scores_pred = self._forward_infer_e2e(
                spectrogram, language_ids=language_ids, mask=t_mask
            )
            regions_pred = boundaries_to_regions(boundaries_pred, mask=t_mask)
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
            if self.use_parallel_dirty_metrics:
                boundaries_pred_dirty, presence_pred_dirty, scores_pred_dirty = self._forward_infer_e2e(
                    sample["spectrogram_dirty"], language_ids=language_ids, mask=t_mask
                )
                regions_pred_dirty = boundaries_to_regions(boundaries_pred_dirty, mask=t_mask)
                self._update_metrics(
                    boundaries_pred=boundaries_pred_dirty,
                    regions_pred=regions_pred_dirty,
                    presence_pred=presence_pred_dirty,
                    scores_pred=scores_pred_dirty,
                    boundaries_gt=boundaries,
                    regions_gt=regions,
                    presence_gt=presence,
                    scores_gt=scores,
                    postfix="_dirty"
                )
            return {
                "boundaries": boundaries_pred,
                "regions": regions_pred,
                "presence": presence_pred,
                "scores": scores_pred,
            }
        else:
            x_seg, x_est = self.model.forward_encoder(spectrogram)
            boundary_logits, seg_latent = self._forward_train_segmentation(
                x_seg, language_ids=language_ids, boundaries=boundaries, mask=t_mask
            )
            note_logits = self._forward_estimation(
                x_est, regions=regions, t_mask=t_mask, n_mask=n_mask
            )
            region_loss = self.losses["region_loss"](seg_latent, regions)
            boundary_loss = self.losses["boundary_loss"](boundary_logits, boundaries, mask=t_mask)
            note_loss = self.losses["note_loss"](note_logits, scores, presence, mask=n_mask)
            return {
                "region_loss": region_loss,
                "boundary_loss": boundary_loss,
                "note_loss": note_loss,
            }

    def _forward_infer_e2e(self, spectrogram, language_ids, mask):
        x_seg, x_est = self.model.forward_encoder(spectrogram)
        boundaries = self._forward_infer_segmentation(
            x_seg, language_ids=language_ids, mask=mask
        )
        regions = boundaries_to_regions(boundaries, mask=mask)
        max_n = regions.max()
        idx = torch.arange(max_n, dtype=torch.long, device=regions.device).unsqueeze(0)  # [1, N]
        max_idx = regions.amax(dim=-1, keepdim=True)  # [B, 1]
        n_mask = idx < max_idx  # [B, N]
        presence, scores = self._forward_infer_estimation(
            x_est, regions=regions, t_mask=mask, n_mask=n_mask
        )
        return boundaries, presence, scores

    def _forward_train_segmentation(self, x_seg, language_ids, boundaries, mask):
        B = x_seg.shape[0]
        if self.model_config.mode == "d3pm":
            # Choose random t, remove boundaries by p(t)
            t = torch.rand(B, device=x_seg.device)
            p = d3pm_time_schedule(t)
            boundaries_noise = remove_boundaries(boundaries, p=p)  # [B, T]
        elif self.model_config.mode == "completion":
            # Choose random p, merge regions by p
            t = None
            p = torch.rand(B, device=x_seg.device)
            boundaries_noise = remove_boundaries(boundaries, p=p)  # [B, T]
        else:
            raise ValueError(f"Unknown mode: {self.model_config.mode}.")
        regions_noise = boundaries_to_regions(boundaries_noise, mask=mask)  # [B, T]

        logits, latent = self.model.forward_segmentation(
            x_seg, noise=regions_noise, t=t,
            language=language_ids, mask=mask,
        )  # [B, T]

        return logits, latent

    def _forward_infer_segmentation(self, x_seg, language_ids, mask):
        B = x_seg.shape[0]
        if self.model_config.mode == "d3pm":
            # 1. Initialize with no boundaries.
            # 2. Remove boundaries by p(t) before each step.
            # 3. Predict full boundaries.
            boundaries = torch.zeros_like(mask)
            num_steps = self.training_config.validation.d3pm_sample_steps
            timestep = torch.full(
                (B,), fill_value=1 / num_steps,
                dtype=torch.float32, device=x_seg.device
            )
            for i in range(num_steps):
                t = i * timestep
                p = d3pm_time_schedule(t)
                boundaries = remove_boundaries(
                    boundaries, p=p,
                )  # [B, T]
                regions = boundaries_to_regions(boundaries, mask=mask)  # [B, T]
                logits, _ = self.model.forward_segmentation(
                    x_seg, noise=regions, t=t,
                    language=language_ids, mask=mask,
                )  # [B, T]
                boundaries = self._decode_boundaries(logits, mask)
        elif self.model_config.mode == "completion":
            # One-step prediction from no boundaries.
            logits, _ = self.model.forward_segmentation(
                x_seg, regions=mask.long(),
                language=language_ids, mask=mask,
            )  # [B, T]
            boundaries = self._decode_boundaries(logits, mask)
        else:
            raise ValueError(f"Unknown mode: {self.model_config.mode}.")

        return boundaries

    def _decode_boundaries(self, logits, mask):
        soft_boundaries = logits.sigmoid()
        boundaries = decode_soft_boundaries(
            boundaries=soft_boundaries, mask=mask,
            threshold=self.training_config.validation.boundary_decoding_threshold,
            radius=self.training_config.validation.boundary_decoding_radius,
        )
        return boundaries

    def _forward_estimation(self, x_est, regions, t_mask, n_mask):
        logits = self.model.forward_estimation(
            x_est, regions=regions,
            t_mask=t_mask, n_mask=n_mask,
        )  # [B, N, C_out]
        return logits

    def _forward_infer_estimation(self, x_est, regions, t_mask, n_mask):
        logits = self._forward_estimation(
            x_est, regions=regions,
            t_mask=t_mask, n_mask=n_mask
        )
        presence, scores = self._decode_notes(logits, mask=n_mask)
        return presence, scores

    def _decode_notes(self, logits, mask):
        probs = logits.sigmoid()
        scores, presence = decode_gaussian_blurred_probs(
            probs=probs,
            min_val=self.training_config.loss.note_loss.midi_min,
            max_val=self.training_config.loss.note_loss.midi_max,
            deviation=self.training_config.loss.note_loss.midi_std * 3,  # use 3 std as the decoding deviation
            threshold=self.training_config.validation.note_presence_threshold,
        )
        if mask is not None:
            presence = presence & mask
            scores = scores * presence.float()
        return presence, scores

    def _update_metrics(
            self, boundaries_pred, regions_pred, presence_pred, scores_pred,
            boundaries_gt, regions_gt, presence_gt, scores_gt,
            postfix: str = ""
    ):
        self.metrics[f"average_chamfer_distance{postfix}"].update(boundaries_pred, boundaries_gt)
        self.metrics[f"quantity_metric_collection{postfix}"].update(boundaries_pred, boundaries_gt)
        presence_pred_frame = flatten_sequences(presence_pred, regions_pred)
        scores_pred_frame = flatten_sequences(scores_pred, regions_pred)
        presence_gt_frame = flatten_sequences(presence_gt, regions_gt)
        scores_gt_frame = flatten_sequences(scores_gt, regions_gt)
        mask = regions_gt != 0
        self.metrics[f"presence_metric_collection{postfix}"].update(
            presence_pred_frame, presence_gt_frame, mask=mask,
        )
        self.metrics[f"raw_pitch_rmse{postfix}"].update(
            pred_scores=scores_pred_frame,
            target_scores=scores_gt_frame,
            target_presence=presence_gt_frame,
            mask=mask,
        )
        self.metrics[f"raw_pitch_accuracy{postfix}"].update(
            pred_scores=scores_pred_frame,
            target_scores=scores_gt_frame,
            target_presence=presence_gt_frame,
            mask=mask,
        )
        self.metrics[f"overall_accuracy{postfix}"].update(
            pred_scores=scores_pred_frame,
            pred_presence=presence_pred_frame,
            target_scores=scores_gt_frame,
            target_presence=presence_gt_frame,
            mask=mask,
        )
        self.metrics[f"overlap_metric_collection{postfix}"].update(
            pred_scores=scores_pred_frame,
            pred_presence=presence_pred_frame,
            target_scores=scores_gt_frame,
            target_presence=presence_gt_frame,
            mask=mask,
        )

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        for i in range(len(sample["indices"])):
            data_idx = sample['indices'][i].item()
            if data_idx >= self.training_config.validation.max_plots:
                continue
            T = self.valid_dataset.info["lengths"][data_idx]
            N = self.valid_dataset.info["durations"][data_idx]
            scores_gt = sample["scores"][i, :N]
            presence_gt = sample["presence"][i, :N]
            durations_gt = sample["durations"][i, :N]
            M = outputs["regions"][i].max().item()
            scores_pred = outputs["scores"][i, :M]
            presence_pred = outputs["presence"][i, :M]
            durations_pred = regions_to_durations(outputs["regions"][i, :T])
            self.plot_notes(
                idx=data_idx,
                scores_gt=scores_gt, presence_gt=presence_gt, durations_gt=durations_gt,
                scores_pred=scores_pred, presence_pred=presence_pred, durations_pred=durations_pred,
                title=self.valid_dataset.info["item_paths"][data_idx]
            )

    def plot_notes(
            self, idx: int,
            scores_gt: torch.Tensor, presence_gt: torch.Tensor, durations_gt: torch.Tensor,
            scores_pred: torch.Tensor, presence_pred: torch.Tensor, durations_pred: torch.Tensor,
            title=None
    ):
        scores_gt = scores_gt.cpu().numpy()
        presence_gt = presence_gt.cpu().numpy()
        durations_gt = durations_gt.cpu().numpy()
        scores_pred = scores_pred.cpu().numpy()
        presence_pred = presence_pred.cpu().numpy()
        durations_pred = durations_pred.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"notes/notes_{idx}", note_to_figure(
            note_midi_gt=scores_gt, note_rest_gt=~presence_gt, note_dur_gt=durations_gt,
            note_midi_pred=scores_pred, note_rest_pred=~presence_pred, note_dur_pred=durations_pred,
            title=title
        ), global_step=self.global_step)
