import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from lib.plot import (
    note_to_figure,
    probs_to_figure,
)
from modules.decoding import decode_cascaded_dial_pointers
from modules.losses import (
    RegionalCosineSimilarityLoss,
    CascadedDialCaliperLoss,
)
from modules.metrics.pitch import (
    NotePresenceMetricCollection,
    RawPitchRMSE,
    RawPitchAccuracy,
    OverallAccuracy,
)
from modules.midi_extraction import EstimationModel
from training.data import BaseDataset
from training.pl_module_base import BaseLightningModule


class EstimationDataset(BaseDataset):
    __non_zero_paddings__ = {
        **BaseDataset.__non_zero_paddings__,
        "durations": -1,
    }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        if (pitch_shift := sample["_augmentation"].get("pitch_shift")) is not None:
            sample["scores"] += pitch_shift
        return sample


class EstimationLightningModule(BaseLightningModule):
    __dataset__ = EstimationDataset

    def build_model(self) -> nn.Module:
        return EstimationModel(self.model_config)

    def register_losses_and_metrics(self) -> None:
        self.register_loss("region_adapt_loss", RegionalCosineSimilarityLoss(
            neighborhood_size=self.training_config.loss.region_loss.neighborhood_size,
            exponential_decay=self.training_config.loss.region_loss.exponential_decay,
        ))
        self.register_loss("note_presence_loss", nn.BCEWithLogitsLoss(reduction="mean"))
        self.register_loss("note_beam_loss", nn.MSELoss(reduction="mean"))
        self.register_loss("note_dial_loss", CascadedDialCaliperLoss(
            periods=self.training_config.loss.note_loss.dial_periods,
        ))
        self._register_metrics()
        if self.use_parallel_dirty_metrics:
            self._register_metrics(postfix="_dirty")

    def _register_metrics(self, postfix: str = ""):
        self.register_metric(f"presence_metric_collection{postfix}", NotePresenceMetricCollection(
            postfix=postfix
        ))
        self.register_metric(f"raw_pitch_rmse{postfix}", RawPitchRMSE())
        for tol in self.training_config.validation.note_accuracy_tolerances:
            self.register_metric(f"raw_pitch_accuracy_{100 * tol:.0f}cents{postfix}", RawPitchAccuracy(
                tolerance=tol,
            ))
            self.register_metric(f"overall_accuracy_{100 * tol:.0f}cents{postfix}", OverallAccuracy(
                tolerance=tol,
            ))

    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        spectrogram = sample["spectrogram"]
        regions = sample["regions"]
        durations = sample["durations"]
        t_mask = regions != 0
        n_mask = durations >= 0
        max_n = durations.shape[1]
        scores = sample["scores"]
        presence = sample["presence"]

        if infer:
            weights = durations.clamp(min=0).float()
            presence_pred, scores_pred = self._forward_infer(
                spectrogram, regions=regions,
                max_n=max_n, t_mask=t_mask, n_mask=n_mask
            )
            self._update_metrics(
                presence_pred=presence_pred, scores_pred=scores_pred,
                presence_gt=presence, scores_gt=scores,
                weights=weights, mask=n_mask,
            )

            if self.use_parallel_dirty_metrics:
                presence_pred_dirty, scores_pred_dirty = self._forward_infer(
                    sample["spectrogram_dirty"], regions=regions,
                    max_n=max_n, t_mask=t_mask, n_mask=n_mask
                )
                self._update_metrics(
                    presence_pred=presence_pred_dirty, scores_pred=scores_pred_dirty,
                    presence_gt=presence, scores_gt=scores,
                    weights=weights, mask=n_mask,
                    postfix="_dirty"
                )

            return {
                "scores": scores_pred,
                "presence": presence_pred,
            }
        else:
            presence_logits, beam_norm_pred, dials_pred, latent = self._forward(
                spectrogram, regions=regions,
                max_n=max_n, t_mask=t_mask, n_mask=n_mask
            )

            min_val = self.training_config.loss.note_loss.midi_min
            max_val = self.training_config.loss.note_loss.midi_max
            region_adapt_loss = self.losses["region_adapt_loss"](latent, regions)
            if not n_mask.any():
                note_presence_loss = torch.tensor(0.0, device=presence_logits.device)
            else:
                note_presence_loss = self.losses["note_presence_loss"](
                    presence_logits[n_mask], presence[n_mask].float()
                )
            voiced = presence & n_mask
            if not voiced.any():
                note_beam_loss = torch.tensor(0.0, device=beam_norm_pred.device)
                note_dial_loss = torch.tensor(0.0, device=dials_pred.device)
            else:
                note_beam_loss = self.losses["note_beam_loss"](
                    beam_norm_pred[voiced], (scores[voiced] - min_val) / (max_val - min_val)
                )
                note_dial_loss = self.losses["note_dial_loss"](
                    dials=dials_pred[voiced], targets=scores[voiced]
                )
            return {
                "region_adapt_loss": region_adapt_loss,
                "note_presence_loss": note_presence_loss,
                "note_beam_loss": note_beam_loss,
                "note_dial_loss": note_dial_loss,
            }

    def _forward(self, spectrogram, regions, max_n, t_mask, n_mask):
        num_dials = len(self.training_config.loss.note_loss.dial_periods)
        estimations, latent = self.model(
            spectrogram, regions=regions, max_n=max_n,
            t_mask=t_mask, n_mask=n_mask,
        )  # [B, N, C_out]
        presence_logits = estimations[:, :, 0]  # [B, N]
        beam_norm = estimations[:, :, 1]  # [B, N]
        dials = estimations[:, :, 2:].reshape(-1, max_n, num_dials, 2)  # [B, N, num_dials, 2]
        return presence_logits, beam_norm, dials, latent

    def _forward_infer(self, spectrogram, regions, max_n, t_mask, n_mask):
        presence_logits, beam_norm, dials, latent = self._forward(
            spectrogram, regions=regions,
            max_n=max_n, t_mask=t_mask, n_mask=n_mask
        )
        min_val = self.training_config.loss.note_loss.midi_min
        max_val = self.training_config.loss.note_loss.midi_max
        presence = presence_logits.sigmoid() >= self.training_config.validation.note_presence_threshold
        beam = beam_norm * (max_val - min_val) + min_val
        scores = decode_cascaded_dial_pointers(
            beam=beam,
            dials=dials,
            periods=self.training_config.loss.note_loss.dial_periods,
        )
        return presence, scores

    def _update_metrics(self, presence_pred, scores_pred, presence_gt, scores_gt, weights, mask, postfix=""):
        self.metrics[f"presence_metric_collection{postfix}"].update(
            presence_pred, presence_gt, weights=weights, mask=mask,
        )
        self.metrics[f"raw_pitch_rmse{postfix}"].update(
            pred_scores=scores_pred,
            target_scores=scores_gt, target_presence=presence_gt,
            weights=weights, mask=mask,
        )
        for tol in self.training_config.validation.note_accuracy_tolerances:
            self.metrics[f"raw_pitch_accuracy_{100 * tol:.0f}cents{postfix}"].update(
                pred_scores=scores_pred,
                target_scores=scores_gt, target_presence=presence_gt,
                weights=weights, mask=mask,
            )
            self.metrics[f"overall_accuracy_{100 * tol:.0f}cents{postfix}"].update(
                pred_scores=scores_pred, pred_presence=presence_pred,
                target_scores=scores_gt, target_presence=presence_gt,
                weights=weights, mask=mask,
            )

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        for i in range(len(sample["indices"])):
            data_idx = sample['indices'][i].item()
            if data_idx >= self.training_config.validation.max_plots:
                continue
            N = self.valid_dataset.info["durations"][data_idx]
            durations = sample["durations"][i, :N]  # [N]
            scores = sample["scores"][i, :N]  # [N]
            presence = sample["presence"][i, :N]  # [N]
            scores_pred = outputs["scores"][i, :N]  # [N]
            presence_pred = outputs["presence"][i, :N]  # [N]
            self.plot_notes(
                idx=data_idx,
                durations=durations,
                scores_gt=scores,
                presence_gt=presence,
                scores_pred=scores_pred,
                presence_pred=presence_pred,
                title=self.valid_dataset.info["item_paths"][data_idx]
            )

    def plot_probs(
            self, idx: int,
            frame2note: torch.Tensor,
            probs_gt: torch.Tensor, probs_pred: torch.Tensor,
            title=None
    ):
        bin_idx = torch.arange(probs_gt.shape[1], dtype=torch.long, device=probs_gt.device)  # [C_out]

        def get_bin_range(probs: torch.Tensor):
            bin_visible = probs.amax(dim=0) >= 1e-3  # [N, C_out] -> [C_out]
            if not bin_visible.any():
                return None
            C = probs.shape[1]
            bin_min = max(bin_idx[bin_visible].min().item() - 5, 0)
            bin_max = min(bin_idx[bin_visible].max().item() + 5, C - 1)
            return bin_min, bin_max + 1

        # Cut out bins with too low values
        bin_range = get_bin_range(probs_gt)
        if bin_range is None:
            bin_range = get_bin_range(probs_pred)
        if bin_range is None:
            bin_range = (0, probs_gt.shape[1])
        bin_start, bin_end = bin_range
        probs_gt = probs_gt[:, bin_start: bin_end]
        probs_pred = probs_pred[:, bin_start: bin_end]
        # Repeat according to durations
        gather_idx = frame2note.unsqueeze(-1).repeat(1, probs_gt.shape[1])  # [T, C_cut]
        probs_gt_repeat = torch.gather(F.pad(probs_gt, (0, 0, 1, 0)), dim=0, index=gather_idx)
        probs_pred_repeat = torch.gather(F.pad(probs_pred, (0, 0, 1, 0)), dim=0, index=gather_idx)
        probs_gt_repeat = probs_gt_repeat.cpu().numpy()
        probs_pred_repeat = probs_pred_repeat.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"probs/probs_{idx}", probs_to_figure(
            probs_gt=probs_gt_repeat, probs_pred=probs_pred_repeat,
            title=title
        ), global_step=self.global_step)

    def plot_notes(
            self, idx: int, durations: torch.Tensor,
            scores_gt: torch.Tensor, presence_gt: torch.Tensor,
            scores_pred: torch.Tensor, presence_pred: torch.Tensor,
            title=None
    ):
        durations = durations.cpu().numpy()
        scores_gt = scores_gt.cpu().numpy()
        presence_gt = presence_gt.cpu().numpy()
        scores_pred = scores_pred.cpu().numpy()
        presence_pred = presence_pred.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"notes/notes_{idx}", note_to_figure(
            durations,
            note_midi_gt=scores_gt, note_rest_gt=~presence_gt,
            note_midi_pred=scores_pred, note_rest_pred=~presence_pred,
            title=title
        ), global_step=self.global_step)
