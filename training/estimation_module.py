import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from lib.plot import similarity_to_figure, distance_boundary_to_figure, boundary_to_figure
from modules.decoding import decode_gaussian_blurred_probs
from modules.losses import (
    RegionalCosineSimilarityLoss,
)
from modules.losses.note_loss import GaussianBlurredBinsLoss
from modules.midi_extraction import EstimationModel
from training.data import BaseDataset
from training.pl_module_base import BaseLightningModule

NOTE_DECODING_THRESHOLD = 0.1
NOTE_ACCURACY_TOLERANCE = 0.5


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
        self.register_loss("note_loss", GaussianBlurredBinsLoss(
            min_val=self.model_config.midi_min,
            max_val=self.model_config.midi_max,
            num_bins=self.model_config.midi_num_bins,
            deviation=self.training_config.loss.note_loss.deviation,
        ))
        # TODO pitch metrics

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
            probs, _ = self.model(
                spectrogram, regions=regions, max_n=max_n,
                t_mask=t_mask, n_mask=n_mask, sigmoid=True,
            )  # [B, N, C_out]
            scores, presence = decode_gaussian_blurred_probs(
                probs=probs,
                min_val=self.model_config.midi_min,
                max_val=self.model_config.midi_max,
                deviation=3 * self.training_config.loss.note_loss.deviation,
                threshold=NOTE_DECODING_THRESHOLD,
            )
            # TODO: pitch metrics
            return {
                "scores": scores,
                "presence": presence,
            }
        else:
            logits, latent = self.model(
                spectrogram, regions=regions, max_n=max_n,
                t_mask=t_mask, n_mask=n_mask, sigmoid=False,
            )
            region_adapt_loss = self.losses["region_adapt_loss"](latent, regions)
            note_loss = self.losses["note_loss"](logits, scores, presence, mask=n_mask)
            return {
                "region_adapt_loss": region_adapt_loss,
                "note_loss": note_loss,
            }

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        pass
