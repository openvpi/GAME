import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from lib.functional import mel2ph_to_dur
from lib.plot import similarity_to_figure, boundary_to_figure
from modules.losses import (
    BoundaryEarthMoversDistanceLoss,
    RegionalCosineSimilarityLoss
)
from modules.syllable_splitter import SyllableSplitter
from .data import BaseDataset
from .pl_module_base import BaseLightningModule


class SyllablesDataset(BaseDataset):
    pass


class SyllablesLightningModule(BaseLightningModule):
    __dataset__ = SyllablesDataset

    def build_model(self) -> nn.Module:
        return SyllableSplitter(self.model_config)

    # noinspection PyAttributeOutsideInit
    def post_init(self) -> None:
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def register_losses_and_metrics(self) -> None:
        self.register_loss("region_loss", RegionalCosineSimilarityLoss(
            neighborhood_size=self.training_config.loss.region_loss.neighborhood_size,
        ))
        self.register_loss("boundary_loss", BoundaryEarthMoversDistanceLoss(
            bidirectional=self.training_config.loss.boundary_loss.bidirectional,
        ))

    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        spectrogram = sample["spectrogram"]
        language_ids = sample["language_id"]
        if not infer:
            language_ids = torch.where(
                torch.rand(language_ids.shape, device=language_ids.device) < 0.5,
                language_ids,
                torch.zeros_like(language_ids)
            )
        frame2syllable = sample["frame2syllable"]
        boundaries_gt = torch.diff(
            frame2syllable, dim=1, prepend=frame2syllable.new_ones((frame2syllable.shape[0], 1))
        ) > 0

        features, boundaries = self.model(spectrogram, language_ids)  # [B, T, T]
        if infer:
            similarities = self.cosine_similarity(
                features.unsqueeze(2), features.unsqueeze(1)
            )  # [B, T, T]
            return {
                "similarities": similarities,
                "boundaries": boundaries,
            }
        else:
            region_loss = self.losses["region_loss"](features, frame2syllable)
            boundary_loss = self.losses["boundary_loss"](boundaries, boundaries_gt.float())
            return {
                "region_loss": region_loss,
                "boundary_loss": boundary_loss,
            }

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        for i in range(len(sample["indices"])):
            data_idx = sample['indices'][i].item()
            T = self.valid_dataset.info["lengths"][data_idx]
            if data_idx >= self.training_config.validation.max_plots:
                continue
            similarities = outputs["similarities"][i, :T, :T]  # [T, T]
            boundaries = outputs["boundaries"][i, :T]  # [T]
            frame2syllable = sample["frame2syllable"][i, :T]  # [T]
            durations = mel2ph_to_dur(frame2syllable.unsqueeze(0), frame2syllable.max()).squeeze(0)  # [N]
            boundaries_gt = torch.diff(
                frame2syllable, dim=0, prepend=frame2syllable.new_ones((1,))
            ) > 0
            self.plot_regions(
                data_idx, similarities, durations,
                title=self.valid_dataset.info["item_paths"][data_idx]
            )
            self.plot_boundaries(
                data_idx, boundaries_gt, boundaries, durations_gt=durations,
                title=self.valid_dataset.info["item_paths"][data_idx]
            )

    def plot_regions(
            self, idx: int,
            similarities: torch.Tensor, durations: torch.Tensor,
            title=None
    ):
        similarities = similarities.cpu().numpy()
        durations = durations.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"regions/regions_{idx}", similarity_to_figure(
            similarities, durations, title=title
        ), global_step=self.global_step)

    def plot_boundaries(
            self, idx: int,
            boundaries_gt: torch.Tensor, boundaries_pred: torch.Tensor,
            durations_gt: torch.Tensor = None, durations_pred: torch.Tensor = None,
            title=None
    ):
        boundaries_gt = boundaries_gt.cpu().numpy()
        boundaries_pred = boundaries_pred.cpu().numpy()
        if durations_gt is not None:
            durations_gt = durations_gt.cpu().numpy()
        if durations_pred is not None:
            durations_pred = durations_pred.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"boundaries/boundaries_{idx}", boundary_to_figure(
            boundaries_gt, boundaries_pred, dur_gt=durations_gt, dur_pred=durations_pred, title=title
        ), global_step=self.global_step)
