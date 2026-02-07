import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from lib.plot import (
    similarity_to_figure,
    distance_boundary_to_figure,
    boundary_to_figure,
)
from modules.decoding import decode_boundaries_from_velocities
from modules.losses import (
    ApproachingMomentumLoss,
    RegionalCosineSimilarityLoss,
)
from modules.losses.boundary_loss import distance_transform
from modules.losses.region_loss import self_cosine_similarity
from modules.metrics import (
    AverageChamferDistance,
    QuantityMetricCollection,
)
from modules.metrics.quantity import match_nearest_boundaries
from modules.midi_extraction import SegmentationModel
from .data import BaseDataset
from .pl_module_base import BaseLightningModule


class SegmentationDataset(BaseDataset):
    pass


class SegmentationLightningModule(BaseLightningModule):
    __dataset__ = SegmentationDataset

    def build_model(self) -> nn.Module:
        return SegmentationModel(self.model_config)

    def register_losses_and_metrics(self) -> None:
        self.register_loss("region_loss", RegionalCosineSimilarityLoss(
            neighborhood_size=self.training_config.loss.region_loss.neighborhood_size,
            exponential_decay=self.training_config.loss.region_loss.exponential_decay,
        ))
        self.register_loss("boundary_loss", ApproachingMomentumLoss(
            radius=self.training_config.loss.boundary_loss.radius,
            decay_start=self.training_config.loss.boundary_loss.decay_start,
            decay_width=self.training_config.loss.boundary_loss.decay_width,
            decay_alpha=self.training_config.loss.boundary_loss.decay_alpha,
            decay_power=self.training_config.loss.boundary_loss.decay_power,
        ))
        self.register_metric("average_chamfer_distance", AverageChamferDistance())
        self.register_metric("quantity_metric_collection", QuantityMetricCollection(
            tolerance=self.training_config.validation.boundary_matching_tolerance
        ))

    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        B = sample["size"]
        spectrogram = sample["spectrogram"]
        if self.model_config.use_languages:
            language_ids = sample["language_id"]
            if not infer:
                language_ids = torch.where(
                    torch.rand(language_ids.shape, device=language_ids.device) < 0.5,
                    language_ids,
                    torch.zeros_like(language_ids)
                )
        else:
            language_ids = None
        regions = sample["regions"]
        boundaries = sample["boundaries"]
        mask = regions != 0

        if infer:
            if self.model_config.mode == "d3pm":
                # 1. Initialize with a whole region.
                # 2. Merge regions by p(t) before each step.
                # 3. Predict full regions.
                latent = None
                velocities = None
                boundaries_pred = None
                num_steps = self.training_config.validation.d3pm_sample_steps
                timestep = torch.full(
                    (B,), fill_value=1 / num_steps,
                    dtype=torch.float32, device=regions.device
                )
                regions_pred = mask.long()
                for i in range(num_steps):
                    t = i * timestep
                    noise = d3pm_noise(regions_pred, t=t)  # [B, T]
                    velocities, latent = self.model(
                        spectrogram, regions=noise, t=t,
                        language=language_ids, mask=mask,
                    )  # [B, T]
                    boundaries_pred = decode_boundaries_from_velocities(
                        velocities, mask=mask,
                        threshold=self.training_config.validation.boundary_decoding_threshold,
                        radius=self.training_config.validation.boundary_decoding_radius,
                    )
                    regions_pred = (boundaries_pred.long().cumsum(dim=-1) + 1) * mask.long()
            elif self.model_config.mode == "completion":
                # One-step prediction from a whole region.
                velocities, latent = self.model(
                    spectrogram, regions=mask.long(),
                    language=language_ids, mask=mask,
                )  # [B, T]
                boundaries_pred = decode_boundaries_from_velocities(
                    velocities, mask=mask,
                    threshold=self.training_config.validation.boundary_decoding_threshold,
                    radius=self.training_config.validation.boundary_decoding_radius,
                )
            else:
                raise ValueError(f"Unknown mode: {self.model_config.mode}.")

            similarities = self_cosine_similarity(latent)  # [B, T, T]

            self.metrics["average_chamfer_distance"].update(boundaries_pred, boundaries)
            self.metrics["quantity_metric_collection"].update(boundaries_pred, boundaries)

            return {
                "similarities": similarities,
                "velocities": velocities,
                "boundaries": boundaries_pred,
            }
        else:
            if self.model_config.mode == "d3pm":
                # Choose random t, merge regions by p(t)
                t = torch.rand(B, device=spectrogram.device)
                noise = d3pm_noise(regions, t=t)  # [B, T]
            elif self.model_config.mode == "completion":
                # Choose random p, merge regions by p
                t = None
                p = torch.randn(B, device=spectrogram.device)
                noise = merge_random_regions(regions, p=p)
            else:
                raise ValueError(f"Unknown mode: {self.model_config.mode}.")

            velocities, latent = self.model(
                spectrogram, regions=noise, t=t,
                language=language_ids, mask=mask,
            )  # [B, T]

            region_loss = self.losses["region_loss"](latent, regions)
            boundary_loss = self.losses["boundary_loss"](velocities, boundaries, mask=mask)

            return {
                "region_loss": region_loss,
                "boundary_loss": boundary_loss,
            }

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        for i in range(len(sample["indices"])):
            data_idx = sample['indices'][i].item()
            if data_idx >= self.training_config.validation.max_plots:
                continue
            T = self.valid_dataset.info["lengths"][data_idx]
            N = self.valid_dataset.info["durations"][data_idx]
            durations = sample["durations"][i, :N]  # [N]
            boundaries = sample["boundaries"][i, :T]  # [T]
            similarities = outputs["similarities"][i, :T, :T]  # [T, T]
            velocities = outputs["velocities"][i, :T]  # [T]
            boundaries_pred = outputs["boundaries"][i, :T]  # [T]

            match_pred_to_target, match_target_to_pred = match_nearest_boundaries(
                boundaries_pred, boundaries, tolerance=self.training_config.validation.boundary_matching_tolerance
            )
            boundaries_tp = match_pred_to_target
            boundaries_fp = boundaries_pred & ~match_pred_to_target
            boundaries_fn = boundaries & ~match_target_to_pred

            distance_gt = distance_transform(boundaries, max_distance=self.training_config.loss.boundary_loss.radius)
            distance_pred = velocities.cumsum(dim=0)
            threshold = self.training_config.validation.boundary_decoding_threshold
            d_min = distance_pred.min().cpu().numpy().item()
            d_max = distance_pred.max().cpu().numpy().item()
            threshold_denorm = threshold * (d_max - d_min + 1e-8) + d_min

            self.plot_regions(
                data_idx, similarities, durations,
                title=self.valid_dataset.info["item_paths"][data_idx]
            )
            self.plot_distance(
                data_idx, distance_gt, distance_pred,
                threshold=threshold_denorm,
                boundaries_tp=boundaries_tp,
                boundaries_fp=boundaries_fp,
                boundaries_fn=boundaries_fn,
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

    def plot_distance(
            self, idx: int,
            distance_gt: torch.Tensor, distance_pred: torch.Tensor,
            threshold: float = None,
            boundaries_tp: torch.Tensor = None,
            boundaries_fp: torch.Tensor = None,
            boundaries_fn: torch.Tensor = None,
            title=None
    ):
        distance_gt = distance_gt.cpu().numpy()
        distance_pred = distance_pred.cpu().numpy()
        if boundaries_tp is not None:
            boundaries_tp = boundaries_tp.cpu().numpy()
        if boundaries_fp is not None:
            boundaries_fp = boundaries_fp.cpu().numpy()
        if boundaries_fn is not None:
            boundaries_fn = boundaries_fn.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"boundaries/boundaries_{idx}", distance_boundary_to_figure(
            distance_gt, distance_pred,
            threshold=threshold,
            boundaries_tp=boundaries_tp,
            boundaries_fp=boundaries_fp,
            boundaries_fn=boundaries_fn,
            title=title
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


def d3pm_noise(regions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    p = (1 + torch.cos(t * torch.pi)) / 2
    regions = merge_random_regions(regions, p=p)
    return regions


def merge_random_regions(regions: torch.Tensor, p: torch.Tensor):
    """
    :param regions: [..., T]
    :param p: [...]
    :return: [..., T]
    """
    N = regions.max()
    if N <= 1:
        return regions.clone()
    *B, _ = regions.shape
    drops = torch.rand((*B, N - 1), device=regions.device) < p.unsqueeze(-1)  # [..., N-1]
    shifts = F.pad(drops.long().cumsum(dim=-1), (2, 0), mode="constant", value=0)  # [..., N+1]
    regions_merged = regions - shifts.gather(dim=-1, index=regions)  # [..., T]
    return regions_merged


def split_random_regions(regions: torch.Tensor, p: torch.Tensor):
    """
    :param regions: [..., T]
    :param p: [...]
    :return: [..., T]
    """
    p = p.clamp(max=0.99).unsqueeze(-1)  # [..., 1]
    N = regions.amax(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    L = (regions != 0).sum(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    pi_hat = (N.float() / L.float() / (1 - p)).clamp(max=1.0)  # [..., 1]
    P = p * pi_hat / (p * pi_hat + (1 - pi_hat))  # [..., 1]
    boundaries = F.pad(torch.diff(regions, dim=-1) > 0, (1, 0), mode="constant", value=0)  # [..., T]
    inserts = (torch.rand_like(regions, dtype=torch.float32) < P) & ~boundaries  # [..., T]
    shifts = inserts.long().cumsum(dim=-1)  # [..., T]
    regions_split = regions + shifts * (regions != 0).long()  # [..., T]
    return regions_split
