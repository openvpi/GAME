import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from lib.plot import (
    similarity_to_figure,
    distance_boundary_to_figure,
    boundary_to_figure,
)
from modules.decoding import decode_boundaries_from_velocities, decode_boundaries_from_velocities_with_confidences
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
from modules.midi_extraction import SegmentationModel, GeneratorSegmentationModel
from .data import BaseDataset
from .pl_module_base import BaseLightningModule


class SegmentationDataset(BaseDataset):
    pass


class GSegmentationLightningModule(BaseLightningModule):
    __dataset__ = SegmentationDataset

    def build_model(self) -> nn.Module:
        # return torch.compile(GeneratorSegmentationModel(self.model_config), mode="default")
        return GeneratorSegmentationModel(self.model_config)

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

    def build_input(self, regions: torch.Tensor):
        r_time = torch.rand(regions.shape[0], 1, device=regions.device)
        pm = (torch.cos(torch.pi * r_time) + 1) / 2
        m_regions = batch_merge_random_regions(regions, pm)

        return m_regions, r_time

    def diff_infer(self, spectrogram, regions, language_ids=None, mask=None, infer_time_step=20, ):
        B = regions.shape[0]
        time_step = 1 / infer_time_step
        time_step_t = torch.tensor(time_step, device=regions.device).expand(B, 1)
        for i in range(infer_time_step):



            velocities, latent = self.model(
                spectrogram, regions=regions, times=time_step_t * i,
                language=language_ids, mask=mask
            )  # [B, T]

            if i == infer_time_step - 1:
                similarities = self_cosine_similarity(latent)  # [B, T, T]
                boundaries_pred = decode_boundaries_from_velocities(
                    velocities, mask=mask,
                    threshold=self.training_config.validation.boundary_decoding_threshold,
                    radius=self.training_config.validation.boundary_decoding_radius,
                )
                return velocities ,similarities, boundaries_pred

            else:
                # d_threshold
                boundaries_pred, conf = decode_boundaries_from_velocities_with_confidences(velocities, mask=mask,
                                                                                          threshold=self.training_config.validation.boundary_decoding_threshold,
                                                                                          radius=self.training_config.validation.boundary_decoding_radius)

                keep_prop=(torch.cos(torch.pi * time_step_t*(i+1)) + 1) / 2
                kp=torch.rand_like(boundaries_pred.float())*boundaries_pred.float()
                keep_mask=kp>keep_prop
                final_boundaries=boundaries_pred&keep_mask
                cr=torch.cumsum(final_boundaries.float(),dim=-1)+1
                if mask is not None:
                    regions=cr*mask
                else:
                    regions=cr
                regions=regions.long()


    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
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
            superregions = mask.long()  # a whole region
            velocities, similarities, boundaries_pred=self.diff_infer(spectrogram, superregions, language_ids=language_ids, mask=mask)
            self.metrics["average_chamfer_distance"].update(boundaries_pred, boundaries)
            self.metrics["quantity_metric_collection"].update(boundaries_pred, boundaries)
            return {
                "similarities": similarities,
                "velocities": velocities,
                "boundaries": boundaries_pred,
            }


        else:
            # superregions = merge_random_regions(regions, p=self.training_config.validation.boundary_drop_probability)
            # superregions = batch_merge_random_regions(regions, p=torch.rand(regions.shape[0],1, device=regions.device))

            superregions, r_time = self.build_input(regions)
            velocities, latent = self.model(
                spectrogram, regions=superregions.long(), times=r_time,
                language=language_ids, mask=mask
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


def merge_random_regions(regions: torch.Tensor, p: float):
    N = regions.max()
    if N <= 1:
        return regions.clone()
    *B, _ = regions.shape
    drops = torch.rand((*B, N - 1), device=regions.device) < p  # [..., N-1]
    shifts = F.pad(drops.long().cumsum(dim=-1), (2, 0), mode="constant", value=0)  # [..., N+1]
    regions_merged = regions - shifts.gather(dim=-1, index=regions)  # [..., T]
    return regions_merged


def batch_merge_random_regions(regions: torch.Tensor, p: torch.Tensor):
    N = regions.max()
    if N <= 1:
        return regions.clone()
    *B, _ = regions.shape
    drops = torch.rand((*B, N - 1), device=regions.device) < p  # [..., N-1]
    shifts = F.pad(drops.long().cumsum(dim=-1), (2, 0), mode="constant", value=0)  # [..., N+1]
    regions_merged = regions - shifts.gather(dim=-1, index=regions)  # [..., T]
    return regions_merged


def split_random_regions(regions: torch.Tensor, p: float):
    p = min(0.99, p)
    N = regions.amax(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    L = (regions != 0).sum(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    pi_hat = (N.float() / L.float() / (1 - p)).clamp(max=1.0)  # [..., 1]
    P = p * pi_hat / (p * pi_hat + (1 - pi_hat))  # [..., 1]
    boundaries = F.pad(torch.diff(regions, dim=-1) > 0, (1, 0), mode="constant", value=0)  # [..., T]
    draws = (torch.rand_like(regions, dtype=torch.float32) <= P) & ~boundaries  # [..., T]
    shifts = draws.long().cumsum(dim=-1)  # [..., T]
    regions_split = regions + shifts * (regions != 0).long()  # [..., T]
    return regions_split


def batch_split_random_regions(regions: torch.Tensor, p: torch.Tensor):
    p = p.clamp(max=0.99, min=0.001)
    N = regions.amax(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    L = (regions != 0).sum(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    pi_hat = (N.float() / L.float() / (1 - p)).clamp(max=1.0)  # [..., 1]
    P = p * pi_hat / (p * pi_hat + (1 - pi_hat))  # [..., 1]
    boundaries = F.pad(torch.diff(regions, dim=-1) > 0, (1, 0), mode="constant", value=0)  # [..., T]
    draws = (torch.rand_like(regions, dtype=torch.float32) <= P) & ~boundaries  # [..., T]
    shifts = draws.long().cumsum(dim=-1)  # [..., T]
    regions_split = regions + shifts * (regions != 0).long()  # [..., T]
    return regions_split
