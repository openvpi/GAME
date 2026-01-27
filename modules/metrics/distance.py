import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor


class AverageChamferDistance(torchmetrics.Metric):
    """
    This metric computes the average Chamfer Distance between predicted and ground truth boundary sequences.
    Arguments:
        None
    Inputs:
        - pred_boundaries: Tensor of shape [B, T], predicted boundary indicators, 0 = no boundary, 1 = boundary.
        - target_boundaries: Tensor of shape [B, T], ground truth boundary indicators, 0 = no boundary, 1 = boundary.
    Outputs:
        Scalar tensor representing the average Chamfer Distance.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("distance", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, pred_boundaries: Tensor, target_boundaries: Tensor) -> None:
        chamfer_distance = calculate_chamfer_distance(pred_boundaries, target_boundaries)
        boundary_count = target_boundaries.long().sum(dim=-1) + 2
        self.distance += chamfer_distance.sum()
        self.count += boundary_count.sum()

    def compute(self) -> Tensor:
        return self.distance / self.count.float()


def calculate_chamfer_distance(boundaries1: Tensor, boundaries2: Tensor) -> Tensor:
    """
    Compute the Chamfer Distance between two sets of boundaries.
    Arguments:
        boundaries1: Tensor of shape [..., T], first set of boundary indicators.
        boundaries2: Tensor of shape [..., T], second set of boundary indicators.
    Returns:
        Tensor of shape [...,], Chamfer Distance for each batch element.
    """
    boundaries1 = F.pad(boundaries1, (1, 1), mode="constant", value=1)
    boundaries2 = F.pad(boundaries2, (1, 1), mode="constant", value=1)
    indices = torch.arange(
        boundaries1.shape[-1], dtype=torch.float32, device=boundaries1.device
    ).reshape(*(1,) * (boundaries1.ndim - 1), boundaries1.shape[-1])  # [..., T]
    indices1 = torch.where(boundaries1, indices, torch.full_like(indices, float("inf")))  # [..., T]
    indices2 = torch.where(boundaries2, indices, torch.full_like(indices, float("inf")))  # [..., T]
    dist1to2 = torch.min(torch.abs(indices1.unsqueeze(-1) - indices2.unsqueeze(-2)), dim=-1).values  # [..., T]
    dist2to1 = torch.min(torch.abs(indices2.unsqueeze(-1) - indices1.unsqueeze(-2)), dim=-1).values  # [..., T]
    dist1to2_sum = torch.where(dist1to2.isfinite(), dist1to2, torch.zeros_like(dist1to2)).sum(dim=-1)  # [...,]
    dist2to1_sum = torch.where(dist2to1.isfinite(), dist2to1, torch.zeros_like(dist2to1)).sum(dim=-1)  # [...,]
    chamfer_distance = (dist1to2_sum + dist2to1_sum) / 2.0  # [...,]
    return chamfer_distance
