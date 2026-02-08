import math

import torch
import torchmetrics
from torch import Tensor


class QuantityMetricCollection(torchmetrics.Metric):
    """
    This metric collection computes a set of quantity-based metrics for boundary matching tasks.
    Arguments:
        tolerance: int, maximum distance to consider a match between predicted and ground truth boundaries.
    Inputs:
        - pred_boundaries: Tensor of shape [..., T], predicted boundary indicators, 0 = no boundary, 1 = boundary.
        - target_boundaries: Tensor of shape [..., T], ground truth boundary indicators, 0 = no boundary, 1 = boundary.
    Outputs:
        - quantity_error_rate: Scalar tensor representing the Quantity Error Rate (QER).
        - quantity_precision: Scalar tensor representing the precision, i.e. TP / (TP + FP).
        - quantity_recall: Scalar tensor representing the recall, i.e. TP / (TP + FN).
    """

    def __init__(self, *, tolerance: int = 5, postfix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.postfix = postfix
        self.add_state("tp", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        # total batch size, for conversion from boundary counts to region counts
        self.add_state("N", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, pred_boundaries: Tensor, target_boundaries: Tensor) -> None:
        match_pred_to_target, match_target_to_pred = match_nearest_boundaries(
            pred_boundaries, target_boundaries, tolerance=self.tolerance
        )
        self.tp += match_pred_to_target.long().sum()
        self.fp += (pred_boundaries & ~match_pred_to_target).long().sum()
        self.fn += (target_boundaries & ~match_target_to_pred).long().sum()
        self.total += target_boundaries.long().sum()
        self.N += math.prod(target_boundaries.shape[:-1])

    def compute(self) -> dict[str, Tensor]:
        return {
            f"quantity_error_rate{self.postfix}": (self.fn + self.fp).float() / (self.total + self.N).float(),
            f"quantity_precision{self.postfix}": (self.tp + self.N).float() / (self.tp + self.fp + self.N).float(),
            f"quantity_recall{self.postfix}": (self.tp + self.N).float() / (self.tp + self.fn + self.N).float(),
        }


def match_nearest_boundaries(boundaries1: Tensor, boundaries2: Tensor, tolerance: int = 5) -> tuple[Tensor, Tensor]:
    """
    Match boundaries between two sets. Two boundaries match
    if their distance is within tolerance and each boundary is the nearest in its set to the other.
    Arguments:
        boundaries1: Tensor of shape [..., T], first set of boundary indicators.
        boundaries2: Tensor of shape [..., T], second set of boundary indicators.
        tolerance: int, maximum distance to consider a match.
    Returns:
        Two Tensors of shape [..., T], 1 means matched, 0 means no match or non-boundary.
    """
    indices = torch.arange(
        boundaries1.shape[-1], dtype=torch.long, device=boundaries1.device
    ).expand_as(boundaries1)  # [..., T]
    indices_float = indices.float()
    indices1 = torch.where(boundaries1, indices_float, torch.full_like(indices_float, float("inf")))  # [..., T]
    indices2 = torch.where(boundaries2, indices_float, torch.full_like(indices_float, float("inf")))  # [..., T]
    dist1to2, index1to2 = torch.min(torch.abs(indices1.unsqueeze(-1) - indices2.unsqueeze(-2)), dim=-1)  # [..., T]
    dist2to1, index2to1 = torch.min(torch.abs(indices2.unsqueeze(-1) - indices1.unsqueeze(-2)), dim=-1)  # [..., T]
    index1to2to1 = torch.gather(index2to1, dim=-1, index=index1to2)  # [..., T]
    index2to1to2 = torch.gather(index1to2, dim=-1, index=index2to1)  # [..., T]
    match1to2 = boundaries1 & (dist1to2 <= tolerance) & (index1to2to1 == indices)  # [..., T]
    match2to1 = boundaries2 & (dist2to1 <= tolerance) & (index2to1to2 == indices)  # [..., T]
    return match1to2, match2to1
