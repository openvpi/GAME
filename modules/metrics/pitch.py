from typing import Any

import torch
import torchmetrics
from torch import Tensor


class NotePresenceMetricCollection(torchmetrics.Metric):
    """
    This metric collection computes a set of binary classification metrics for pitch presence prediction.
    Arguments:
        None
    Inputs:
        - pred_presence: Boolean tensor of shape [..., T], predicted presence indicators, 0 means no score or unvoiced.
        - target_presence: Boolean tensor of shape [..., T], target presence indicators, 0 means no score or unvoiced.
        - weights: Optional Tensor of shape [..., T], weights to apply on each frame.
        - mask: Optional Tensor of shape [..., T], mask to apply on the frames.
    Outputs:
        - presence_precision: Scalar tensor representing the precision, i.e. TP / (TP + FP).
        - presence_recall: Scalar tensor representing the recall, i.e. TP / (TP + FN).
        - presence_tnr: Scalar tensor representing the true negative rate, i.e. TN / (TN + FP).
        - presence_f1_score: Scalar tensor representing the F1 score.
    """

    def __init__(self, postfix: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.postfix = postfix
        self.add_state("tp", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(
            self, pred_presence: Tensor, target_presence: Tensor,
            weights: Tensor = None, mask: Tensor = None
    ) -> None:
        if weights is None:
            weights = torch.ones_like(target_presence).float()
        if mask is not None:
            weights = weights * mask.float()
        self.tp += (pred_presence & target_presence).float().mul(weights).sum()
        self.tn += (~pred_presence & ~target_presence).float().mul(weights).sum()
        self.fp += (pred_presence & ~target_presence).float().mul(weights).sum()
        self.fn += (~pred_presence & target_presence).float().mul(weights).sum()

    def compute(self) -> Any:
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        tnr = self.tn / (self.tn + self.fp + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        return {
            f"presence_precision{self.postfix}": precision,
            f"presence_recall{self.postfix}": recall,
            f"presence_tnr{self.postfix}": tnr,
            f"presence_f1_score{self.postfix}": f1_score,
        }


class RawPitchRMSE(torchmetrics.Metric):
    """
    This metric computes Root Mean Squared Error (RMSE) for pitch scores, considering only the voiced frames
    in the ground truth.
    Inputs:
        - pred_scores: Tensor of shape [..., T], predicted pitch scores.
        - target_scores: Tensor of shape [..., T], target pitch scores.
        - target_presence: Boolean tensor of shape [..., T], target presence indicators, 0 means no score or unvoiced.
        - weights: Optional Tensor of shape [..., T], weights to apply on each frame.
        - mask: Optional Tensor of shape [..., T], mask to apply on the frames.
    Outputs:
        Scalar tensor representing the Root Mean Squared Error.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("squared_error", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(
            self, pred_scores: Tensor,
            target_scores: Tensor, target_presence: Tensor,
            weights: Tensor = None, mask: Tensor = None
    ) -> None:
        if weights is None:
            weights = torch.ones_like(target_scores).float()
        if mask is not None:
            weights = weights * mask.float()
        squared_errors = (pred_scores - target_scores) ** 2
        squared_errors = squared_errors * target_presence.float()
        self.squared_error += (squared_errors * weights).sum()
        self.total += (target_presence.float() * weights).sum()

    def compute(self) -> Any:
        return torch.sqrt(self.squared_error / (self.total + 1e-6))


class RawPitchAccuracy(torchmetrics.Metric):
    """
    This metric computes Raw Pitch Accuracy (RPA), which is the number of correctly predicted pitch frames
    divided by the total number of pitch frames, considering only the voiced frames in the ground truth.
    Arguments:
        tolerance: float, maximum allowed difference between predicted and target scores to be considered correct.
    Inputs:
        - pred_scores: Tensor of shape [..., T], predicted pitch scores.
        - target_scores: Tensor of shape [..., T], target pitch scores.
        - target_presence: Boolean tensor of shape [..., T], target presence indicators, 0 means no score or unvoiced.
        - weights: Optional Tensor of shape [..., T], weights to apply on each frame.
        - mask: Optional Tensor of shape [..., T], mask to apply on the frames.
    Outputs:
        Scalar tensor representing the Raw Pitch Accuracy.
    """

    def __init__(self, *, tolerance: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.add_state("correct", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(
            self, pred_scores: Tensor,
            target_scores: Tensor, target_presence: Tensor,
            weights: Tensor = None, mask: Tensor = None
    ) -> None:
        if weights is None:
            weights = torch.ones_like(target_scores).float()
        if mask is not None:
            weights = weights * mask.float()
        score_diffs = torch.abs(pred_scores - target_scores)
        correct = (score_diffs <= self.tolerance) & target_presence
        total = target_presence
        self.correct += (correct.float() * weights).sum()
        self.total += (total.float() * weights).sum()

    def compute(self) -> Any:
        return self.correct / (self.total + 1e-6)


class OverallAccuracy(torchmetrics.Metric):
    """
    This metric computes Overall Accuracy (OA), which is the number of correctly predicted frames
    (both voiced and unvoiced) divided by the total number of frames.
    Arguments:
        tolerance: float, maximum allowed difference between predicted and target scores to be considered correct.
    Inputs:
        - pred_scores: Tensor of shape [..., T], predicted pitch scores.
        - pred_presence: Boolean tensor of shape [..., T], predicted presence indicators, 0 means no score or unvoiced.
        - target_scores: Tensor of shape [..., T], target pitch scores.
        - target_presence: Boolean tensor of shape [..., T], target presence indicators, 0 means no score or unvoiced.
        - weights: Optional Tensor of shape [..., T], weights to apply on each frame.
        - mask: Optional Tensor of shape [..., T], mask to apply on the frames.
    Outputs:
        Scalar tensor representing the Overall Accuracy.
    """

    def __init__(self, *, tolerance: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.add_state("correct", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(
            self, pred_scores: Tensor, pred_presence: Tensor,
            target_scores: Tensor, target_presence: Tensor,
            weights: Tensor = None, mask: Tensor = None
    ) -> None:
        if weights is None:
            weights = torch.ones_like(target_scores).float()
        if mask is not None:
            weights = weights * mask.float()
        score_diffs = torch.abs(pred_scores - target_scores)
        v_correct = pred_presence & target_presence & (score_diffs <= self.tolerance)
        uv_correct = (~target_presence) & (~pred_presence)
        correct = v_correct | uv_correct
        self.correct += (correct.float() * weights).sum()
        self.total += weights.sum()

    def compute(self) -> Any:
        return self.correct / (self.total + 1e-6)
