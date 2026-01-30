from typing import Any

import torch
import torchmetrics
from torch import Tensor


class RawPitchAccuracy(torchmetrics.Metric):

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
        correct = (score_diffs <= self.tolerance) & target_presence
        total = target_presence
        self.correct += (correct.float() * weights).sum()
        self.total += (total.float() * weights).sum()

    def compute(self) -> Any:
        return self.correct / (self.total + 1e-6)


class OverallAccuracy(torchmetrics.Metric):

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


class NotePresenceMetricCollection(torchmetrics.Metric):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
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
            "presence_precision": precision,
            "presence_recall": recall,
            "presence_tnr": tnr,
            "presence_f1_score": f1_score,
        }
