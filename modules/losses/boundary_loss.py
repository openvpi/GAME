import math

import torch
import torch.nn
from torch import nn, Tensor

from lib.functional import distance_transform


class BoundaryEarthMoversDistanceLoss(torch.nn.Module):
    """
    This loss computes the Earth Mover's Distance (EMD) between predicted and ground truth boundary sequences.
    Arguments:
        bidirectional: bool, if True, computes EMD in both forward and backward directions and averages the results.
    Inputs:
        - pred: Tensor of shape [B, T], predicted boundary probabilities.
        - gt: Tensor of shape [B, T], ground truth boundary indicators, 0 = no boundary, 1 = boundary.
    Outputs:
        Scalar tensor representing the EMD loss.
    """

    def __init__(self, bidirectional=False):
        super().__init__()
        self.criterion = torch.nn.L1Loss()
        self.bidirectional = bidirectional

    def forward(self, pred, gt):
        scale = math.sqrt(gt.shape[1])
        gt = gt.float()
        loss = self.criterion(pred.cumsum(dim=1) / scale, gt.cumsum(dim=1) / scale)
        if self.bidirectional:
            loss += self.criterion(pred.flip(1).cumsum(dim=1) / scale, gt.flip(1).cumsum(dim=1) / scale)
            loss /= 2
        return loss


class ApproachingMomentumLoss(nn.Module):
    """
    This loss encourages the predicted velocities to approach a series of ground truth boundaries. Steps:
        1. Apply Distance Transform to compute the distance between each position to its nearest boundary;
        2. Compute a momentum weight based on the distance;
        3. Accumulate the velocity through time dimension to get the predicted distance;
        4. Compute the weighted L1 loss between the predicted distance and ground truth distance.
    Arguments:
        constant_radius: int, within this radius from the boundary, full momentum is applied.
        cutoff_radius: int, beyond this radius from the boundary, no momentum is applied.
        decay_power: float, power of the decay function for momentum beyond the constant radius.
        decay_alpha: float, scaling factor for the decay function.
    Inputs:
        - velocities: Tensor of shape [B, T], predicted velocities.
          Negative means approaching from left side, positive means approaching from right side.
        - boundaries: Tensor of shape [B, T], ground truth boundary indicators, 0 = no boundary, 1 = boundary.
        - mask: Optional Tensor of shape [B, T], mask to apply on the loss.
    Outputs:
        Scalar tensor representing the approaching momentum loss.
    """

    def __init__(
            self, constant_radius: int = 20, cutoff_radius: int = 40,
            decay_power: float = 2.0, decay_alpha: float = 0.5,
    ):
        super().__init__()
        self.constant_radius = constant_radius
        self.cutoff_radius = cutoff_radius
        self.decay_power = decay_power
        self.decay_alpha = decay_alpha
        self.criterion = nn.L1Loss(reduction="none")

    def get_momentum(self, distance: Tensor):
        momentum = torch.where(
            distance <= self.constant_radius,
            1.0,
            torch.pow(1.0 + self.decay_alpha * (distance - self.constant_radius), -self.decay_power)
        ).float()
        momentum *= (distance <= self.cutoff_radius).float()
        return momentum

    def forward(self, velocities: Tensor, boundaries: Tensor, mask=None):
        gt_distance = distance_transform(boundaries)
        momentum = self.get_momentum(gt_distance)
        pred_distance = velocities.cumsum(dim=1)
        scale = gt_distance.max(dim=1, keepdim=True).values.clamp(min=1, max=self.cutoff_radius)
        loss = self.criterion(pred_distance / scale, gt_distance / scale) * momentum
        if mask is not None:
            loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss
