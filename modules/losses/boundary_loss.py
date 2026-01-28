import math

import torch
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
    This loss constraints the velocities on the time dimension to approach a set of boundaries. Steps:
        1. Apply Distance Transform to compute the distance between each position to its nearest boundary;
        2. Compute a momentum weight based on the distance;
        3. Use the momentum to perform a weighted gradient detach on the predicted velocity;
        4. Accumulate the velocity through time dimension to get the predicted distance;
        5. Compute the L1 loss between the predicted distance and ground truth distance.
    Arguments:
        radius: int, maximum distance to consider for Distance Transform.
         The velocities within this radius are towards the boundary, otherwise zeros.
        decay_start: int, regions where distance <= decay_start are applied with full momentum.
        decay_width: int, regions where distance > decay_start + decay_width have no momentum.
        decay_alpha: float, spacial scaling factor for the decay function.
        decay_power: float, power of the decay function.
    Inputs:
        - velocities: Tensor of shape [..., T], predicted velocities.
          Negative means approaching from left side, positive means approaching from right side.
        - boundaries: Tensor of shape [..., T], ground truth boundary indicators, 0 = no boundary, 1 = boundary.
        - mask: Optional Tensor of shape [..., T], mask to apply on the loss.
    Outputs:
        Scalar tensor representing the approaching momentum loss.
    """

    def __init__(
            self, radius: int = 20,
            decay_start: int = 20, decay_width: int = 20,
            decay_alpha: float = 0.5, decay_power: float = 2.0,
    ):
        super().__init__()
        self.radius = radius
        self.decay_start = decay_start
        self.decay_end = decay_start + decay_width
        self.decay_alpha = decay_alpha
        self.decay_power = decay_power
        self.criterion = nn.L1Loss(reduction="none")

    def get_momentum(self, distance: Tensor):
        momentum = torch.where(
            distance <= self.decay_start,
            1.0,
            torch.pow(1.0 + self.decay_alpha * (distance - self.decay_start), -self.decay_power)
        ).float()
        momentum *= (distance <= self.decay_end).float()
        return momentum

    def forward(self, velocities: Tensor, boundaries: Tensor, mask=None):
        if mask is not None:
            mask = mask.float()
            velocities = velocities * mask
        gt_distance = distance_transform(boundaries, max_distance=self.radius)
        momentum = self.get_momentum(gt_distance)
        velocities = velocities * momentum + (1.0 - momentum) * velocities.detach()
        pred_distance = velocities.cumsum(dim=-1)
        scale = gt_distance.amax(dim=-1, keepdim=True) + 1e-6
        loss = self.criterion(pred_distance / scale, gt_distance / scale)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss
