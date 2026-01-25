import math

import torch.nn


class BoundaryEarthMoversDistanceLoss(torch.nn.Module):
    """
    This loss computes the Earth Mover's Distance (EMD) between predicted and ground truth boundary sequences.
    Arguments:
        bidirectional: bool, if True, computes EMD in both forward and backward directions and averages the results.
    Inputs:
        pred: Tensor of shape [B, T], predicted boundary probabilities.
        gt: Tensor of shape [B, T], ground truth boundary indicators, 0 = no boundary, 1 = boundary.
    Outputs:
        loss: Scalar tensor representing the EMD loss.
    """

    def __init__(self, bidirectional=False):
        super().__init__()
        self.criterion = torch.nn.L1Loss()
        self.bidirectional = bidirectional

    def forward(self, pred, gt):
        scale = math.sqrt(gt.shape[1])
        loss = self.criterion(pred.cumsum(dim=1) / scale, gt.cumsum(dim=1) / scale)
        if self.bidirectional:
            loss += self.criterion(pred.flip(1).cumsum(dim=1) / scale, gt.flip(1).cumsum(dim=1) / scale)
            loss /= 2
        return loss
