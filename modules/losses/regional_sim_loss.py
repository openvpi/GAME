import torch
import torch.nn as nn
from torch import Tensor

from lib.functional import self_cosine_similarity


class RegionalCosineSimilarityLoss(nn.Module):
    """
    This loss computes the cosine similarity between neighboring regions,
    encouraging similar features within the same region and dissimilar features across different regions.
    Arguments:
        neighborhood_size: int, regions with index difference within this size are considered neighbors.
        exponential_decay: bool, if True, the penalty for different regions decays exponentially with distance.
    Inputs:
        x: Tensor of shape [B, T, C] where B is batch size, T is time frames, C is feature dimension.
        regions: Tensor of shape [B, T] mapping each frame to a region index (1, 2, ..., N). 0 indicates no region.
    Outputs:
        loss: Scalar tensor representing the average regional cosine similarity loss.
    """

    def __init__(self, neighborhood_size: int, exponential_decay: bool = False):
        super().__init__()
        self.neighborhood_size = neighborhood_size
        self.exponential_decay = exponential_decay

    def get_sign_and_mask(self, regions: Tensor) -> Tensor:
        regions1 = regions.unsqueeze(2)  # [B, T, 1]
        regions2 = regions.unsqueeze(1)  # [B, 1, T]
        distance = torch.abs(regions1 - regions2)  # [B, T, T]
        if self.exponential_decay:
            negative_sign = -torch.exp(1.0 - distance)
        else:
            negative_sign = -1.0
        sign = torch.where(distance == 0, 1.0, negative_sign).float()  # [B, T, T]
        neighbor_mask = distance <= self.neighborhood_size  # [B, T, T]
        padding_mask = (regions1 != 0) & (regions2 != 0)  # [B, T, T]
        mask = torch.triu(neighbor_mask & padding_mask, diagonal=1)  # [B, T, T]
        return sign, mask

    def forward(self, x: Tensor, regions: Tensor):
        sign, mask = self.get_sign_and_mask(regions)  # [B, T, T]
        cos_sim_pred = self_cosine_similarity(x)  # [B, T, T]
        cos_sim_pred[~mask] = 0.0
        loss = 1.0 - (sign * cos_sim_pred).sum() / (mask.float().sum() + 1e-6)
        return loss
