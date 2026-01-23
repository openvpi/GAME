import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def self_cosine_similarity(x: Tensor) -> Tensor:
    x_norm = F.normalize(x.float(), p=2, dim=-1, eps=1e-8)
    sim = x_norm @ x_norm.transpose(-1, -2)
    return sim


class RegionalCosineSimilarityLoss(nn.Module):
    """
    This loss computes the cosine similarity between neighboring regions,
    encouraging similar features within the same region and dissimilar features across different regions.
    Arguments:
        neighborhood_size: int, regions with index difference within this size are considered neighbors.
    Inputs:
        x: Tensor of shape [B, T, C] where B is batch size, T is time frames, C is feature dimension.
        regions: Tensor of shape [B, T] mapping each frame to a region index (1, 2, ..., N). 0 indicates no region.
    Outputs:
        loss: Scalar tensor representing the average regional cosine similarity loss.
    """

    def __init__(self, neighborhood_size: int = 3):
        super().__init__()
        self.neighborhood_size = neighborhood_size

    def get_sign_and_mask(self, regions: Tensor) -> Tensor:
        regions1 = regions.unsqueeze(2)  # [B, T, 1]
        regions2 = regions.unsqueeze(1)  # [B, 1, T]
        neighbor_distance = torch.abs(regions1 - regions2)
        sign = torch.where(regions1 == regions2, 1.0, -1.0).float()  # [B, T, T]
        neighbor_mask = neighbor_distance <= self.neighborhood_size  # [B, T, T]
        padding_mask = (regions1 != 0) & (regions2 != 0)  # [B, T, T]
        mask = torch.triu(neighbor_mask & padding_mask, diagonal=1)  # [B, T, T]
        return sign, mask

    def forward(self, x: Tensor, regions: Tensor):
        sign, mask = self.get_sign_and_mask(regions)  # [B, T, T]
        cos_sim_pred = self_cosine_similarity(x)  # [B, T, T]
        cos_sim_pred[~mask] = 0.0
        loss = 1.0 - (sign * cos_sim_pred).sum() / (mask.float().sum() + 1e-6)
        return loss
