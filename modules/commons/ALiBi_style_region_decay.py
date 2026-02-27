import torch
import torch.nn as nn


class RegionBias(nn.Module):
    """bias = -exp(log_alpha) * |region_i - region_j|, per head"""

    def __init__(self, num_heads):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(num_heads))

    def forward(self, q_region_idx, k_region_idx):
        """
        :param q_region_idx: [B, Lq]
        :param k_region_idx: [B, Lk]
        :return: [B, H, Lq, Lk]
        """
        dist = (q_region_idx.unsqueeze(-1) - k_region_idx.unsqueeze(-2)).abs().float()
        alpha = self.log_alpha.exp()
        return -alpha.view(1, -1, 1, 1) * dist.unsqueeze(1)
