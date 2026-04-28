import math

import torch
import torch.nn.functional as F
from torch import nn as nn


def regions_to_local_positions_v1(regions):
    """O(T^2) original"""
    B, T = regions.shape
    valid = regions > 0
    same_region = (regions.unsqueeze(-1) == regions.unsqueeze(-2))
    causal = torch.tril(torch.ones(T, T, device=regions.device, dtype=torch.bool), diagonal=-1)
    same_region_before = same_region & causal.unsqueeze(0) & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    local_pos = same_region_before.sum(dim=-1)
    return local_pos * valid.long()


def regions_to_local_positions_v2(regions):
    """O(T) cumsum"""
    shifted = F.pad(regions[:, :-1], (1, 0), value=0)
    is_start = (regions != shifted)
    ones = torch.ones_like(regions, dtype=torch.long)
    cumsum = ones.cumsum(dim=-1)
    start_cumsum = torch.where(is_start, cumsum, torch.zeros_like(cumsum))
    start_cumsum = start_cumsum.cummax(dim=-1).values
    local_pos = cumsum - start_cumsum
    local_pos = local_pos * (regions > 0).long()
    return local_pos


def regions_to_local_positions_v3(regions):
    """O(T) cumsum, ONNX compatible (no cummax)"""
    B, T = regions.shape
    device = regions.device

    shifted = F.pad(regions[:, :-1], (1, 0), value=0)
    is_start = (regions != shifted)

    ones = torch.ones_like(regions, dtype=torch.long)
    cumsum = ones.cumsum(dim=-1)

    start_cumsum = torch.where(is_start, cumsum, torch.zeros_like(cumsum))
    segment_id = is_start.long().cumsum(dim=-1)

    # 关键修复：只对 is_start 为 True 的位置进行 scatter
    # 将 is_start 为 False 的位置的索引改为 0，这样它们只会写入 segment_start[0]（我们不使用）
    masked_segment_id = torch.where(is_start, segment_id, torch.zeros_like(segment_id))

    segment_start = torch.zeros(B, T + 1, device=device, dtype=cumsum.dtype)
    segment_start.scatter_(1, masked_segment_id, start_cumsum)

    broadcast_start = segment_start.gather(1, segment_id)

    local_pos = cumsum - broadcast_start
    return local_pos * (regions > 0).long()


def compute_positions_local(regions, region_token_num, n, use_pool_offset=False):
    B, T = regions.shape
    R = region_token_num
    P = n * R
    device = regions.device
    if use_pool_offset:
        offsets = torch.arange(R, device=device)
    else:
        offsets = torch.zeros(R, device=device, dtype=torch.long)
    pool_pos = offsets.unsqueeze(0).expand(n, -1).reshape(1, P).expand(B, -1)
    x_local = regions_to_local_positions_v3(regions)
    x_pos = x_local + R
    x_pos = x_pos * (regions > 0).long()
    return pool_pos, x_pos


class RegionBias(nn.Module):
    """Single learnable decay, shared across heads. Output [B, 1, Lq, Lk]."""

    def __init__(self, alpha=1.0, learnable=True):
        super().__init__()
        if learnable:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))
        else:
            self.register_buffer('log_alpha', torch.tensor(math.log(alpha)))

    def forward(self, q_region_idx, k_region_idx):
        dist = (q_region_idx.unsqueeze(-1) - k_region_idx.unsqueeze(-2)).abs().float()
        return (-self.log_alpha.exp() * dist).unsqueeze(1)
