import torch
import torch.nn.functional as F


def build_joint_attention_mask(regions, region_token_num, t_mask, n_mask):
    """
    构建 MMDiT attention mask。

    Args:
        regions: [B, T] 每个时间步的区域ID (0=padding, 1~N=有效区域)
        region_token_num: R, 每个区域的pool token数量
        t_mask: [B, T] 时间步有效性mask
        n_mask: [B, N] 区域有效性mask

    Returns:
        mask: [B, 1, P+T, P+T] attention mask (True=可attend)
    """
    B, T = regions.shape
    N = n_mask.shape[1]
    R = region_token_num
    P = N * R  # pool token总数
    device = regions.device

    # pool tokens的区域ID: [1,1,...,1, 2,2,...,2, ..., N,N,...,N]
    pool_region = torch.arange(1, N + 1, device=device) \
        .unsqueeze(-1).expand(-1, R).reshape(1, P).expand(B, -1)

    # 完整的区域ID序列: [pool_regions, x_regions]
    full_region = torch.cat([pool_region, regions], dim=-1)

    # pool tokens的有效性
    pool_valid = n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P)
    full_valid = torch.cat([pool_valid, t_mask], dim=-1)

    # 标记哪些是pool tokens
    is_pool = torch.cat([
        torch.ones(B, P, device=device, dtype=torch.bool),
        torch.zeros(B, T, device=device, dtype=torch.bool),
    ], dim=-1)

    # same_stream: pool-pool 或 x-x (全局attention)
    same_stream = is_pool.unsqueeze(-1) == is_pool.unsqueeze(-2)

    # same_region: 相同区域 (局部attention)
    same_region = full_region.unsqueeze(-1) == full_region.unsqueeze(-2)
    non_pad_region = (full_region != 0).unsqueeze(-1) & (full_region != 0).unsqueeze(-2)
    same_region = same_region & non_pad_region

    # 最终规则: same_stream OR same_region
    attn_allowed = same_stream | same_region

    # 只有有效的pair才能attend
    valid_pair = full_valid.unsqueeze(-1) & full_valid.unsqueeze(-2)

    return (attn_allowed & valid_pair).unsqueeze(1)


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


def regions_to_local_positions_v1(regions):
    """O(T^2) original"""
    B, T = regions.shape
    valid = regions > 0
    same_region = (regions.unsqueeze(-1) == regions.unsqueeze(-2))
    causal = torch.tril(torch.ones(T, T, device=regions.device, dtype=torch.bool), diagonal=-1)
    same_region_before = same_region & causal.unsqueeze(0) & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    local_pos = same_region_before.sum(dim=-1)
    return local_pos * valid.long()


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


def build_split_attention_masks(regions, region_token_num, t_mask, n_mask, region_bias=None):
    """
    Pre-build masks for SplitJointAttention to avoid rebuilding every forward pass.

    Returns:
        pp_mask: [B, 1, P, P] pool->pool padding mask (None if no padding)
        xx_mask: [B, 1, T, T] x->x padding mask (None if no padding)
        px_mask: [B, 1, P, T] pool->x cross-stream mask with region bias
        xp_mask: [B, 1, T, P] x->pool cross-stream mask with region bias
        # pool_region: [B, P] region indices for pool tokens
        # pool_valid: [B, P] valid mask for pool tokens
    """
    B, T = regions.shape
    N = n_mask.shape[1]
    R = region_token_num
    P = N * R
    device = regions.device

    # Region indices
    pool_region = torch.arange(1, N + 1, device=device) \
        .unsqueeze(-1).expand(-1, R).reshape(1, P).expand(B, -1)
    pool_valid = n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P)

    def _build_pad_bias(valid):
        mask = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        return torch.where(mask, 0.0, -10000.0).unsqueeze(1)

    def _build_cross_bias(q_region, k_region, q_valid, k_valid):
        pad_mask = q_valid.unsqueeze(-1) & k_valid.unsqueeze(-2)

        if region_bias is not None:
            # Soft mask: region bias decay (different regions get negative bias)
            pad_bias = torch.where(pad_mask, 0.0, -10000.0).unsqueeze(1)
            decay = region_bias(q_region, k_region)
            return pad_bias + decay
        else:
            # Hard mask: only same region can attend (cross-stream局部attention)
            same_region = q_region.unsqueeze(-1) == k_region.unsqueeze(-2)
            # 排除 padding region (region=0)
            non_pad = (q_region != 0).unsqueeze(-1) & (k_region != 0).unsqueeze(-2)
            valid_mask = pad_mask & same_region & non_pad
            return torch.where(valid_mask, 0.0, -10000.0).unsqueeze(1)

    # Same-stream masks (None if no padding for flash path)
    pp_mask = None if pool_valid.all() else _build_pad_bias(pool_valid)
    xx_mask = None if t_mask.all() else _build_pad_bias(t_mask)

    # Cross-stream masks
    px_mask = _build_cross_bias(pool_region, regions, pool_valid, t_mask)
    xp_mask = _build_cross_bias(regions, pool_region, t_mask, pool_valid)

    return pp_mask, xx_mask, px_mask, xp_mask
