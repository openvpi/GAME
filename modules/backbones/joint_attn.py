import torch
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F

from deployment.context import is_export_mode
from modules.backbones.layers import RMSNorm
from modules.backbones.regions import compute_positions_local
from modules.backbones.rope import RegionRoPE


def fill_with_attn_mask(x, attn_mask):
    # In some buggy ONNX exporting logic, fully masked queries may produce NaN.
    # We fix this by manually filling zeros to match the PyTorch native behavior.
    if is_export_mode():
        x = torch.where(attn_mask.any(dim=-2).unsqueeze(-1), x, 0)
    return x


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


class JointAttention(nn.Module):
    def __init__(
            self, dim,
            num_heads,
            head_dim,
            region_token_num=1,
            qk_norm=True,
            use_rope=True,
            rope_mode='mixed',
            use_pool_offset=False,
            theta=10000.0,
            dropout_attn: float = 0.0,
            out_drop_x: float = 0.0,
            out_drop_pool: float = 0.0,
    ):
        super().__init__()

        self.out_drop_x = nn.Dropout(out_drop_x) if out_drop_x > 0. else nn.Identity()
        self.out_drop_pool = nn.Dropout(out_drop_pool) if out_drop_pool > 0. else nn.Identity()
        self.region_token_num = region_token_num
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.rope_mode = rope_mode
        self.use_rope = use_rope
        self.use_pool_offset = use_pool_offset
        self.dropout_attn = dropout_attn

        attn_dim = num_heads * head_dim

        self.pool_qkv = nn.Linear(dim, attn_dim * 3, bias=True)
        self.x_qkv = nn.Linear(dim, attn_dim * 3, bias=True)

        self.qk_norm = qk_norm
        if qk_norm:
            self.pool_q_norm = RMSNorm(head_dim)
            self.pool_k_norm = RMSNorm(head_dim)
            self.x_q_norm = RMSNorm(head_dim)
            self.x_k_norm = RMSNorm(head_dim)

        self.pool_out = nn.Linear(attn_dim, dim, bias=True)
        self.x_out = nn.Linear(attn_dim, dim, bias=True)

        self.pool_norm = RMSNorm(dim)
        self.x_norm = RMSNorm(dim)

        if use_rope:
            if rope_mode == 'mixed':
                self.rope = RegionRoPE(head_dim, mode='global', theta=theta)
            else:
                self.rope = RegionRoPE(head_dim, mode='local', theta=theta)

    def _to_heads(self, x):
        return rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

    def forward(self, pool, x, regions, t_mask, n_mask, attn_mask):
        N = n_mask.shape[1]
        R = self.region_token_num
        P = N * R
        B = x.shape[0]
        T = regions.shape[1]

        pool_normed = self.pool_norm(pool)
        x_normed = self.x_norm(x)

        pool_q, pool_k, pool_v = self.pool_qkv(pool_normed).chunk(3, dim=-1)
        x_q, x_k, x_v = self.x_qkv(x_normed).chunk(3, dim=-1)

        pool_q, pool_k, pool_v = map(self._to_heads, (pool_q, pool_k, pool_v))
        x_q, x_k, x_v = map(self._to_heads, (x_q, x_k, x_v))

        if self.qk_norm:
            pool_q = self.pool_q_norm(pool_q)
            pool_k = self.pool_k_norm(pool_k)
            x_q = self.x_q_norm(x_q)
            x_k = self.x_k_norm(x_k)

        q = torch.cat([pool_q, x_q], dim=2)
        k = torch.cat([pool_k, x_k], dim=2)
        v = torch.cat([pool_v, x_v], dim=2)

        if self.use_rope:
            if self.rope_mode == 'local':
                pool_pos, x_pos = compute_positions_local(regions, R, N, self.use_pool_offset)
                full_pos = torch.cat([pool_pos.float(), x_pos.float()], dim=-1)
                q, k = self.rope(q, k, full_pos, full_pos)
            elif self.rope_mode == 'global':
                pool_pos = torch.arange(P, device=regions.device).unsqueeze(0).expand(B, -1).float()
                x_pos = torch.arange(T, device=regions.device).unsqueeze(0).expand(B, -1).float()
                full_pos = torch.cat([pool_pos, x_pos], dim=-1)
                q, k = self.rope(q, k, full_pos, full_pos)
            elif self.rope_mode == 'mixed':
                pool_seq_pos = torch.arange(P, device=regions.device).unsqueeze(0).expand(B, -1).float()
                x_seq_pos = torch.arange(T, device=regions.device).unsqueeze(0).expand(B, -1).float()
                q_gpos = torch.cat([pool_seq_pos, x_seq_pos], dim=-1)
                pool_lpos, x_lpos = compute_positions_local(regions, R, N, self.use_pool_offset)
                q_ridx = torch.cat([pool_lpos.float(), x_lpos.float()], dim=-1)
                q, k = self.rope(q, k, q_gpos, q_gpos, q_ridx, q_ridx)
            else:
                raise ValueError(f"Unknown rope_mode: {self.rope_mode}")

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )
        out = fill_with_attn_mask(out, attn_mask)

        pool_attn = rearrange(out[:, :, :P, :], 'b h t d -> b t (h d)')
        x_attn = rearrange(out[:, :, P:, :], 'b h t d -> b t (h d)')

        pool_attn = self.pool_out(pool_attn)
        x_attn = self.x_out(x_attn)
        pool = self.out_drop_pool(pool_attn)
        x = self.out_drop_x(x_attn)

        pool = pool * n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P, 1).float()
        x = x * t_mask.unsqueeze(-1).float()

        return pool, x


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


class SplitJointAttention(nn.Module):
    """
    4-way split attention (drop-in replacement for JointAttention):
    - pool->pool: same-stream with RoPE
    - x->x: same-stream with RoPE
    - pool->x: cross-stream with RoPE + region bias decay
    - x->pool: cross-stream with RoPE + region bias decay

    Each stream gets its own normalized attention, results are summed.
    Supports 3 RoPE modes: local, global, mixed (same as JointAttention).
    Can accept pre-built masks via split_masks parameter for efficiency.
    """

    def __init__(
            self, dim,
            num_heads,
            head_dim,
            region_token_num=1,
            qk_norm=True,
            use_rope=True,
            rope_mode='mixed',
            use_pool_offset=False,
            theta=10000.0,
            dropout_attn: float = 0.0,
            out_drop_x: float = 0.0,
            out_drop_pool: float = 0.0,
    ):
        super().__init__()

        self.out_drop_x = nn.Dropout(out_drop_x) if out_drop_x > 0. else nn.Identity()
        self.out_drop_pool = nn.Dropout(out_drop_pool) if out_drop_pool > 0. else nn.Identity()
        self.region_token_num = region_token_num
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.rope_mode = rope_mode
        self.use_rope = use_rope
        self.use_pool_offset = use_pool_offset
        self.dropout_attn = dropout_attn

        attn_dim = num_heads * head_dim

        self.pool_qkv = nn.Linear(dim, attn_dim * 3, bias=True)
        self.x_qkv = nn.Linear(dim, attn_dim * 3, bias=True)

        self.qk_norm = qk_norm
        if qk_norm:
            self.pool_q_norm = RMSNorm(head_dim)
            self.pool_k_norm = RMSNorm(head_dim)
            self.x_q_norm = RMSNorm(head_dim)
            self.x_k_norm = RMSNorm(head_dim)

        self.pool_norm = RMSNorm(dim)
        self.x_norm = RMSNorm(dim)

        # Learnable merge for same-stream + cross-stream outputs
        self.pool_merge = nn.Linear(attn_dim * 2, dim, bias=True)
        self.x_merge = nn.Linear(attn_dim * 2, dim, bias=True)

        # RoPE for same-stream attention only
        if use_rope:
            if rope_mode == 'mixed':
                self.rope = RegionRoPE(head_dim, mode='global', theta=theta)
            else:
                self.rope = RegionRoPE(head_dim, mode='local', theta=theta)

    def _to_heads(self, x):
        return rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

    def forward(self, pool, x, regions, t_mask, n_mask, attn_mask):
        """
        Args:
            pool, x, regions, t_mask, n_mask, attn_mask: same as JointAttention
            attn_mask: optional pre-built masks tuple (pp_mask, xx_mask, px_mask, xp_mask)
                         If None, masks are built internally. Pass for efficiency when reusing masks.
        """
        N = n_mask.shape[1]
        R = self.region_token_num
        P = N * R
        B = x.shape[0]
        T = regions.shape[1]
        device = x.device
        dp = self.dropout_attn if self.training else 0.0

        # Unwrap pre-built masks
        pp_mask, xx_mask, px_mask, xp_mask = attn_mask

        # Pre-norm
        pool_normed = self.pool_norm(pool)
        x_normed = self.x_norm(x)

        # QKV
        pool_q, pool_k, pool_v = self.pool_qkv(pool_normed).chunk(3, dim=-1)
        x_q, x_k, x_v = self.x_qkv(x_normed).chunk(3, dim=-1)

        pool_q, pool_k, pool_v = map(self._to_heads, (pool_q, pool_k, pool_v))
        x_q, x_k, x_v = map(self._to_heads, (x_q, x_k, x_v))

        # QK Norm
        if self.qk_norm:
            pool_q = self.pool_q_norm(pool_q)
            pool_k = self.pool_k_norm(pool_k)
            x_q = self.x_q_norm(x_q)
            x_k = self.x_k_norm(x_k)

        # Apply RoPE based on mode (for all 4 attention paths)
        if self.use_rope:
            if self.rope_mode == 'local':
                pool_lpos, x_lpos = compute_positions_local(regions, R, N, self.use_pool_offset)
                pool_q_r, pool_k_r = self.rope(pool_q, pool_k, pool_lpos.float(), pool_lpos.float())
                x_q_r, x_k_r = self.rope(x_q, x_k, x_lpos.float(), x_lpos.float())
            elif self.rope_mode == 'global':
                pool_pos = torch.arange(P, device=device).unsqueeze(0).expand(B, -1).float()
                x_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).float()
                pool_q_r, pool_k_r = self.rope(pool_q, pool_k, pool_pos, pool_pos)
                x_q_r, x_k_r = self.rope(x_q, x_k, x_pos, x_pos)
            elif self.rope_mode == 'mixed':
                pool_gpos = torch.arange(P, device=device).unsqueeze(0).expand(B, -1).float()
                x_gpos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).float()
                pool_lpos, x_lpos = compute_positions_local(regions, R, N, self.use_pool_offset)
                pool_q_r, pool_k_r = self.rope(pool_q, pool_k, pool_gpos, pool_gpos, pool_lpos.float(),
                                               pool_lpos.float())
                x_q_r, x_k_r = self.rope(x_q, x_k, x_gpos, x_gpos, x_lpos.float(), x_lpos.float())
            else:
                raise ValueError(f"Unknown rope_mode: {self.rope_mode}")
        else:
            pool_q_r, pool_k_r = pool_q, pool_k
            x_q_r, x_k_r = x_q, x_k

        # --- 1. pool -> pool (same-stream with RoPE) ---
        pp_out = F.scaled_dot_product_attention(pool_q_r, pool_k_r, pool_v, attn_mask=pp_mask, dropout_p=dp)
        pp_out = fill_with_attn_mask(pp_out, pp_mask)

        # --- 2. x -> x (same-stream with RoPE) ---
        xx_out = F.scaled_dot_product_attention(x_q_r, x_k_r, x_v, attn_mask=xx_mask, dropout_p=dp)
        xx_out = fill_with_attn_mask(xx_out, xx_mask)

        # --- 3. pool -> x (cross-stream, NO RoPE, use mask only) ---
        px_out = F.scaled_dot_product_attention(pool_q_r, x_k_r, x_v, attn_mask=px_mask, dropout_p=dp)
        px_out = fill_with_attn_mask(px_out, px_mask)

        # --- 4. x -> pool (cross-stream, NO RoPE, use mask only) ---
        xp_out = F.scaled_dot_product_attention(x_q_r, pool_k_r, pool_v, attn_mask=xp_mask, dropout_p=dp)
        xp_out = fill_with_attn_mask(xp_out, xp_mask)

        # Combine: learnable merge of same-stream + cross-stream
        pp_flat = rearrange(pp_out, 'b h t d -> b t (h d)')
        px_flat = rearrange(px_out, 'b h t d -> b t (h d)')
        xx_flat = rearrange(xx_out, 'b h t d -> b t (h d)')
        xp_flat = rearrange(xp_out, 'b h t d -> b t (h d)')

        pool_attn = self.pool_merge(torch.cat([pp_flat, px_flat], dim=-1))
        x_attn = self.x_merge(torch.cat([xx_flat, xp_flat], dim=-1))

        pool = self.out_drop_pool(pool_attn)
        x = self.out_drop_x(x_attn)

        # Mask output
        pool = pool * n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P, 1).float()
        x = x * t_mask.unsqueeze(-1).float()

        return pool, x
