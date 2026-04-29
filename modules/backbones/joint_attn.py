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
    if is_export_mode() and attn_mask is not None:
        x = torch.where(attn_mask.any(dim=-2).unsqueeze(-1), x, 0)
    return x


def build_joint_attention_mask_components(regions, region_token_num, t_mask, n_mask):
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

    # 只有有效的pair才能attend
    valid_pair = full_valid.unsqueeze(-1) & full_valid.unsqueeze(-2)

    return full_region, same_stream, same_region, valid_pair


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
    _, same_stream, same_region, valid_pair = build_joint_attention_mask_components(
        regions, region_token_num, t_mask, n_mask
    )

    # 最终规则: same_stream OR same_region
    attn_allowed = same_stream | same_region

    return (attn_allowed & valid_pair).unsqueeze(1)


class JointAttentionBase(nn.Module):
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

        self.dim = dim
        self.attn_dim = num_heads * head_dim
        self.out_drop_x = nn.Dropout(out_drop_x) if out_drop_x > 0. else nn.Identity()
        self.out_drop_pool = nn.Dropout(out_drop_pool) if out_drop_pool > 0. else nn.Identity()
        self.region_token_num = region_token_num
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.rope_mode = rope_mode
        self.use_rope = use_rope
        self.use_pool_offset = use_pool_offset
        self.dropout_attn = dropout_attn

        self.pool_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)
        self.x_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)

        self.qk_norm = qk_norm
        if qk_norm:
            self.pool_q_norm = RMSNorm(head_dim)
            self.pool_k_norm = RMSNorm(head_dim)
            self.x_q_norm = RMSNorm(head_dim)
            self.x_k_norm = RMSNorm(head_dim)

        self.pool_norm = RMSNorm(dim)
        self.x_norm = RMSNorm(dim)

        if use_rope:
            if rope_mode == 'mixed':
                self.rope = RegionRoPE(head_dim, mode='global', theta=theta)
            else:
                self.rope = RegionRoPE(head_dim, mode='local', theta=theta)

    def _to_heads(self, x):
        return rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

    def _flatten_heads(self, x):
        return rearrange(x, 'b h t d -> b t (h d)')

    def _pool_token_num(self, n_mask):
        return n_mask.shape[1] * self.region_token_num

    def _project_qkv(self, pool, x):
        pool_q, pool_k, pool_v = self.pool_qkv(self.pool_norm(pool)).chunk(3, dim=-1)
        x_q, x_k, x_v = self.x_qkv(self.x_norm(x)).chunk(3, dim=-1)

        pool_q, pool_k, pool_v = map(self._to_heads, (pool_q, pool_k, pool_v))
        x_q, x_k, x_v = map(self._to_heads, (x_q, x_k, x_v))

        if self.qk_norm:
            pool_q = self.pool_q_norm(pool_q)
            pool_k = self.pool_k_norm(pool_k)
            x_q = self.x_q_norm(x_q)
            x_k = self.x_k_norm(x_k)

        return pool_q, pool_k, pool_v, x_q, x_k, x_v

    def _global_positions(self, size, batch_size, device):
        return torch.arange(size, device=device).unsqueeze(0).expand(batch_size, -1).float()

    def _rope_positions(self, regions, n_mask):
        B, T = regions.shape
        N = n_mask.shape[1]
        R = self.region_token_num
        P = N * R
        device = regions.device

        if self.rope_mode == 'local':
            pool_lpos, x_lpos = compute_positions_local(regions, R, N, self.use_pool_offset)
            return (pool_lpos.float(), x_lpos.float()), None
        if self.rope_mode == 'global':
            pool_gpos = self._global_positions(P, B, device)
            x_gpos = self._global_positions(T, B, device)
            return (pool_gpos, x_gpos), None
        if self.rope_mode == 'mixed':
            pool_gpos = self._global_positions(P, B, device)
            x_gpos = self._global_positions(T, B, device)
            pool_lpos, x_lpos = compute_positions_local(regions, R, N, self.use_pool_offset)
            return (pool_gpos, x_gpos), (pool_lpos.float(), x_lpos.float())
        raise ValueError(f"Unknown rope_mode: {self.rope_mode}")

    def _apply_rope_pair(self, q, k, q_pos, k_pos, q_ridx=None, k_ridx=None):
        if q_ridx is None:
            return self.rope(q, k, q_pos, k_pos)
        return self.rope(q, k, q_pos, k_pos, q_ridx, k_ridx)

    def _attention(self, q, k, v, attn_mask):
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_attn if self.training else 0.0
        )
        return fill_with_attn_mask(out, attn_mask)

    def _mask_outputs(self, pool, x, t_mask, n_mask):
        B = x.shape[0]
        P = self._pool_token_num(n_mask)
        pool = pool * n_mask.unsqueeze(-1).expand(-1, -1, self.region_token_num).reshape(B, P, 1).float()
        x = x * t_mask.unsqueeze(-1).float()
        return pool, x


class JointAttention(JointAttentionBase):
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
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            region_token_num=region_token_num,
            qk_norm=qk_norm,
            use_rope=use_rope,
            rope_mode=rope_mode,
            use_pool_offset=use_pool_offset,
            theta=theta,
            dropout_attn=dropout_attn,
            out_drop_x=out_drop_x,
            out_drop_pool=out_drop_pool,
        )
        self.pool_out = nn.Linear(self.attn_dim, self.dim, bias=True)
        self.x_out = nn.Linear(self.attn_dim, self.dim, bias=True)

    def _apply_joint_rope(self, q, k, regions, n_mask):
        if not self.use_rope:
            return q, k

        pos, ridx = self._rope_positions(regions, n_mask)
        full_pos = torch.cat(pos, dim=-1)
        if ridx is None:
            return self._apply_rope_pair(q, k, full_pos, full_pos)

        full_ridx = torch.cat(ridx, dim=-1)
        return self._apply_rope_pair(q, k, full_pos, full_pos, full_ridx, full_ridx)

    def forward(self, pool, x, regions, t_mask, n_mask, attn_mask):
        P = self._pool_token_num(n_mask)
        pool_q, pool_k, pool_v, x_q, x_k, x_v = self._project_qkv(pool, x)
        q = torch.cat([pool_q, x_q], dim=2)
        k = torch.cat([pool_k, x_k], dim=2)
        v = torch.cat([pool_v, x_v], dim=2)
        q, k = self._apply_joint_rope(q, k, regions, n_mask)

        out = self._attention(q, k, v, attn_mask)
        pool_attn = self._flatten_heads(out[:, :, :P, :])
        x_attn = self._flatten_heads(out[:, :, P:, :])

        pool_attn = self.pool_out(pool_attn)
        x_attn = self.x_out(x_attn)
        pool = self.out_drop_pool(pool_attn)
        x = self.out_drop_x(x_attn)
        pool, x = self._mask_outputs(pool, x, t_mask, n_mask)

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


class SplitJointAttention(JointAttentionBase):
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
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            region_token_num=region_token_num,
            qk_norm=qk_norm,
            use_rope=use_rope,
            rope_mode=rope_mode,
            use_pool_offset=use_pool_offset,
            theta=theta,
            dropout_attn=dropout_attn,
            out_drop_x=out_drop_x,
            out_drop_pool=out_drop_pool,
        )

        # Learnable merge for same-stream + cross-stream outputs
        self.pool_merge = nn.Linear(self.attn_dim * 2, dim, bias=True)
        self.x_merge = nn.Linear(self.attn_dim * 2, dim, bias=True)

    def _apply_split_rope(self, pool_q, pool_k, x_q, x_k, regions, n_mask):
        if not self.use_rope:
            return pool_q, pool_k, x_q, x_k

        pos, ridx = self._rope_positions(regions, n_mask)
        pool_pos, x_pos = pos
        if ridx is None:
            pool_q, pool_k = self._apply_rope_pair(pool_q, pool_k, pool_pos, pool_pos)
            x_q, x_k = self._apply_rope_pair(x_q, x_k, x_pos, x_pos)
        else:
            pool_ridx, x_ridx = ridx
            pool_q, pool_k = self._apply_rope_pair(pool_q, pool_k, pool_pos, pool_pos, pool_ridx, pool_ridx)
            x_q, x_k = self._apply_rope_pair(x_q, x_k, x_pos, x_pos, x_ridx, x_ridx)
        return pool_q, pool_k, x_q, x_k

    def forward(self, pool, x, regions, t_mask, n_mask, attn_mask):
        """
        Args:
            pool, x, regions, t_mask, n_mask, attn_mask: same as JointAttention
            attn_mask: optional pre-built masks tuple (pp_mask, xx_mask, px_mask, xp_mask)
                         If None, masks are built internally. Pass for efficiency when reusing masks.
        """
        # Unwrap pre-built masks
        pp_mask, xx_mask, px_mask, xp_mask = attn_mask

        pool_q, pool_k, pool_v, x_q, x_k, x_v = self._project_qkv(pool, x)
        pool_q_r, pool_k_r, x_q_r, x_k_r = self._apply_split_rope(
            pool_q, pool_k, x_q, x_k, regions, n_mask
        )

        # --- 1. pool -> pool (same-stream with RoPE) ---
        pp_out = self._attention(pool_q_r, pool_k_r, pool_v, pp_mask)

        # --- 2. x -> x (same-stream with RoPE) ---
        xx_out = self._attention(x_q_r, x_k_r, x_v, xx_mask)

        # --- 3. pool -> x (cross-stream with RoPE + mask) ---
        px_out = self._attention(pool_q_r, x_k_r, x_v, px_mask)

        # --- 4. x -> pool (cross-stream with RoPE + mask) ---
        xp_out = self._attention(x_q_r, pool_k_r, pool_v, xp_mask)

        # Combine: learnable merge of same-stream + cross-stream
        pp_flat = self._flatten_heads(pp_out)
        px_flat = self._flatten_heads(px_out)
        xx_flat = self._flatten_heads(xx_out)
        xp_flat = self._flatten_heads(xp_out)

        pool_attn = self.pool_merge(torch.cat([pp_flat, px_flat], dim=-1))
        x_attn = self.x_merge(torch.cat([xx_flat, xp_flat], dim=-1))

        pool = self.out_drop_pool(pool_attn)
        x = self.out_drop_x(x_attn)
        pool, x = self._mask_outputs(pool, x, t_mask, n_mask)

        return pool, x
