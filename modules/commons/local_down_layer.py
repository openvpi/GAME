import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.ALiBi_style_region_decay import RegionBias
from modules.commons.region_rope import regions_to_local_positions, region_start_positions, RegionRoPE


class LearnablePoolTokens(nn.Module):
    """
    Generates pool/query tokens from learnable embeddings.
    Output: [B, N * region_token_num, C], masked regions produce zeros.
    """

    def __init__(self, dim, region_token_num=1):
        super().__init__()
        self.region_token_num = region_token_num
        self.emb = nn.Parameter(torch.randn(region_token_num, dim) * 0.02)

    def forward(self, max_n, n_mask):
        """
        :param max_n: int
        :param n_mask: [B, N] bool
        :return: [B, N * region_token_num, C]
        """
        B, N = n_mask.shape
        tokens = self.emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, R, C]
        tokens = tokens * n_mask.unsqueeze(-1).unsqueeze(-1).float()
        return tokens.reshape(B, N * self.region_token_num, -1)


def build_local_mask_sa(regions, region_token_num, max_n, t_mask, n_mask):
    """
    Local mask for self-attention: sequence = [pool, x].
    Same region can attend to each other, different regions blocked.
    :return: [B, 1, P+T, P+T] bool, True = allowed
    """
    B, T = regions.shape
    R = region_token_num
    P = max_n * R
    device = regions.device

    # Pool region assignment: [B, P]
    pool_region = torch.arange(1, max_n + 1, device=device) \
        .unsqueeze(-1).expand(-1, R).reshape(1, P).expand(B, -1)
    pool_valid = n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P)

    # Full sequence region and validity
    full_region = torch.cat([pool_region, regions], dim=-1)  # [B, P+T]
    full_valid = torch.cat([pool_valid, t_mask], dim=-1)  # [B, P+T]

    # Same region & both valid & both non-padding-region
    same = full_region.unsqueeze(-1) == full_region.unsqueeze(-2)  # [B, L, L]
    valid = full_valid.unsqueeze(-1) & full_valid.unsqueeze(-2)
    non_pad = (full_region != 0).unsqueeze(-1) & (full_region != 0).unsqueeze(-2)

    return (same & valid & non_pad).unsqueeze(1)  # [B, 1, L, L]


def build_local_mask_ca(regions, region_token_num, max_n, t_mask, n_mask):
    """
    Local mask for cross-attention: Q = pool, KV = x.
    Pool token attends only to same-region x tokens.
    :return: [B, 1, P, T] bool, True = allowed
    """
    B, T = regions.shape
    R = region_token_num
    P = max_n * R
    device = regions.device

    pool_region = torch.arange(1, max_n + 1, device=device) \
        .unsqueeze(-1).expand(-1, R).reshape(1, P).expand(B, -1)
    pool_valid = n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P)

    same = pool_region.unsqueeze(-1) == regions.unsqueeze(-2)  # [B, P, T]
    valid = pool_valid.unsqueeze(-1) & t_mask.unsqueeze(-2)
    non_pad = (pool_region != 0).unsqueeze(-1) & (regions != 0).unsqueeze(-2)

    return (same & valid & non_pad).unsqueeze(1)  # [B, 1, P, T]


def build_global_mask_sa(region_token_num, max_n, t_mask, n_mask):
    """
    Pad mask for global self-attention: sequence = [pool, x].
    :return: [B, 1, P+T, P+T] bool
    """
    B, T = t_mask.shape
    R = region_token_num
    P = max_n * R

    pool_valid = n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P)
    full_valid = torch.cat([pool_valid, t_mask], dim=-1)  # [B, P+T]
    return (full_valid.unsqueeze(-1) & full_valid.unsqueeze(-2)).unsqueeze(1)


def build_global_mask_ca(region_token_num, max_n, t_mask, n_mask):
    """
    Pad mask for global cross-attention: Q = pool, KV = x.
    :return: [B, 1, P, T] bool
    """
    B, T = t_mask.shape
    R = region_token_num
    P = max_n * R

    pool_valid = n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P)
    return (pool_valid.unsqueeze(-1) & t_mask.unsqueeze(-2)).unsqueeze(1)


def compute_positions_local(regions, region_token_num, max_n, use_pool_offset=False):
    """
    Local mode positions for [pool, x] sequence.
    Pool tokens: [0,...,R-1] or all 0 per region
    X tokens: local pos within region + R offset
    :return: pool_pos [B, P], x_pos [B, T]
    """
    B, T = regions.shape
    R = region_token_num
    P = max_n * R
    device = regions.device

    if use_pool_offset:
        offsets = torch.arange(R, device=device)
    else:
        offsets = torch.zeros(R, device=device, dtype=torch.long)

    pool_pos = offsets.unsqueeze(0).expand(max_n, -1).reshape(1, P).expand(B, -1)

    x_local = regions_to_local_positions(regions)  # [B, T]
    x_pos = x_local + R
    x_pos = x_pos * (regions > 0).long()

    return pool_pos, x_pos


def compute_positions_global(regions, region_token_num, max_n, use_pool_offset=False):
    """
    Global mode positions.
    :return: (pool_global_pos, x_global_pos, pool_region_idx, x_region_idx)
    """
    B, T = regions.shape
    R = region_token_num
    P = max_n * R
    device = regions.device

    x_global_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    x_region_idx = regions

    starts = region_start_positions(regions, max_n)  # [B, N]

    if use_pool_offset:
        offsets = torch.arange(R, device=device, dtype=torch.float).view(1, 1, R) * 0.5
        pool_global_pos = starts.unsqueeze(-1).float() + offsets
    else:
        pool_global_pos = starts.unsqueeze(-1).expand(-1, -1, R).float()

    pool_global_pos = pool_global_pos.reshape(B, P)

    pool_region_idx = torch.arange(1, max_n + 1, device=device) \
        .unsqueeze(-1).expand(-1, R).reshape(1, P).expand(B, -1)

    return pool_global_pos, x_global_pos, pool_region_idx, x_region_idx


class LocalAttentionPoolDown(nn.Module):
    """
    Self-attention pool with local (within-region) mask.
    Sequence = [pool_tokens, x], self-attention, return pool outputs only.
    RoPE: local mode, region-internal positions, full head_dim.
    """

    def __init__(
            self,
            head_dim,
            region_token_num=1,
            use_rope=True,
            use_pool_offset=False,
            dropout_attn=0.0,
            theta=10000.0,
    ):
        super().__init__()
        self.region_token_num = region_token_num
        self.use_rope = use_rope
        self.use_pool_offset = use_pool_offset
        self.dropout_attn = dropout_attn

        if use_rope:
            self.rope = RegionRoPE(head_dim, mode='local', theta=theta)

    def forward(
            self,
            pool_q, pool_k, pool_v,
            x_q, x_k, x_v,
            regions, t_mask, n_mask, max_n,
    ):
        """
        :param pool_q/k/v: [B, H, P, D] where P = max_n * region_token_num
        :param x_q/k/v: [B, H, T, D]
        :param regions: [B, T] int64, 0=pad, 1..N
        :param t_mask: [B, T] bool
        :param n_mask: [B, N] bool
        :param max_n: int
        :return: [B, H, P, D]
        """
        R = self.region_token_num
        P = max_n * R

        # Concat [pool, x]
        q = torch.cat([pool_q, x_q], dim=2)  # [B, H, P+T, D]
        k = torch.cat([pool_k, x_k], dim=2)
        v = torch.cat([pool_v, x_v], dim=2)

        # RoPE
        if self.use_rope:
            pool_pos, x_pos = compute_positions_local(
                regions, R, max_n, self.use_pool_offset
            )
            q_pos = torch.cat([pool_pos, x_pos], dim=-1)  # [B, P+T]
            k_pos = q_pos
            q, k = self.rope(q, k, q_pos, k_pos)

        # Local mask
        mask = build_local_mask_sa(regions, R, max_n, t_mask, n_mask)  # [B, 1, P+T, P+T]

        # Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )

        # Return only pool token outputs
        return out[:, :, :P, :]


class AttentionPoolDown(nn.Module):
    """
    Self-attention pool with global visibility.
    Sequence = [pool_tokens, x], self-attention, return pool outputs only.
    RoPE: global mode, split head_dim = global pos + region index.
    Optional region_bias for ALiBi-style decay.
    """

    def __init__(
            self,
            head_dim,
            num_heads,
            region_token_num=1,
            use_rope=True,
            use_region_rope=True,
            use_region_bias=False,
            use_pool_offset=False,
            dropout_attn=0.0,
            theta=10000.0,
    ):
        super().__init__()
        self.region_token_num = region_token_num
        self.use_rope = use_rope
        self.use_region_rope = use_region_rope
        self.use_region_bias = use_region_bias
        self.use_pool_offset = use_pool_offset
        self.dropout_attn = dropout_attn

        if use_rope:
            mode = 'global' if use_region_rope else 'local'
            self.rope = RegionRoPE(head_dim, mode=mode, theta=theta)

        if use_region_bias:
            self.region_bias = RegionBias(num_heads)

    def forward(
            self,
            pool_q, pool_k, pool_v,
            x_q, x_k, x_v,
            regions, t_mask, n_mask, max_n,
    ):
        """
        :param pool_q/k/v: [B, H, P, D]
        :param x_q/k/v: [B, H, T, D]
        :param regions: [B, T]
        :param t_mask: [B, T] bool
        :param n_mask: [B, N] bool
        :param max_n: int
        :return: [B, H, P, D]
        """
        R = self.region_token_num
        P = max_n * R
        T = x_q.shape[2]

        q = torch.cat([pool_q, x_q], dim=2)
        k = torch.cat([pool_k, x_k], dim=2)
        v = torch.cat([pool_v, x_v], dim=2)

        # RoPE
        if self.use_rope:
            if self.use_region_rope:
                pool_gpos, x_gpos, pool_ridx, x_ridx = compute_positions_global(
                    regions, R, max_n, self.use_pool_offset
                )
                q_gpos = torch.cat([pool_gpos, x_gpos.float()], dim=-1)
                k_gpos = q_gpos
                q_ridx = torch.cat([pool_ridx, x_ridx], dim=-1)
                k_ridx = q_ridx
                q, k = self.rope(q, k, q_gpos, k_gpos, q_ridx, k_ridx)
            else:
                # No region rope, just use sequential positions
                L = P + T
                seq_pos = torch.arange(L, device=q.device).unsqueeze(0).expand(q.shape[0], -1)
                q, k = self.rope(q, k, seq_pos, seq_pos)

        # Mask (pad only, no local constraint)
        mask = build_global_mask_sa(R, max_n, t_mask, n_mask)  # [B, 1, P+T, P+T]

        # Region bias
        if self.use_region_bias:
            pool_ridx_b = torch.arange(1, max_n + 1, device=regions.device) \
                .unsqueeze(-1).expand(-1, R).reshape(1, P).expand(regions.shape[0], -1)
            full_ridx = torch.cat([pool_ridx_b, regions], dim=-1)  # [B, P+T]
            bias = self.region_bias(full_ridx, full_ridx)  # [B, H, P+T, P+T]
            # Convert bool mask to float and add bias
            mask = mask.float()
            mask = torch.where(mask > 0, bias, torch.tensor(-torch.inf, device=mask.device))

        # Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask if isinstance(mask, torch.Tensor) and mask.dtype == torch.float else mask,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )

        return out[:, :, :P, :]


class CrossAttentionDown(nn.Module):
    """
    Cross-attention: query tokens do Q, x does KV. Global visibility.
    RoPE: global mode, split head_dim = global pos + region index.
    Optional region_bias.
    """

    def __init__(
            self,
            head_dim,
            num_heads,
            region_token_num=1,
            use_rope=True,
            use_region_rope=True,
            use_region_bias=False,
            use_pool_offset=False,
            dropout_attn=0.0,
            theta=10000.0,
    ):
        super().__init__()
        self.region_token_num = region_token_num
        self.use_rope = use_rope
        self.use_region_rope = use_region_rope
        self.use_region_bias = use_region_bias
        self.use_pool_offset = use_pool_offset
        self.dropout_attn = dropout_attn

        if use_rope:
            mode = 'global' if use_region_rope else 'local'
            self.rope = RegionRoPE(head_dim, mode=mode, theta=theta)

        if use_region_bias:
            self.region_bias = RegionBias(num_heads)

    def forward(
            self,
            query_q,
            x_k, x_v,
            regions, t_mask, n_mask, max_n,
    ):
        """
        :param query_q: [B, H, P, D] where P = max_n * region_token_num
        :param x_k/v: [B, H, T, D]
        :param regions: [B, T]
        :param t_mask: [B, T] bool
        :param n_mask: [B, N] bool
        :param max_n: int
        :return: [B, H, P, D]
        """
        R = self.region_token_num
        P = max_n * R

        q = query_q
        k = x_k
        v = x_v

        # RoPE
        if self.use_rope:
            if self.use_region_rope:
                pool_gpos, x_gpos, pool_ridx, x_ridx = compute_positions_global(
                    regions, R, max_n, self.use_pool_offset
                )
                q, k = self.rope(q, k, pool_gpos, x_gpos.float(), pool_ridx, x_ridx)
            else:
                q_pos = torch.arange(P, device=q.device).unsqueeze(0).expand(q.shape[0], -1)
                k_pos = torch.arange(x_k.shape[2], device=q.device).unsqueeze(0).expand(q.shape[0], -1)
                q, k = self.rope(q, k, q_pos, k_pos)

        # Mask (pad only)
        mask = build_global_mask_ca(R, max_n, t_mask, n_mask)  # [B, 1, P, T]

        # Region bias
        if self.use_region_bias:
            pool_ridx = torch.arange(1, max_n + 1, device=regions.device) \
                .unsqueeze(-1).expand(-1, R).reshape(1, P).expand(regions.shape[0], -1)
            bias = self.region_bias(pool_ridx, regions)  # [B, H, P, T]
            mask = mask.float()
            mask = torch.where(mask > 0, bias, torch.tensor(-torch.inf, device=mask.device))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask if isinstance(mask, torch.Tensor) and mask.dtype == torch.float else mask,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )

        return out


class LocalCrossAttentionDown(nn.Module):
    """
    Cross-attention with local (within-region) mask.
    Query tokens do Q, x does KV. Only same-region pairs allowed.
    RoPE: local mode, region-internal positions, full head_dim.
    """

    def __init__(
            self,
            head_dim,
            region_token_num=1,
            use_rope=True,
            use_pool_offset=False,
            dropout_attn=0.0,
            theta=10000.0,
    ):
        super().__init__()
        self.region_token_num = region_token_num
        self.use_rope = use_rope
        self.use_pool_offset = use_pool_offset
        self.dropout_attn = dropout_attn

        if use_rope:
            self.rope = RegionRoPE(head_dim, mode='local', theta=theta)

    def forward(
            self,
            query_q,
            x_k, x_v,
            regions, t_mask, n_mask, max_n,
    ):
        """
        :param query_q: [B, H, P, D]
        :param x_k/v: [B, H, T, D]
        :param regions: [B, T]
        :param t_mask: [B, T] bool
        :param n_mask: [B, N] bool
        :param max_n: int
        :return: [B, H, P, D]
        """
        R = self.region_token_num
        P = max_n * R

        q = query_q
        k = x_k
        v = x_v

        # RoPE
        if self.use_rope:
            pool_pos, x_pos = compute_positions_local(
                regions, R, max_n, self.use_pool_offset
            )
            # For CA: q uses pool_pos, k uses x_pos (no R offset for CA,
            # since pool tokens are separate from x sequence)
            # Actually keep R offset on x_pos so pool at 0..R-1, x at R..
            q, k = self.rope(q, k, pool_pos, x_pos)

        # Local mask
        mask = build_local_mask_ca(regions, R, max_n, t_mask, n_mask)  # [B, 1, P, T]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )

        return out
