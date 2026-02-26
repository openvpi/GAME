import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.backbones.EBF import CgMLP, GLUFFN, FFN, LayScale
from modules.backbones.eglu import HalfCacheGLUFFN


def compute_inv_freq(dim: int, theta: float = 10000.0):
    return 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))


def single_apply_rotary_emb(x, freqs_cos, freqs_sin):
    x_ = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
    x_r, x_i = x_[..., 0], x_[..., 1]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(-2)
    return x_out.type_as(x)


def apply_rotary_by_positions(x, positions, inv_freq):
    pos = positions.unsqueeze(-1).float()
    inv = inv_freq.view(*((1,) * (pos.ndim - 1)), -1)
    freqs = pos * inv
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    n_extra = x.ndim - positions.ndim - 1
    shape = freqs_cos.shape[:1] + (1,) * n_extra + freqs_cos.shape[1:]
    freqs_cos = freqs_cos.view(shape)
    freqs_sin = freqs_sin.view(shape)
    return single_apply_rotary_emb(x, freqs_cos, freqs_sin)


class RegionRoPE(nn.Module):
    def __init__(self, head_dim, mode='local', theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.mode = mode
        if mode == 'local':
            self.register_buffer('inv_freq', compute_inv_freq(head_dim, theta), persistent=False)
        elif mode == 'global':
            assert head_dim % 2 == 0
            half = head_dim // 2
            self.register_buffer('inv_freq_global', compute_inv_freq(half, theta), persistent=False)
            self.register_buffer('inv_freq_region', compute_inv_freq(half, theta), persistent=False)

    def forward(self, q, k, q_positions, k_positions, q_region_idx=None, k_region_idx=None):
        if self.mode == 'local':
            q = apply_rotary_by_positions(q, q_positions, self.inv_freq)
            k = apply_rotary_by_positions(k, k_positions, self.inv_freq)
        else:
            half = self.head_dim // 2
            q_g, q_r = q[..., :half], q[..., half:]
            k_g, k_r = k[..., :half], k[..., half:]
            q_g = apply_rotary_by_positions(q_g, q_positions, self.inv_freq_global)
            k_g = apply_rotary_by_positions(k_g, k_positions, self.inv_freq_global)
            q_r = apply_rotary_by_positions(q_r, q_region_idx, self.inv_freq_region)
            k_r = apply_rotary_by_positions(k_r, k_region_idx, self.inv_freq_region)
            q = torch.cat([q_g, q_r], dim=-1)
            k = torch.cat([k_g, k_r], dim=-1)
        return q, k


class RMSnorm(torch.nn.Module):
    def __init__(self, dim: int, init_num=1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * init_num)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


def build_join_attention_mask(regions, region_token_num, max_n, t_mask, n_mask):
    """
    构建 MMDiT attention mask。

    Args:
        regions: [B, T] 每个时间步的区域ID (0=padding, 1~N=有效区域)
        region_token_num: R, 每个区域的pool token数量
        max_n: N, 最大区域数量
        t_mask: [B, T] 时间步有效性mask
        n_mask: [B, N] 区域有效性mask

    Returns:
        mask: [B, 1, P+T, P+T] attention mask (True=可attend)
    """
    B, T = regions.shape
    R = region_token_num
    P = max_n * R  # pool token总数
    device = regions.device

    # pool tokens的区域ID: [1,1,...,1, 2,2,...,2, ..., N,N,...,N]
    pool_region = torch.arange(1, max_n + 1, device=device) \
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


def regions_to_local_positions_v2(regions):
    """O(T) cumsum"""
    B, T = regions.shape
    shifted = F.pad(regions[:, :-1], (1, 0), value=0)
    is_start = (regions != shifted)
    ones = torch.ones_like(regions, dtype=torch.long)
    cumsum = ones.cumsum(dim=-1)
    start_cumsum = torch.where(is_start, cumsum, torch.zeros_like(cumsum))
    start_cumsum = start_cumsum.cummax(dim=-1).values
    local_pos = cumsum - start_cumsum
    local_pos = local_pos * (regions > 0).long()
    return local_pos


def compute_positions_local(regions, region_token_num, max_n, use_pool_offset=False):
    B, T = regions.shape
    R = region_token_num
    P = max_n * R
    device = regions.device
    if use_pool_offset:
        offsets = torch.arange(R, device=device)
    else:
        offsets = torch.zeros(R, device=device, dtype=torch.long)
    pool_pos = offsets.unsqueeze(0).expand(max_n, -1).reshape(1, P).expand(B, -1)
    x_local = regions_to_local_positions_v2(regions)
    x_pos = x_local + R
    x_pos = x_pos * (regions > 0).long()
    return pool_pos, x_pos


class JoinAtention(nn.Module):
    def __init__(self, dim,
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
            self.pool_q_norm = RMSnorm(head_dim)
            self.pool_k_norm = RMSnorm(head_dim)
            self.x_q_norm = RMSnorm(head_dim)
            self.x_k_norm = RMSnorm(head_dim)

        self.pool_out = nn.Linear(attn_dim, dim, bias=True)
        self.x_out = nn.Linear(attn_dim, dim, bias=True)

        self.pool_norm = RMSnorm(dim)
        self.x_norm = RMSnorm(dim)

        if use_rope:
            if rope_mode == 'mixed':
                self.rope = RegionRoPE(head_dim, mode='global', theta=theta)
            else:
                self.rope = RegionRoPE(head_dim, mode='local', theta=theta)

    def _to_heads(self, x):
        return rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

    def forward(self, pool, x, regions, t_mask, n_mask, attn_mask, max_n=None):
        if max_n is None:
            max_n = n_mask.shape[1]

        R = self.region_token_num
        P = max_n * R
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
                pool_pos, x_pos = compute_positions_local(regions, R, max_n, self.use_pool_offset)
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
                pool_lpos, x_lpos = compute_positions_local(regions, R, max_n, self.use_pool_offset)
                q_ridx = torch.cat([pool_lpos.float(), x_lpos.float()], dim=-1)
                q, k = self.rope(q, k, q_gpos, q_gpos, q_ridx, q_ridx)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )

        pool_attn = rearrange(out[:, :, :P, :], 'b h t d -> b t (h d)')
        x_attn = rearrange(out[:, :, P:, :], 'b h t d -> b t (h d)')

        pool_attn = self.pool_out(pool_attn)
        x_attn = self.x_out(x_attn)
        pool = self.out_drop_pool(pool_attn)
        x = self.out_drop_x(x_attn)
        # pool = pool + pool_attn
        # x = x + x_attn

        pool = pool * n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P, 1).float()
        x = x * t_mask.unsqueeze(-1).float()

        return pool, x


class PJAC(nn.Module):
    def __init__(
            self, dim,
            num_heads,
            head_dim,
            c_kernel_size_pool=7,
            m_kernel_size_pool=5,
            c_kernel_size_x=31,
            m_kernel_size_x=31,

            c_out_drop_x=0.1,
            c_latent_drop_x=0.0,
            c_out_drop_pool=0.1,
            c_latent_drop_pool=0.0,

            region_token_num=1,
            qk_norm=True,
            use_rope=True,
            rope_mode='mixed',
            use_pool_offset=False,
            theta=10000.0,
            dropout_attn: float = 0.0,
            attn_out_drop_x: float = 0.0,
            attn_out_drop_pool: float = 0.0,
    ):
        super().__init__()
        self.jattn = JoinAtention(dim=dim, num_heads=num_heads, region_token_num=region_token_num, qk_norm=qk_norm,
                                  use_rope=use_rope, rope_mode=rope_mode, use_pool_offset=use_pool_offset, theta=theta,
                                  dropout_attn=dropout_attn, out_drop_x=attn_out_drop_x,
                                  out_drop_pool=attn_out_drop_pool, head_dim=head_dim)

        self.c_x = CgMLP(
            dim, kernel_size=c_kernel_size_x,
            latent_drop=c_latent_drop_x, out_drop=c_out_drop_x
        )
        self.c_pool = CgMLP(
            dim, kernel_size=c_kernel_size_pool,
            latent_drop=c_latent_drop_pool, out_drop=c_out_drop_pool
        )

        self.c_norm_x = RMSnorm(dim)
        self.c_norm_pool = RMSnorm(dim)

        self.merge_linear_x = nn.Linear(dim * 2, dim)
        self.merge_dw_conv_x = (
            nn.Conv1d(
                dim * 2, dim * 2, kernel_size=m_kernel_size_x, stride=1,
                padding=m_kernel_size_x // 2,
                groups=dim * 2
            )
            if m_kernel_size_x != 0 else
            None
        )
        self.merge_linear_pool = nn.Linear(dim * 2, dim)
        self.merge_dw_conv_pool = (
            nn.Conv1d(
                dim * 2, dim * 2, kernel_size=m_kernel_size_pool, stride=1,
                padding=m_kernel_size_pool // 2,
                groups=dim * 2
            )
            if m_kernel_size_pool != 0 else
            None
        )

    def forward(self, pool, x, regions, t_mask, n_mask, attn_mask, max_n=None):

        a_pool, a_x = self.jattn(pool, x, regions, t_mask, n_mask, attn_mask, max_n)
        c_pool, c_x = self.c_pool(self.c_norm_pool(pool)), self.c_x(self.c_norm_x(x))
        m_pool, m_x = torch.cat([a_pool, c_pool], dim=-1), torch.cat([a_x, c_x], dim=-1)
        if self.merge_dw_conv_pool is not None:
            m_pool = self.merge_dw_conv_pool(m_pool.transpose(1, 2)).transpose(1, 2) + m_pool
        m_pool = self.merge_linear_pool(m_pool)
        if self.merge_dw_conv_x is not None:
            m_x = self.merge_dw_conv_x(m_x.transpose(1, 2)).transpose(1, 2) + m_x
        m_x = self.merge_linear_x(m_x)
        return m_pool, m_x


class JEBF(nn.Module):
    def __init__(
            self, dim,
            num_heads,
            head_dim,
            c_kernel_size_pool=7,
            m_kernel_size_pool=5,
            c_kernel_size_x=31,
            m_kernel_size_x=31,

            c_out_drop_x=0.1,
            c_latent_drop_x=0.0,
            c_out_drop_pool=0.1,
            c_latent_drop_pool=0.0,

            region_token_num=1,
            qk_norm=True,
            use_rope=True,
            rope_mode='mixed',
            use_pool_offset=False,
            theta=10000.0,
            dropout_attn: float = 0.0,
            attn_out_drop_x: float = 0.0,
            attn_out_drop_pool: float = 0.0,

            use_ls=True, ffn_type='glu', ffn_latent_drop=0.1, ffn_out_drop=0.1, skip_fist_ffn=False
    ):
        super().__init__()
        self.skip_fist_ffn = skip_fist_ffn
        if ffn_type == 'glu':
            if not skip_fist_ffn:
                self.ffn1_x = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
                self.ffn1_pool = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
            self.ffn2_x = GLUFFN(
                dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                dropout_output=ffn_out_drop
            )
            self.ffn2_pool = GLUFFN(
                dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                dropout_output=ffn_out_drop
            )
        elif ffn_type == 'ffn':
            if not skip_fist_ffn:
                self.ffn1_x = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
                self.ffn1_pool = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
            self.ffn2_x = FFN(
                dim, latent_dim=dim * 4,
                dropout_latent=ffn_latent_drop,
                dropout_output=ffn_out_drop
            )
            self.ffn2_pool = FFN(
                dim, latent_dim=dim * 4,
                dropout_latent=ffn_latent_drop,
                dropout_output=ffn_out_drop
            )
        elif ffn_type == 'cgmlp':
            if not skip_fist_ffn:
                self.ffn1_x = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=21
                )
                self.ffn1_pool = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=21
                )
            self.ffn2_x = CgMLP(
                dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                out_drop=ffn_out_drop, kernel_size=7
            )
            self.ffn2_pool = CgMLP(
                dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                out_drop=ffn_out_drop, kernel_size=7
            )
        elif ffn_type == 'eglu':
            if not skip_fist_ffn:
                self.ffn1_x = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
                self.ffn1_pool = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
            self.ffn2_x = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
            self.ffn2_pool = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)

        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.attn = PJAC(dim=dim, num_heads=num_heads, head_dim=head_dim, c_kernel_size_pool=c_kernel_size_pool,
                         m_kernel_size_pool=m_kernel_size_pool, c_kernel_size_x=c_kernel_size_x,
                         m_kernel_size_x=m_kernel_size_x, c_out_drop_x=c_out_drop_x, c_latent_drop_x=c_latent_drop_x,
                         c_out_drop_pool=c_out_drop_pool, c_latent_drop_pool=c_latent_drop_pool,
                         region_token_num=region_token_num, qk_norm=qk_norm, use_rope=use_rope, rope_mode=rope_mode,
                         use_pool_offset=use_pool_offset, theta=theta, dropout_attn=dropout_attn,
                         attn_out_drop_x=attn_out_drop_x, attn_out_drop_pool=attn_out_drop_pool)
        if not skip_fist_ffn:
            self.norm_ffn1_x = RMSnorm(dim)
            self.norm_ffn1_pool = RMSnorm(dim)
        self.norm_ffn2_x = RMSnorm(dim)
        self.norm_ffn2_pool = RMSnorm(dim)

        if use_ls:
            self.lay_scale_ffn2_x = LayScale(dim)
            self.lay_scale_ffn2_pool = LayScale(dim)
            if not skip_fist_ffn:
                self.lay_scale_ffn1_x = LayScale(dim)
                self.lay_scale_ffn1_pool = LayScale(dim)

            self.lay_scale_jpac_x = LayScale(dim)
            self.lay_scale_jpac_pool = LayScale(dim)

        else:
            self.lay_scale_ffn2_x = nn.Identity()
            self.lay_scale_ffn2_pool = nn.Identity()
            if not skip_fist_ffn:
                self.lay_scale_ffn1_x = nn.Identity()
                self.lay_scale_ffn1_pool = nn.Identity()

            self.lay_scale_jpac_x = nn.Identity()
            self.lay_scale_jpac_pool =nn.Identity()

    def forward(self, pool, x, regions, t_mask, n_mask, attn_mask, max_n=None):
        # Expand n_mask to match pool shape: [B, N] -> [B, N*R]
        B, N = n_mask.shape
        R = pool.shape[1] // N
        pool_mask = n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, N * R)  # [B, N*R]
        
        if t_mask is not None:
            x = x.masked_fill(~t_mask.unsqueeze(-1), 0)
        if n_mask is not None:
            pool = pool.masked_fill(~pool_mask.unsqueeze(-1), 0)

        if not self.skip_fist_ffn:
            x = self.lay_scale_ffn1_x(self.ffn1_x(self.norm_ffn1_x(x)))  + x
            pool=self.lay_scale_ffn1_pool(self.ffn1_pool(self.norm_ffn1_pool(pool))) + pool

        if t_mask is not None:
            x = x.masked_fill(~t_mask.unsqueeze(-1), 0)
        if n_mask is not None:
            pool = pool.masked_fill(~pool_mask.unsqueeze(-1), 0)

        p_o,x_o=self.attn(pool,x,regions,t_mask,n_mask,attn_mask,max_n)
        x=self.lay_scale_jpac_x(x_o)+x
        pool=self.lay_scale_jpac_pool(p_o)+pool

        if t_mask is not None:
            x = x.masked_fill(~t_mask.unsqueeze(-1), 0)
        if n_mask is not None:
            pool = pool.masked_fill(~pool_mask.unsqueeze(-1), 0)

        x=self.lay_scale_ffn2_x(self.ffn2_x(self.norm_ffn2_x(x))) + x
        pool=self.lay_scale_ffn2_pool(self.ffn2_pool(self.norm_ffn2_pool(pool))) + pool

        return x,pool


# ============================================================
# Pool Token Generators
# ============================================================

class LearnablePoolTokens(nn.Module):
    """Learnable pool tokens per region."""
    def __init__(self, dim, region_token_num=1):
        super().__init__()
        self.region_token_num = region_token_num
        self.emb = nn.Parameter(torch.randn(region_token_num, dim) * 0.02)

    def forward(self, x, regions, max_n, n_mask):
        """
        :param x: [B, T, C] (unused, for interface compatibility)
        :param regions: [B, T] (unused)
        :param max_n: int
        :param n_mask: [B, N] bool
        :return: [B, N * R, C]
        """
        B, N = n_mask.shape
        R = self.region_token_num
        tokens = self.emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, R, C]
        tokens = tokens * n_mask.unsqueeze(-1).unsqueeze(-1).float()
        return tokens.reshape(B, N * R, -1)


class LocalAvgPoolTokens(nn.Module):
    """Local average pooling per region."""
    def __init__(self, dim, region_token_num=1):
        super().__init__()
        self.region_token_num = region_token_num
        # Project to match region_token_num if > 1
        if region_token_num > 1:
            self.expand_proj = nn.Linear(dim, dim * region_token_num)
        else:
            self.expand_proj = None

    def forward(self, x, regions, max_n, n_mask):
        """
        :param x: [B, T, C]
        :param regions: [B, T]
        :param max_n: int
        :param n_mask: [B, N] bool
        :return: [B, N * R, C]
        """
        B, T, C = x.shape
        R = self.region_token_num
        device = x.device

        # Compute local average per region
        idx = torch.arange(max_n + 1, dtype=torch.long, device=device).reshape(1, -1, 1)  # [1, N+1, 1]
        region_map = idx == regions.unsqueeze(-2)  # [B, N+1, T]
        region_weight = region_map.float()
        region_size = torch.where(
            torch.any(region_map, dim=-1, keepdim=True),
            region_weight.sum(dim=-1, keepdim=True),
            1.0
        )
        weight = region_weight / region_size  # [B, N+1, T]
        weight = weight[:, 1:, :]  # [B, N, T]
        tokens = weight @ x  # [B, N, C]

        # Expand to R tokens per region
        if self.expand_proj is not None:
            tokens = self.expand_proj(tokens)  # [B, N, C*R]
            tokens = tokens.reshape(B, max_n, R, C)
        else:
            tokens = tokens.unsqueeze(2)  # [B, N, 1, C]

        tokens = tokens * n_mask.unsqueeze(-1).unsqueeze(-1).float()
        return tokens.reshape(B, max_n * R, C)


# ============================================================
# JEBFBackbone
# ============================================================

class JEBFBackbone(nn.Module): #todo 其他的到时候再说
    """
    JEBF Backbone with joint attention between pool tokens and x.
    Internally generates pool tokens and attention mask.
    
    Input: x, regions, t_mask, n_mask, max_n=None
    Output: out (x), pool (downsampled)
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            dim: int = 256,
            num_layers: int = 8,
            num_heads: int = 8,
            head_dim: int = 64,
            region_token_num: int = 1,
            pool_token_source: str = 'learnable',  # 'learnable' or 'local_avg'
            
            # JEBF layer params
            c_kernel_size_pool: int = 7,
            m_kernel_size_pool: int = 5,
            c_kernel_size_x: int = 31,
            m_kernel_size_x: int = 31,
            c_out_drop_x: float = 0.1,
            c_latent_drop_x: float = 0.0,
            c_out_drop_pool: float = 0.1,
            c_latent_drop_pool: float = 0.0,
            qk_norm: bool = True,
            use_rope: bool = True,
            rope_mode: str = 'mixed',
            use_pool_offset: bool = False,
            theta: float = 10000.0,
            dropout_attn: float = 0.0,
            attn_out_drop_x: float = 0.0,
            attn_out_drop_pool: float = 0.0,
            use_ls: bool = True,
            ffn_type: str = 'glu',
            ffn_latent_drop: float = 0.1,
            ffn_out_drop: float = 0.1,
            
            use_out_norm: bool = True,
            skip_fist_ffn=False,
            pool_out_dim: int = None,
    ):
        super().__init__()
        self.region_token_num = region_token_num
        self.use_out_norm = use_out_norm
        self.pool_out_dim = pool_out_dim if pool_out_dim is not None else out_dim

        # Input projection
        self.input_proj = nn.Linear(in_dim, dim)

        # Pool token generator
        if pool_token_source == 'learnable':
            self.pool_token_gen = LearnablePoolTokens(dim, region_token_num)
        elif pool_token_source == 'local_avg':
            self.pool_token_gen = LocalAvgPoolTokens(dim, region_token_num)
        else:
            raise ValueError(f"Unknown pool_token_source: {pool_token_source}")

        # JEBF layers
        self.layers = nn.ModuleList([
            JEBF(
                dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size_pool=c_kernel_size_pool, m_kernel_size_pool=m_kernel_size_pool,
                c_kernel_size_x=c_kernel_size_x, m_kernel_size_x=m_kernel_size_x,
                c_out_drop_x=c_out_drop_x, c_latent_drop_x=c_latent_drop_x,
                c_out_drop_pool=c_out_drop_pool, c_latent_drop_pool=c_latent_drop_pool,
                region_token_num=region_token_num, qk_norm=qk_norm,
                use_rope=use_rope, rope_mode=rope_mode,
                use_pool_offset=use_pool_offset, theta=theta,
                dropout_attn=dropout_attn, attn_out_drop_x=attn_out_drop_x,
                attn_out_drop_pool=attn_out_drop_pool,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_fist_ffn=skip_fist_ffn,  # skip first FFN for first layer #todo
            )
            for i in range(num_layers)
        ])

        # Output norms and projections
        if self.use_out_norm:
            self.output_norm_x = RMSnorm(dim)
            self.output_norm_pool = RMSnorm(dim)
        self.output_proj_x = nn.Linear(dim, out_dim)
        self.output_proj_pool = nn.Linear(dim, self.pool_out_dim)

    def forward(self, x, regions, t_mask, n_mask, max_n=None):
        """
        Args:
            x: [B, T, in_dim] input tensor
            regions: [B, T] region indices (0=padding, 1~N=valid regions)
            t_mask: [B, T] valid mask for x
            n_mask: [B, N] valid mask for regions
            max_n: int, max number of regions (default: n_mask.shape[1])
        
        Returns:
            out_x: [B, T, out_dim] output tensor
            out_pool: [B, N*R, pool_out_dim] pooled output
        """
        if max_n is None:
            max_n = n_mask.shape[1]

        B, T, _ = x.shape
        R = self.region_token_num
        P = max_n * R

        # Input projection
        x = self.input_proj(x)

        # Generate pool tokens
        pool = self.pool_token_gen(x, regions, max_n, n_mask)  # [B, P, dim]

        # Build attention mask
        attn_mask = build_join_attention_mask(regions, R, max_n, t_mask, n_mask)  # [B, 1, P+T, P+T]

        # JEBF layers
        for layer in self.layers:
            x, pool = layer(pool, x, regions, t_mask, n_mask, attn_mask, max_n)

        # Output projection
        if self.use_out_norm:
            x = self.output_norm_x(x)
            pool = self.output_norm_pool(pool)
        
        out_x = self.output_proj_x(x)  # [B, T, out_dim]
        out_pool = self.output_proj_pool(pool)  # [B, P, pool_out_dim]

        return out_x, out_pool


