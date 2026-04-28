import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from deployment.context import is_export_mode
from modules.backbones.eglu import HalfCacheGLUFFN
from modules.backbones.layers import LayerScale, RMSNorm, GLUFFN, FFN, CgMLP
from modules.backbones.pool_tokens import LearnablePoolTokens, PoolTokenMerger
from modules.backbones.regions import (
    build_joint_attention_mask,
    compute_positions_local,
    build_split_attention_masks
)
from modules.backbones.rope import RegionRoPE


def fill_with_attn_mask(x, attn_mask):
    # In some buggy ONNX exporting logic, fully masked queries may produce NaN.
    # We fix this by manually filling zeros to match the PyTorch native behavior.
    if is_export_mode():
        x = torch.where(attn_mask.any(dim=-2).unsqueeze(-1), x, 0)
    return x


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

        # Region bias for cross-stream attention

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

        # Build or use pre-built masks

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
            attn_type: str = 'joint',
    ):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == 'joint':
            self.jattn = JointAttention(
                dim=dim, num_heads=num_heads, region_token_num=region_token_num,
                qk_norm=qk_norm,
                use_rope=use_rope, rope_mode=rope_mode, use_pool_offset=use_pool_offset,
                theta=theta,
                dropout_attn=dropout_attn, out_drop_x=attn_out_drop_x,
                out_drop_pool=attn_out_drop_pool, head_dim=head_dim
            )
        elif attn_type == 'split':
            self.jattn = SplitJointAttention(
                dim=dim, num_heads=num_heads, region_token_num=region_token_num,
                qk_norm=qk_norm,
                use_rope=use_rope, rope_mode=rope_mode, use_pool_offset=use_pool_offset,
                theta=theta,
                dropout_attn=dropout_attn, out_drop_x=attn_out_drop_x,
                out_drop_pool=attn_out_drop_pool, head_dim=head_dim,
            )
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.c_x = CgMLP(
            dim, kernel_size=c_kernel_size_x,
            latent_drop=c_latent_drop_x, out_drop=c_out_drop_x
        )
        self.c_pool = CgMLP(
            dim, kernel_size=c_kernel_size_pool,
            latent_drop=c_latent_drop_pool, out_drop=c_out_drop_pool
        )

        self.c_norm_x = RMSNorm(dim)
        self.c_norm_pool = RMSNorm(dim)

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

    def forward(self, pool, x, regions, t_mask, n_mask, attn_mask):
        a_pool, a_x = self.jattn(pool, x, regions, t_mask, n_mask, attn_mask)
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
            skip_first_ffn=False, skip_out_ffn=False,
            use_ls=True, ffn_type='glu', ffn_latent_drop=0.1, ffn_out_drop=0.1,
            attn_type: str = 'joint',

    ):
        super().__init__()
        self.skip_first_ffn = skip_first_ffn
        self.skip_out_ffn = skip_out_ffn
        if ffn_type == 'glu':
            if not skip_first_ffn:
                self.ffn1_x = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
                self.ffn1_pool = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
            if not skip_out_ffn:
                self.ffn2_x = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
                self.ffn2_pool = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
        elif ffn_type == 'ffn':
            if not skip_first_ffn:
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
            if not skip_out_ffn:
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
            if not skip_first_ffn:
                self.ffn1_x = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=21
                )
                self.ffn1_pool = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=21
                )
            if not skip_out_ffn:
                self.ffn2_x = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=7
                )
                self.ffn2_pool = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=7
                )
        elif ffn_type == 'eglu':
            if not skip_first_ffn:
                self.ffn1_x = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
                self.ffn1_pool = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)

            if not skip_out_ffn:
                self.ffn2_x = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
                self.ffn2_pool = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)

        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.attn = PJAC(
            dim=dim, num_heads=num_heads, head_dim=head_dim, c_kernel_size_pool=c_kernel_size_pool,
            m_kernel_size_pool=m_kernel_size_pool, c_kernel_size_x=c_kernel_size_x,
            m_kernel_size_x=m_kernel_size_x, c_out_drop_x=c_out_drop_x, c_latent_drop_x=c_latent_drop_x,
            c_out_drop_pool=c_out_drop_pool, c_latent_drop_pool=c_latent_drop_pool,
            region_token_num=region_token_num, qk_norm=qk_norm, use_rope=use_rope, rope_mode=rope_mode,
            use_pool_offset=use_pool_offset, theta=theta, dropout_attn=dropout_attn,
            attn_out_drop_x=attn_out_drop_x, attn_out_drop_pool=attn_out_drop_pool,
            attn_type=attn_type,
        )
        if not skip_first_ffn:
            self.norm_ffn1_x = RMSNorm(dim)
            self.norm_ffn1_pool = RMSNorm(dim)
        if not skip_out_ffn:
            self.norm_ffn2_x = RMSNorm(dim)
            self.norm_ffn2_pool = RMSNorm(dim)

        if use_ls:
            if not skip_out_ffn:
                self.lay_scale_ffn2_x = LayerScale(dim)
                self.lay_scale_ffn2_pool = LayerScale(dim)
            if not skip_first_ffn:
                self.lay_scale_ffn1_x = LayerScale(dim)
                self.lay_scale_ffn1_pool = LayerScale(dim)

            self.lay_scale_jpac_x = LayerScale(dim)
            self.lay_scale_jpac_pool = LayerScale(dim)

        else:
            if not skip_out_ffn:
                self.lay_scale_ffn2_x = nn.Identity()
                self.lay_scale_ffn2_pool = nn.Identity()
            if not skip_first_ffn:
                self.lay_scale_ffn1_x = nn.Identity()
                self.lay_scale_ffn1_pool = nn.Identity()

            self.lay_scale_jpac_x = nn.Identity()
            self.lay_scale_jpac_pool = nn.Identity()

    def forward(self, pool, x, regions, t_mask, n_mask, attn_mask):
        # Expand n_mask to match pool shape: [B, N] -> [B, N*R]
        B, N = n_mask.shape
        R = pool.shape[1] // N
        pool_mask = n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, N * R)  # [B, N*R]

        if t_mask is not None:
            x = x.masked_fill(~t_mask.unsqueeze(-1), 0)
        if n_mask is not None:
            pool = pool.masked_fill(~pool_mask.unsqueeze(-1), 0)

        if not self.skip_first_ffn:
            x = self.lay_scale_ffn1_x(self.ffn1_x(self.norm_ffn1_x(x))) + x
            pool = self.lay_scale_ffn1_pool(self.ffn1_pool(self.norm_ffn1_pool(pool))) + pool

        if t_mask is not None:
            x = x.masked_fill(~t_mask.unsqueeze(-1), 0)
        if n_mask is not None:
            pool = pool.masked_fill(~pool_mask.unsqueeze(-1), 0)

        p_o, x_o = self.attn(pool, x, regions, t_mask, n_mask, attn_mask)
        x = self.lay_scale_jpac_x(x_o) + x
        pool = self.lay_scale_jpac_pool(p_o) + pool

        if t_mask is not None:
            x = x.masked_fill(~t_mask.unsqueeze(-1), 0)
        if n_mask is not None:
            pool = pool.masked_fill(~pool_mask.unsqueeze(-1), 0)
        if not self.skip_out_ffn:
            x = self.lay_scale_ffn2_x(self.ffn2_x(self.norm_ffn2_x(x))) + x
            pool = self.lay_scale_ffn2_pool(self.ffn2_pool(self.norm_ffn2_pool(pool))) + pool

        return x, pool


# ============================================================
# JEBFBackbone
# ============================================================

class JEBFBackbone(nn.Module):
    """
    JEBF Backbone with joint attention between pool tokens and x.
    Internally generates pool tokens and attention mask.
    
    Input: x, regions, t_mask, n_mask,
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
            skip_first_ffn=False,
            skip_out_ffn=False,
            pool_out_dim: int = None,
            attn_type: str = 'joint',  # split
            use_region_bias: bool = False,
            bias_alpha: float = 2.35,
            bias_learnable: bool = False,
            pool_merge_mode: str = 'mean',  # 'mean', 'max', 'first', 'learned', 'attention'
            pool_merge_heads: int = 4,  # only for attention mode
    ):
        super().__init__()
        self.region_token_num = region_token_num
        self.use_out_norm = use_out_norm
        self.pool_out_dim = pool_out_dim if pool_out_dim is not None else out_dim
        self.attn_type = attn_type
        self.use_region_bias = use_region_bias
        if use_region_bias:
            self.region_bias = RegionBias(alpha=bias_alpha, learnable=bias_learnable)

        # Pool token merger (only when R > 1)
        if region_token_num > 1:
            self.pool_merger = PoolTokenMerger(dim, region_token_num, mode=pool_merge_mode, num_heads=pool_merge_heads)
        else:
            self.pool_merger = None

        # Input projection
        self.input_proj = nn.Linear(in_dim, dim)

        # Pool token generator
        self.pool_token_gen = LearnablePoolTokens(dim, region_token_num)

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
                skip_first_ffn=skip_first_ffn,
                attn_type=attn_type, skip_out_ffn=skip_out_ffn
            )
            for _ in range(num_layers)
        ])

        # Output norms and projections
        if self.use_out_norm:
            self.output_norm_x = RMSNorm(dim)
            self.output_norm_pool = RMSNorm(dim)
        self.output_proj_x = nn.Linear(dim, out_dim)
        self.output_proj_pool = nn.Linear(dim, self.pool_out_dim)

    def _build_region_bias_mask(self, regions, region_token_num, t_mask, n_mask):
        """
        Build float attention mask with region bias for cross-stream attention.
        Same-stream: 0.0 (or -10000 for invalid)
        Cross-stream: region_bias decay (or -10000 for invalid)
        """
        B, T = regions.shape
        N = n_mask.shape[1]
        R = region_token_num
        P = N * R
        device = regions.device

        # Pool tokens region indices
        pool_region = torch.arange(1, N + 1, device=device) \
            .unsqueeze(-1).expand(-1, R).reshape(1, P).expand(B, -1)
        full_region = torch.cat([pool_region, regions], dim=-1)  # [B, P+T]

        # Valid masks
        pool_valid = n_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, P)
        full_valid = torch.cat([pool_valid, t_mask], dim=-1)  # [B, P+T]

        # Is pool token
        is_pool = torch.cat([
            torch.ones(B, P, device=device, dtype=torch.bool),
            torch.zeros(B, T, device=device, dtype=torch.bool),
        ], dim=-1)  # [B, P+T]

        # Same stream mask
        same_stream = is_pool.unsqueeze(-1) == is_pool.unsqueeze(-2)  # [B, P+T, P+T]

        # Valid pairs
        valid_pair = full_valid.unsqueeze(-1) & full_valid.unsqueeze(-2)  # [B, P+T, P+T]

        # Base mask: -10000 for invalid, 0 for valid
        base_mask = torch.where(valid_pair, 0.0, -10000.0)  # [B, P+T, P+T]

        # Region bias for cross-stream
        region_decay = self.region_bias(full_region, full_region)  # [B, 1, P+T, P+T]
        region_decay = region_decay.squeeze(1)  # [B, P+T, P+T]

        # Apply region bias only to cross-stream (different stream)
        # Same-stream: keep 0, Cross-stream: add region decay
        attn_bias = torch.where(same_stream, base_mask, base_mask + region_decay)

        return attn_bias.unsqueeze(1)  # [B, 1, P+T, P+T]

    def forward(self, x, regions, t_mask, n_mask, ):
        """
        Args:
            x: [B, T, in_dim] input tensor
            regions: [B, T] region indices (0=padding, 1~N=valid regions)
            t_mask: [B, T] valid mask for x
            n_mask: [B, N] valid mask for regions

        Returns:
            out_x: [B, T, out_dim] output tensor
            out_pool: [B, N, pool_out_dim] pooled output (merged if R>1, else [B, N*R, pool_out_dim])
        """

        R = self.region_token_num

        # Input projection
        x = self.input_proj(x)

        # Generate pool tokens
        pool = self.pool_token_gen(n_mask)  # [B, P, dim]

        # Build attention mask

        if self.attn_type == 'joint':
            # Use region bias (soft) or hard mask
            if self.use_region_bias:
                attn_mask = self._build_region_bias_mask(regions, R, t_mask, n_mask)
            else:
                attn_mask = build_joint_attention_mask(regions, R, t_mask, n_mask)  # [B, 1, P+T, P+T]
        elif self.attn_type == 'split':

            region_bias = self.region_bias if self.use_region_bias else None
            attn_mask = build_split_attention_masks(
                regions, R, t_mask, n_mask, region_bias
            )

        else:
            raise ValueError(f"Unknown attn_type: {self.attn_type}")

        # JEBF layers
        for layer in self.layers:
            x, pool = layer(pool, x, regions, t_mask, n_mask, attn_mask)

        # Output projection
        if self.use_out_norm:
            x = self.output_norm_x(x)
            pool = self.output_norm_pool(pool)

        # Merge pool tokens if R > 1
        if self.pool_merger is not None:
            pool = self.pool_merger(pool, n_mask)  # [B, N*R, dim] -> [B, N, dim]

        out_x = self.output_proj_x(x)  # [B, T, out_dim]
        out_pool = self.output_proj_pool(pool)  # [B, N, pool_out_dim] (merged) or [B, N*R, pool_out_dim]

        return out_x, out_pool
