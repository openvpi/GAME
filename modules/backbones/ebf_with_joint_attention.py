import torch
import torch.nn as nn

from modules.backbones.eglu import HalfCacheGLUFFN
from modules.backbones.joint_attn import JointAttention, SplitJointAttention, build_joint_attention_mask, \
    build_split_attention_masks
from modules.backbones.layers import LayerScale, RMSNorm, GLUFFN, FFN, CgMLP
from modules.backbones.pool_tokens import LearnablePoolTokens, PoolTokenMerger
from modules.backbones.regions import RegionBias


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
