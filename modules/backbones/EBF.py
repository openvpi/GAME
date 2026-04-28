import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from deployment.context import is_export_mode
from modules.backbones.eglu import HalfCacheGLUFFN
from modules.backbones.layers import LayerScale, RMSNorm, GLUFFN, FFN, CgMLP
from modules.backbones.rope import SingleRoPosEmb


class AttentionWithRoPE(nn.Module):
    def __init__(
            self, dim, num_heads, head_dim,
            use_rope=True, rope_cache=True,
            dropout_attn: float = 0.0,
            out_drop: float = 0.0
    ):
        super().__init__()

        self.num_heads = num_heads
        attn_dim = head_dim * num_heads
        self.q_linear = nn.Linear(dim, out_features=attn_dim, bias=True)
        self.kv_linear = nn.Linear(dim, out_features=attn_dim * 2, bias=True)

        self.out_linear = nn.Linear(attn_dim, dim, bias=True)
        self.dropout_attn = dropout_attn
        self.out_drop = nn.Dropout(out_drop) if out_drop > 0. else nn.Identity()
        if use_rope:
            self.rope = SingleRoPosEmb(head_dim, use_cache=rope_cache)
        else:
            self.rope = None

    def forward(self, x):

        q = self.q_linear(x)

        k, v = self.kv_linear(x).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.num_heads), (q, k, v)
        )
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        with torch.backends.cuda.sdp_kernel():
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_attn,
            )

        out = rearrange(out, "b h t c -> b t (h c) ", h=self.num_heads, )
        out = self.out_linear(out)
        out = self.out_drop(out)
        return out


class PAC(nn.Module):
    def __init__(
            self, dim, num_heads, head_dim,
            c_kernel_size=31, m_kernel_size=31, use_rope=True, rope_cache=True,
            dropout_attn: float = 0.0, out_drop: float = 0.0, c_out_drop=0.1,
            c_latent_drop=0.0,
    ):
        super().__init__()
        self.attn = AttentionWithRoPE(dim, num_heads, head_dim, use_rope, rope_cache, dropout_attn, out_drop)
        self.c = CgMLP(
            dim, kernel_size=c_kernel_size,
            latent_drop=c_latent_drop, out_drop=c_out_drop
        )

        self.a_norm = RMSNorm(dim)
        self.c_norm = RMSNorm(dim)

        self.merge_linear = nn.Linear(dim * 2, dim)
        self.merge_dw_conv = (
            nn.Conv1d(
                dim * 2, dim * 2, kernel_size=m_kernel_size, stride=1,
                padding=m_kernel_size // 2,
                groups=dim * 2
            )
            if m_kernel_size != 0 else
            None
        )

    def forward(self, x):
        a_o = self.attn(self.a_norm(x))
        c_o = self.c(self.c_norm(x))
        m_o = torch.cat([a_o, c_o], dim=-1)

        if self.merge_dw_conv is not None:
            m_o = self.merge_dw_conv(m_o.transpose(1, 2)).transpose(1, 2) + m_o
        m_o = self.merge_linear(m_o)
        return m_o


class EBF(nn.Module):
    def __init__(
            self, dim, num_heads, head_dim,
            c_kernel_size=31, m_kernel_size=31, use_rope=True, rope_cache=True,
            dropout_attn: float = 0.0, out_drop: float = 0.0, c_out_drop=0.1,
            c_latent_drop=0.0, use_ls=True, ffn_type='glu', ffn_latent_drop=0.1, ffn_out_drop=0.1,
            skip_first_ffn=False, skip_out_ffn=False,
    ):
        super().__init__()
        self.skip_first_ffn = skip_first_ffn
        self.skip_out_ffn = skip_out_ffn

        if ffn_type == 'glu':
            if not skip_first_ffn:
                self.ffn1 = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
            if not skip_out_ffn:
                self.ffn2 = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
        elif ffn_type == 'ffn':
            if not skip_first_ffn:
                self.ffn1 = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
            if not skip_out_ffn:
                self.ffn2 = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
        elif ffn_type == 'cgmlp':
            if not skip_first_ffn:
                self.ffn1 = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=21
                )
            if not skip_out_ffn:
                self.ffn2 = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=7
                )
        elif ffn_type == 'eglu':
            if not skip_first_ffn:
                self.ffn1 = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
            if not skip_out_ffn:
                self.ffn2 = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)

        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.attn = PAC(
            dim, num_heads, head_dim, c_kernel_size, m_kernel_size, use_rope, rope_cache, dropout_attn,
            out_drop, c_out_drop, c_latent_drop
        )
        if not skip_first_ffn:
            self.norm1 = RMSNorm(dim)
        if not skip_out_ffn:
            self.norm2 = RMSNorm(dim)

        if use_ls:
            if not skip_first_ffn:
                self.lay_scale1 = LayerScale(dim)
            self.lay_scale2 = LayerScale(dim)
            if not skip_out_ffn:
                self.lay_scale3 = LayerScale(dim)
        else:
            if not skip_first_ffn:
                self.lay_scale1 = nn.Identity()
            self.lay_scale2 = nn.Identity()
            if not skip_out_ffn:
                self.lay_scale3 = nn.Identity()

    def forward(self, x, mask=None):
        if not self.skip_first_ffn:
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
            x = self.lay_scale1(self.ffn1(self.norm1(x))) * 0.5 + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.lay_scale2(self.attn(x)) + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        if not self.skip_out_ffn:
            x = self.lay_scale3(self.ffn2(self.norm2(x))) * 0.5 + x
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
        return x


class EBFBackbone(nn.Module):
    def __init__(
            self, in_dim: int, out_dim: int, return_latent: bool,
            dim: int = 256,
            num_layers: int = 8,
            latent_layer_idx: int = None,
            latent_out_dim: int = 16,
            num_heads: int = 8,
            head_dim: int = 64,
            c_kernel_size: int = 31,
            m_kernel_size: int = 31,
            use_rope: bool = True,
            rope_cache: bool = True,
            dropout_attn: float = 0.0,
            out_drop: float = 0.0,
            c_out_drop: float = 0.1,
            c_latent_drop: float = 0.0,
            use_ls: bool = True,
            ffn_type: str = 'glu',
            ffn_latent_drop: float = 0.1,
            ffn_out_drop: float = 0.1,
            use_out_norm: bool = True,
            skip_first_ffn=False,
            skip_out_ffn=False,
    ):
        super().__init__()
        if is_export_mode():
            rope_cache = False

        self.use_out_norm = use_out_norm
        self.return_latent = return_latent
        if return_latent:
            assert latent_layer_idx <= num_layers
        self.latent_layer_idx = latent_layer_idx

        self.input_proj = nn.Linear(in_dim, dim)

        self.layers = nn.ModuleList([
            EBF(dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size=c_kernel_size, m_kernel_size=m_kernel_size,
                use_rope=use_rope, rope_cache=rope_cache,
                dropout_attn=dropout_attn, out_drop=out_drop,
                c_out_drop=c_out_drop, c_latent_drop=c_latent_drop,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_first_ffn=skip_first_ffn, skip_out_ffn=skip_out_ffn,
                )
            for _ in range(num_layers)
        ])

        if self.return_latent:
            self.latent_norm = RMSNorm(dim)
            self.latent_proj = nn.Linear(dim, latent_out_dim)  # -> [B, T, C_latent]
        if self.use_out_norm:
            self.output_norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, out_dim)  # -> [B, T, C_out]

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, in_dim] input tensor
            mask: [B, T] valid mask
        Returns:
            latent: [B, T, C_latent] intermediate latent tensor for self cosine similarity
            out: [B, T, C_out] output tensor
        """
        x = self.input_proj(x)

        latent = None
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask)
            if self.return_latent and i == self.latent_layer_idx - 1:
                latent = self.latent_norm(x)
                latent = self.latent_proj(latent)  # [B, T, C_latent]

        if self.use_out_norm:
            x = self.output_norm(x)
        out = self.output_proj(x)  # [B, T, C_out]

        if self.return_latent:
            return out, latent
        else:
            return out
