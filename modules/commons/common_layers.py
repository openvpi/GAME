import torch
import torch.onnx.operators
from einops import rearrange
from torch import nn

from modules.commons.local_down_layer import LocalAttentionPoolDown, AttentionPoolDown, CrossAttentionDown, \
    LocalCrossAttentionDown, LearnablePoolTokens


class CyclicRegionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, cycle_length: int = 3):
        super().__init__()
        self.cycle_length = cycle_length
        self.embedding = nn.Embedding(cycle_length, embedding_dim)

    def forward(self, idx):
        if self.training:
            *B, _ = idx.shape
            shift = torch.randint(0, self.cycle_length, (*B, 1)).to(idx)
            idx = idx + shift
        return self.embedding(idx % self.cycle_length)


class LocalDownsample(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x, regions, max_n: int = None):
        """
        :param x: [..., T, C] input tensor to downsample
        :param regions: int64 [..., T] mapping from positions to region indices starting from 1.
        :param max_n: int, maximum number of regions. N = max(regions) if not given.
        :return: [..., N, C] where N = max(regions)
        """
        N = regions.max() if max_n is None else max_n
        B = (1,) * (x.ndim - 2)
        idx = torch.arange(N + 1, dtype=torch.long, device=regions.device).reshape(*B, -1, 1)  # [..., N+1, 1]
        region_map = idx == regions.unsqueeze(-2)  # [..., N, T]
        region_weight = region_map.float()
        region_size = torch.where(
            torch.any(region_map, dim=-1, keepdim=True),
            region_weight.sum(dim=-1, keepdim=True),
            1.0
        )  # [..., N, 1]
        weight = region_weight / region_size  # [..., N+1, T]
        weight = weight[..., 1:, :]  # [..., N, T]
        x_down = weight @ x  # [..., N, T] @ [..., T, C] -> [..., N, C]
        return x_down  # [..., N, C]


class AttentionDownsample(nn.Module):
    """
    Unified attention downsample module.
    Interface similar to LocalDownsample, internally dispatches to
    LocalAttentionPoolDown / AttentionPoolDown / CrossAttentionDown / LocalCrossAttentionDown.

    Pool/query token source:
    - 'local_avg': LocalDownsample (mean pooling per region)
    - 'learnable': LearnablePoolTokens (learnable embeddings)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: int,
            mode: str = 'local_attn_pool',  # 'local_attn_pool', 'attn_pool', 'cross_attn', 'local_cross_attn'
            token_source: str = 'local_avg',  # 'local_avg', 'learnable'
            region_token_num: int = 1,
            use_rope: bool = True,
            use_region_rope: bool = True,
            use_region_bias: bool = False,
            use_pool_offset: bool = False,
            dropout_attn: float = 0.0,
            out_drop: float = 0.0,
            theta: float = 10000.0,
    ):
        super().__init__()
        self.mode = mode
        self.token_source = token_source
        self.region_token_num = region_token_num
        self.num_heads = num_heads
        self.head_dim = head_dim
        attn_dim = num_heads * head_dim

        # Token source
        if token_source == 'local_avg':
            self.token_gen = LocalDownsample()
        elif token_source == 'learnable':
            self.token_gen = LearnablePoolTokens(dim, region_token_num)
        else:
            raise ValueError(f"Unknown token_source: {token_source}")

        # Projections
        is_sa = mode in ('local_attn_pool', 'attn_pool')
        is_ca = mode in ('cross_attn', 'local_cross_attn')

        # Pool/query projections
        self.pool_proj_q = nn.Linear(dim, attn_dim, bias=True)
        if is_sa:
            self.pool_proj_k = nn.Linear(dim, attn_dim, bias=True)
            self.pool_proj_v = nn.Linear(dim, attn_dim, bias=True)

        # X projections
        if is_sa:
            self.x_proj_q = nn.Linear(dim, attn_dim, bias=True)
        self.x_proj_k = nn.Linear(dim, attn_dim, bias=True)
        self.x_proj_v = nn.Linear(dim, attn_dim, bias=True)

        # Output projection
        self.out_linear = nn.Linear(attn_dim, dim, bias=True)
        self.out_drop = nn.Dropout(out_drop) if out_drop > 0. else nn.Identity()

        # Core attention module
        if mode == 'local_attn_pool':
            self.attn = LocalAttentionPoolDown(
                head_dim=head_dim, region_token_num=region_token_num,
                use_rope=use_rope, use_pool_offset=use_pool_offset,
                dropout_attn=dropout_attn, theta=theta,
            )
        elif mode == 'attn_pool':
            self.attn = AttentionPoolDown(
                head_dim=head_dim, num_heads=num_heads,
                region_token_num=region_token_num,
                use_rope=use_rope, use_region_rope=use_region_rope,
                use_region_bias=use_region_bias, use_pool_offset=use_pool_offset,
                dropout_attn=dropout_attn, theta=theta,
            )
        elif mode == 'cross_attn':
            self.attn = CrossAttentionDown(
                head_dim=head_dim, num_heads=num_heads,
                region_token_num=region_token_num,
                use_rope=use_rope, use_region_rope=use_region_rope,
                use_region_bias=use_region_bias, use_pool_offset=use_pool_offset,
                dropout_attn=dropout_attn, theta=theta,
            )
        elif mode == 'local_cross_attn':
            self.attn = LocalCrossAttentionDown(
                head_dim=head_dim, region_token_num=region_token_num,
                use_rope=use_rope, use_pool_offset=use_pool_offset,
                dropout_attn=dropout_attn, theta=theta,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _generate_tokens(self, x, regions, max_n, n_mask):
        """Generate pool/query tokens based on token_source."""
        if self.token_source == 'local_avg':
            # [B, N, C], one token per region
            tokens = self.token_gen(x, regions, max_n)  # [B, N, C]
            if self.region_token_num > 1:
                # Repeat for region_token_num
                tokens = tokens.unsqueeze(2).expand(-1, -1, self.region_token_num, -1)
                tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])  # [B, N*R, C]
            # Apply n_mask
            mask = n_mask.unsqueeze(-1)  # [B, N, 1]
            if self.region_token_num > 1:
                mask = mask.unsqueeze(-1).expand(-1, -1, self.region_token_num, -1)
                mask = mask.reshape(mask.shape[0], -1, 1)  # [B, N*R, 1]
            tokens = tokens * mask.float()
            return tokens
        else:
            return self.token_gen(max_n, n_mask)  # [B, N*R, C]

    def _to_heads(self, x):
        return rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

    def forward(self, x, regions, t_mask, n_mask, max_n=None):
        """
        :param x: [B, T, C] input tensor
        :param regions: [B, T] int64, 0=pad, 1..N
        :param t_mask: [B, T] bool, True=valid
        :param n_mask: [B, N] bool, True=valid
        :param max_n: int, optional. N = max(regions) if not given.
        :return: [B, N*R, C] downsampled output (or [B, N, C] if region_token_num=1)
        """
        if max_n is None:
            max_n = regions.max().item()

        # Generate pool/query tokens
        pool = self._generate_tokens(x, regions, max_n, n_mask)  # [B, P, C]

        is_sa = self.mode in ('local_attn_pool', 'attn_pool')
        is_ca = self.mode in ('cross_attn', 'local_cross_attn')

        # Project
        pool_q = self._to_heads(self.pool_proj_q(pool))

        x_k = self._to_heads(self.x_proj_k(x))
        x_v = self._to_heads(self.x_proj_v(x))

        if is_sa:
            pool_k = self._to_heads(self.pool_proj_k(pool))
            pool_v = self._to_heads(self.pool_proj_v(pool))
            x_q = self._to_heads(self.x_proj_q(x))

            out = self.attn(
                pool_q, pool_k, pool_v,
                x_q, x_k, x_v,
                regions, t_mask, n_mask, max_n,
            )
        else:
            out = self.attn(
                pool_q,
                x_k, x_v,
                regions, t_mask, n_mask, max_n,
            )

        # [B, H, P, D] -> [B, P, C]
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.out_linear(out)
        out = self.out_drop(out)

        # Apply n_mask to output
        P = max_n * self.region_token_num
        out_mask = n_mask.unsqueeze(-1).expand(-1, -1, self.region_token_num)
        out_mask = out_mask.reshape(out.shape[0], P, 1).float()
        out = out * out_mask

        return out
