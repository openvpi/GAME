import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Pool Token Generators
# ============================================================

class LearnablePoolTokens(nn.Module):
    """Learnable pool tokens per region."""

    def __init__(self, dim, region_token_num=1):
        super().__init__()
        self.region_token_num = region_token_num
        self.emb = nn.Parameter(torch.randn(region_token_num, dim) * 0.02)

    def forward(self, n_mask):
        """
        :param n_mask: [B, N] bool
        :return: [B, N * R, C]
        """
        B, N = n_mask.shape
        R = self.region_token_num
        tokens = self.emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, R, C]
        tokens = tokens * n_mask.unsqueeze(-1).unsqueeze(-1).float()
        return tokens.reshape(B, N * R, -1)


# ============================================================
# Pool Token Merger (for region_token_num > 1)
# ============================================================

class PoolTokenMerger(nn.Module):
    """
    Merge multiple pool tokens per region back to single token.
    Input: [B, N*R, C] -> Output: [B, N, C]

    Modes:
    - 'mean': average pooling across R tokens
    - 'max': max pooling across R tokens
    - 'first': take the first token only
    - 'learned': learned weighted sum with softmax
    - 'attention': self-attention to merge (query from first token)
    """

    def __init__(self, dim, region_token_num, mode='mean', num_heads=4):
        super().__init__()
        self.dim = dim
        self.R = region_token_num
        self.mode = mode

        if mode == 'learned':
            # Learnable weights for each of the R tokens
            self.merge_weights = nn.Parameter(torch.zeros(region_token_num))
        elif mode == 'attention':
            # Cross-attention: first token queries all R tokens
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.scale = head_dim ** -0.5

    def forward(self, pool, n_mask):
        """
        Args:
            pool: [B, N*R, C] pool tokens
            n_mask: [B, N] valid mask for regions
        Returns:
            merged: [B, N, C] merged pool tokens
        """
        B, P, C = pool.shape
        R = self.R
        N = P // R

        # Reshape to [B, N, R, C]
        pool = pool.reshape(B, N, R, C)

        if self.mode == 'mean':
            merged = pool.mean(dim=2)  # [B, N, C]

        elif self.mode == 'max':
            merged = pool.max(dim=2).values  # [B, N, C]

        elif self.mode == 'first':
            merged = pool[:, :, 0, :]  # [B, N, C]

        elif self.mode == 'learned':
            # Softmax over R dimension
            weights = F.softmax(self.merge_weights, dim=0)  # [R]
            weights = weights.view(1, 1, R, 1)  # [1, 1, R, 1]
            merged = (pool * weights).sum(dim=2)  # [B, N, C]

        elif self.mode == 'attention':
            # First token as query, all tokens as key/value
            q = pool[:, :, 0:1, :]  # [B, N, 1, C]
            k = pool  # [B, N, R, C]
            v = pool  # [B, N, R, C]

            q = self.q_proj(q)  # [B, N, 1, C]
            k = self.k_proj(k)  # [B, N, R, C]
            v = self.v_proj(v)  # [B, N, R, C]

            # Reshape for multi-head attention
            H = self.num_heads
            D = C // H
            q = q.reshape(B, N, 1, H, D).permute(0, 1, 3, 2, 4)  # [B, N, H, 1, D]
            k = k.reshape(B, N, R, H, D).permute(0, 1, 3, 2, 4)  # [B, N, H, R, D]
            v = v.reshape(B, N, R, H, D).permute(0, 1, 3, 2, 4)  # [B, N, H, R, D]

            # Attention
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, N, H, 1, R]
            attn = F.softmax(attn, dim=-1)
            out = attn @ v  # [B, N, H, 1, D]
            out = out.squeeze(-2).reshape(B, N, C)  # [B, N, C]
            merged = self.out_proj(out)

        else:
            raise ValueError(f"Unknown merge mode: {self.mode}")

        # Apply n_mask
        merged = merged * n_mask.unsqueeze(-1).float()
        return merged
