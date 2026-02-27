import torch
import torch.nn as nn


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
    """
    Apply rope given arbitrary per-token positions.
    :param x: [..., T, D]  (D must be even)
    :param positions: [B, T] or matching leading dims except H
    :param inv_freq: [D//2]
    """
    pos = positions.unsqueeze(-1).float()  # [B, T, 1]
    inv = inv_freq.view(*((1,) * (pos.ndim - 1)), -1)  # [1, 1, D//2]
    freqs = pos * inv  # [B, T, D//2]

    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    # Insert dims to match x: [B, T, D//2] -> [B, 1..., T, D//2]
    n_extra = x.ndim - positions.ndim - 1
    shape = freqs_cos.shape[:1] + (1,) * n_extra + freqs_cos.shape[1:]
    freqs_cos = freqs_cos.view(shape)
    freqs_sin = freqs_sin.view(shape)

    return single_apply_rotary_emb(x, freqs_cos, freqs_sin)


# ============================================================
# Region position utilities
# ============================================================

def regions_to_local_positions(regions):
    """
    Convert region indices to local (within-region) positions.
    regions: [B, T], int64, 0 = padding, 1..N = region id
    returns: [B, T], int64
    """
    B, T = regions.shape
    device = regions.device
    valid = regions > 0
    same_region = (regions.unsqueeze(-1) == regions.unsqueeze(-2))  # [B, T, T]
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=-1)
    same_region_before = same_region & causal.unsqueeze(0) & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    local_pos = same_region_before.sum(dim=-1)  # [B, T]
    return local_pos * valid.long()


def region_start_positions(regions, max_n):
    """
    For each region, find the start position (first occurrence index).
    regions: [B, T], int64, 0=pad, 1..N
    returns: [B, N], int64
    """
    B, T = regions.shape
    device = regions.device
    starts = torch.full((B, max_n + 1), T, dtype=torch.long, device=device)
    pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    starts.scatter_reduce_(1, regions, pos_idx, reduce='amin', include_self=True)
    return starts[:, 1:]  # [B, N]


# ============================================================
# RegionRoPE
# ============================================================

class RegionRoPE(nn.Module):
    """
    Unified RoPE module:
    - local mode: region-internal positions, full head_dim
    - global mode: split head_dim = global position rope + region index rope
    """

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
        """
        :param q/k: [B, H, L, D]
        :param q_positions/k_positions: [B, L]
        :param q_region_idx/k_region_idx: [B, L] (global mode only)
        """
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
