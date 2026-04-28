import torch
import torch.nn as nn


def compute_inv_freq(dim: int, theta: float = 10000.0):
    """pre-compute inv_freq, dim is fixed"""
    return 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))


def compute_freqs_cis_dynamic(x: torch.Tensor, inv_freq: torch.Tensor):
    """ONNX兼容：动态计算cos/sin，序列长度从xa tensor shape获取"""
    seq_len = x.shape[-2]
    t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def single_apply_rotary_emb(
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
):
    """ONNX兼容：手动实现复数乘法"""
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


class SingleRoPosEmb(nn.Module):
    def __init__(self, dim: int, max_len=5000, theta=10000.0, use_cache=True):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.use_cache = use_cache
        self.register_buffer('inv_freq', compute_inv_freq(dim, theta), persistent=False)
        if use_cache:
            pe_cos, pe_sin = compute_freqs_cis_dynamic(
                torch.zeros(1, max_len, dim), self.inv_freq)
            self.register_buffer('pe_cos', pe_cos[None, :, :], persistent=False)
            self.register_buffer('pe_sin', pe_sin[None, :, :], persistent=False)

    def extend_pe(self, x):
        if self.pe_cos.size(1) >= x.size(-2):
            return
        pe_cos, pe_sin = compute_freqs_cis_dynamic(x, self.inv_freq)
        self.pe_cos = pe_cos[None, :, :].to(device=x.device)
        self.pe_sin = pe_sin[None, :, :].to(device=x.device)

    def get_pe_dynamic(self, x):
        ndim = x.ndim
        seq_len = x.shape[-2]
        pe_cos, pe_sin = compute_freqs_cis_dynamic(x, self.inv_freq)
        pe_cos = pe_cos.view(*((1,) * (ndim - 2)), seq_len, self.dim // 2)
        pe_sin = pe_sin.view(*((1,) * (ndim - 2)), seq_len, self.dim // 2)
        return pe_cos, pe_sin

    def get_pe_cached(self, x):
        ndim = x.ndim
        seq_len = x.size(-2)
        pe_cos = self.pe_cos[:, :seq_len]
        pe_sin = self.pe_sin[:, :seq_len]
        pe_cos = pe_cos.view(*((1,) * (ndim - 2)), seq_len, self.dim // 2)
        pe_sin = pe_sin.view(*((1,) * (ndim - 2)), seq_len, self.dim // 2)
        return pe_cos, pe_sin

    def forward(self, x):
        if self.use_cache:
            self.extend_pe(x)
            pe_cos, pe_sin = self.get_pe_cached(x)
        else:
            pe_cos, pe_sin = self.get_pe_dynamic(x)
        return single_apply_rotary_emb(x, pe_cos, pe_sin)
