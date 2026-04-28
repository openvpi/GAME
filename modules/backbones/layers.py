from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class LayerScale(nn.Module):
    def __init__(self, dim, lay_scale_init_value=1e-6):
        super().__init__()
        sp = torch.ones(dim) * lay_scale_init_value
        self.scale = nn.Parameter(sp)
        self.dim = dim

    def unc(self, res):
        n_dim = res.ndim
        if n_dim == 1:
            return self.scale
        else:
            return self.scale.view(*((1,) * (n_dim - 1)), self.dim)

    def forward(self, x):
        return x * self.unc(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, init_num=1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * init_num)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


class GLUFFN(nn.Module):
    def __init__(self, dim, latent_dim=None, dropout_latent: float = 0.1, dropout_output: float = 0.1):
        super().__init__()
        if latent_dim is None:
            latent_dim = dim * 4
        self.ln1 = nn.Linear(dim, latent_dim * 2)

        self.ln2 = nn.Linear(latent_dim, dim)
        self.dropout_latent = nn.Dropout(dropout_latent) if dropout_latent > 0. else nn.Identity()
        self.dropout_output = nn.Dropout(dropout_output) if dropout_output > 0. else nn.Identity()

    def forward(self, x):
        x1, x2 = self.ln1(x).chunk(2, dim=-1)
        x = F.gelu(x1) * x2
        x = self.dropout_latent(x)
        x = self.ln2(x)
        return self.dropout_output(x)


class FFN(nn.Module):
    def __init__(self, dim, latent_dim=None, dropout_latent: float = 0.1, dropout_output: float = 0.1):
        super().__init__()
        if latent_dim is None:
            latent_dim = dim * 4
        self.ln1 = nn.Linear(dim, latent_dim)
        self.ln2 = nn.Linear(latent_dim, dim)
        self.dropout_latent = nn.Dropout(dropout_latent) if dropout_latent > 0. else nn.Identity()
        self.dropout_output = nn.Dropout(dropout_output) if dropout_output > 0. else nn.Identity()

    def forward(self, x):
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.dropout_latent(x)
        x = self.ln2(x)
        return self.dropout_output(x)


class CgMLP(nn.Module):
    def __init__(
            self, dim: int,
            kernel_size: int = 31,
            out_drop=0.1,
            latent_drop=0.0,
            bias: bool = True,
            use_dw_act=True,
            latent_dim: Optional[int] = None
    ):
        super().__init__()
        if latent_dim is None:
            latent_dim = dim
        self.pw1 = nn.Conv1d(
            dim,
            latent_dim * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.use_dw_act = use_dw_act
        self.norm = RMSNorm(latent_dim)
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(
            latent_dim, latent_dim, kernel_size,
            stride=1,
            padding=padding,
            groups=latent_dim,
            bias=bias
        )
        self.pw2 = nn.Conv1d(
            latent_dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.out_drop = nn.Dropout(out_drop) if out_drop > 0. else nn.Identity()
        self.latent_drop = nn.Dropout(latent_drop) if latent_drop > 0. else nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pw1(x)
        x = F.gelu(x)
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.norm(x2.transpose(1, 2)).transpose(1, 2)
        x2 = self.dw(x2)
        if self.use_dw_act:
            x2 = F.gelu(x2)
        x = x1 * x2
        x = self.latent_drop(x)
        x = self.pw2(x)
        return self.out_drop(x).transpose(1, 2)
