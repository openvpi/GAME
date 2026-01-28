import torch
from torch import Tensor
from torch.nn import functional as F


@torch.no_grad()
def dur_to_mel2ph(lr, durs, length, timestep):
    non_batched = durs.ndim == 1
    if non_batched:
        b = 1
        durs = durs.unsqueeze(0)
    else:
        b = durs.shape[0]
    ph_acc = torch.round(torch.cumsum(durs, dim=1) / timestep + 0.5).long()
    ph_dur = torch.diff(ph_acc, dim=1, prepend=torch.zeros(b, 1).to(durs.device))
    mel2ph = lr(ph_dur)
    num_frames = mel2ph.shape[1]
    if num_frames < length:
        mel2ph = torch.nn.functional.pad(mel2ph, (0, length - num_frames), mode="replicate")
    elif num_frames > length:
        mel2ph = mel2ph[:, :length]
    if non_batched:
        mel2ph = mel2ph.squeeze(0)
    return mel2ph


def mel2ph_to_dur(mel2ph, n_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, n_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur


def find_local_minima(x: Tensor, threshold: float, radius: int = 2):
    """
    Find local minima in the last dimension of x within a given radius and below a threshold.
    :param x: [..., T]
    :param threshold: ignore values above this threshold
    :param radius: int, radius for local minima search
    :return: [..., T], 1 = minima, 0 = non-minima
    """
    assert radius >= 1
    x_pad = F.pad(
        x, (radius, radius),
        mode="constant", value=float("-inf")
    )  # [..., T + 2r]
    windows = x_pad.unfold(dimension=-1, size=2 * radius + 1, step=1)  # [..., T, 2r+1]
    minima = (windows.argmin(dim=-1) == radius) & (x <= threshold)  # [..., T]
    return minima


def decode_boundaries_from_velocities(velocities: Tensor, threshold: float = 0.2, radius: int = 2):
    """
    Decode boundary indicators from predicted velocities. Steps:
        1. Accumulate velocities to get distances.
        2. Normalize distances, remove positive bias and scale maximum to 1.
        3. Find local minima in distances that are below the threshold.
    :param velocities: [..., T]
    :param threshold: float (0~1), velocity threshold (after normalization) to consider a boundary
    :param radius: int, radius for local minima search
    :return: [..., T], 1 = boundary, 0 = non-boundary
    """
    distances = velocities.cumsum(dim=-1)
    d_min = distances.amin(dim=-1, keepdim=True)
    d_max = distances.amax(dim=-1, keepdim=True)
    distances = (distances - d_min) / (d_max - d_min + 1e-8)
    boundaries = find_local_minima(distances, threshold=threshold, radius=radius)  # [..., T]
    return boundaries
