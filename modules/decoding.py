import math

import torch
from torch import Tensor
from torch.nn import functional as F


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


def decode_boundaries_from_velocities(
        velocities: Tensor, barriers: Tensor = None, mask: Tensor = None,
        threshold: float = 0.2, radius: int = 2
):
    """
    Decode boundary indicators from predicted velocities. Steps:
        1. Accumulate velocities to get distances.
        2. Normalize distances, remove positive bias and scale maximum to 1.
        3. Find local minima in distances that are below the threshold.
    :param velocities: [..., T]
    :param barriers: [..., T],optional preset boundaries or local minima points
    :param mask: [..., T], optional mask to apply before decoding
    :param threshold: float (0~1), velocity threshold (after normalization) to consider a boundary
    :param radius: int, radius for local minima search
    :return: [..., T], 1 = boundary, 0 = non-boundary
    """
    distances = velocities.cumsum(dim=-1)
    if mask is not None:
        distances_upper_masked = torch.where(mask, distances, float("+inf"))
        distances_lower_masked = torch.where(mask, distances, float("-inf"))
    else:
        distances_upper_masked = distances
        distances_lower_masked = distances
    d_min = distances_upper_masked.amin(dim=-1, keepdim=True)
    d_max = distances_lower_masked.amax(dim=-1, keepdim=True)
    distances = (distances - d_min) / (d_max - d_min + 1e-8)
    if mask is not None:
        distances = torch.where(mask, distances, float("-inf"))
    if barriers is not None:
        distances = torch.masked_fill(distances, barriers, float("-inf"))
    boundaries = find_local_minima(distances, threshold=threshold, radius=radius)  # [..., T]
    if mask is not None:
        boundaries &= mask
    return boundaries


def decode_quantized_boundaries(boundaries: Tensor):
    """
    Decode quantized boundary indicators from blurred boundary probabilities.
    :param boundaries: float [..., T], boundary probabilities, should be in [0, 1]
    :return: bool [..., T], 1 = boundary, 0 = non-boundary
    """
    boundaries_step = boundaries.cumsum(dim=-1).round().long()
    boundaries_diff = boundaries_step[..., 1:] > boundaries_step[..., :-1]
    return F.pad(boundaries_diff, (1, 0), mode="constant", value=1)


def decode_gaussian_blurred_probs(
        probs: Tensor,
        min_val: float, max_val: float, deviation: float,
        threshold: float
):
    """
    Decode gaussian-blurred probabilities to continuous values and presence flags.
    :param probs: [..., T, N]
    :param min_val: value of the lowest bin
    :param max_val: value of the highest bin
    :param deviation: deviation of the gaussian blur in the original value scale
    :param threshold: presence threshold
    :return: values [..., T], presence [..., T]
    """
    B = (1,) * (probs.ndim - 2)
    N = probs.shape[-1]
    width = math.ceil(deviation / (max_val - min_val) * (N - 1))
    idx = torch.arange(N, dtype=torch.long, device=probs.device).reshape(*B, 1, -1)  # [..., 1, N]
    center_values = torch.linspace(min_val, max_val, steps=N, device=probs.device).reshape(*B, 1, -1)  # [1, 1, N]
    centers = torch.argmax(probs, dim=-1, keepdim=True)  # [..., T, 1]
    start = torch.clip(centers - width, min=0)  # [..., T, 1]
    end = torch.clip(centers + width + 1, max=N)  # [..., T, 1]
    idx_masks = (idx >= start) & (idx < end)  # [..., T, N]
    weights = probs * idx_masks  # [..., T, N]
    product_sum = torch.sum(weights * center_values, dim=2)  # [..., T]
    weight_sum = torch.sum(weights, dim=2)  # [..., T]
    values = product_sum / (weight_sum + 1e-8)  # avoid dividing by zero, [..., T]
    presence = probs.amax(dim=-1) >= threshold  # [..., T]
    return values, presence
