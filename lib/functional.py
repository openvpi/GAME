import torch
import torch.nn.functional as F
from torch import Tensor


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


def collate_nd(values, pad_value=0, max_len=None):
    """
    Pad a list of Nd tensors on their first dimension and stack them into a (N+1)d tensor.
    """
    size = ((max(v.size(0) for v in values) if max_len is None else max_len), *values[0].shape[1:])
    res = torch.full((len(values), *size), fill_value=pad_value, dtype=values[0].dtype, device=values[0].device)

    for i, v in enumerate(values):
        res[i, :len(v), ...] = v
    return res


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def random_continuous_masks(*shape: int, dim: int, device: str | torch.device = "cpu"):
    start, end = torch.sort(
        torch.randint(
            low=0, high=shape[dim] + 1, size=(*shape[:dim], 2, *((1,) * (len(shape) - dim - 1))), device=device
        ).expand(*((-1,) * (dim + 1)), *shape[dim + 1:]), dim=dim
    )[0].split(1, dim=dim)
    idx = torch.arange(
        0, shape[dim], dtype=torch.long, device=device
    ).reshape(*((1,) * dim), shape[dim], *((1,) * (len(shape) - dim - 1)))
    masks = (idx >= start) & (idx < end)
    return masks


def distance_transform(boundaries: Tensor) -> Tensor:
    """
    Compute the distance transform for binary boundary indicators.
    For consistency with cases where there are no boundaries, we pad boundaries with 1s at both ends.
    For each position, compute the distance to the nearest boundary (where boundary == 1).
    :param boundaries: [..., T], 1 = boundary, 0 = non-boundary
    :return: [..., T]
    """
    boundaries = F.pad(boundaries, (1, 1), mode="constant", value=1)
    indices = torch.arange(
        boundaries.shape[-1], dtype=torch.float32, device=boundaries.device,
    ).view([1] * (boundaries.ndim - 1) + [-1])  # [..., T]
    masked_indices = torch.where(boundaries, indices, torch.full_like(indices, fill_value=float("inf")))
    distance = torch.abs(indices.unsqueeze(-1) - masked_indices.unsqueeze(-2)).min(dim=-1).values
    distance = distance[..., 1:-1]
    return distance


def self_cosine_similarity(x: Tensor) -> Tensor:
    """
    [..., T, C] -> [..., T, T]
    """
    x_norm = F.normalize(x.float(), p=2, dim=-1, eps=1e-8)
    sim = x_norm @ x_norm.transpose(-1, -2)
    return sim


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
    bias = distances.min(dim=-1, keepdim=True).values.clamp(min=0)
    distances = distances - bias
    scale = distances.max(dim=-1, keepdim=True).values.clamp(min=1e-3)
    distances = distances / scale
    boundaries = find_local_minima(distances, threshold=threshold, radius=radius)  # [..., T]
    return boundaries
