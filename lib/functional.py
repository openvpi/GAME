import numpy
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


def resample_align_curve(points: numpy.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = numpy.interp(
        numpy.arange(0, t_max, target_timestep),
        original_timestep * numpy.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = numpy.concatenate((curve_interp, numpy.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp


def resize_curve(curve: numpy.ndarray, target_length: int):
    original_length = len(curve)
    original_indices = numpy.linspace(0, original_length - 1, num=original_length)
    target_indices = numpy.linspace(0, original_length - 1, num=target_length)
    interpolated_curve = numpy.interp(target_indices, original_indices, curve).astype(curve.dtype)
    return interpolated_curve


def self_cosine_similarity(x: Tensor) -> Tensor:
    """
    [..., T, C] -> [..., T, T]
    """
    x_norm = F.normalize(x.float(), p=2, dim=-1, eps=1e-8)
    sim = x_norm @ x_norm.transpose(-1, -2)
    return sim
