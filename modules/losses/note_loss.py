import torch
from torch import nn, Tensor


class GaussianBlurredBinsLoss(nn.Module):

    def __init__(self, min_val: float, max_val: float, num_bins: int, deviation: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
        self.std = deviation / (max_val - min_val) * (num_bins - 1)
        centers = torch.linspace(min_val, max_val, num_bins)
        self.register_buffer("centers", centers, persistent=False)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: Tensor, scores: Tensor, presence: Tensor, weights: Tensor = None, mask=None) -> Tensor:
        """
        :param logits: [..., T, C] predicted logits
        :param scores: [..., T] target scores
        :param presence: [..., T] target presence, 0 means no score
        :param weights: [..., T] optional weights on each position
        :param mask: [..., T] optional mask
        :return: loss value
        """
        B = (1,) * (logits.ndim - 2)
        if mask is not None:
            mask = mask.unsqueeze(-1).float().expand_as(logits)
        centers = self.centers.reshape(*B, 1, -1)  # [..., 1, C]
        diffs = scores.unsqueeze(-1) - centers  # [..., T, C]
        gaussians = torch.exp(-0.5 * (diffs / self.std) ** 2)  # [..., T, C]
        gaussians = gaussians / (gaussians.sum(dim=-1, keepdim=True) + 1e-6)  # normalize
        targets = gaussians * presence.unsqueeze(-1)  # zero out where no presence
        loss = self.criterion(logits, targets)
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss
