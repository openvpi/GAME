from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


@dataclass
class NoteVernierConfig:
    """Note Vernier 配置"""
    note_min: float = 0.0
    note_max: float = 127.0
    periods: Tuple = (48,24,12.0, 4,1.0,)

    @property
    def output_dim(self):
        # 1 (UV) + 1 (note) + len(periods) * 2 (Sin/Cos)
        return 1 + 1 + len(self.periods) * 2


class NoteVernierLoss(nn.Module):
    """Note Vernier Loss"""

    def __init__(self, config: NoteVernierConfig, w_uv=1.0, w_mse=10.0, w_cos=1.0):
        super().__init__()
        self.config = config
        self.w_uv = w_uv
        self.w_mse = w_mse
        self.w_cos = w_cos

    def forward(self, pred_vec, gt_note, gt_uv):
        cfg = self.config
        is_voiced = (gt_uv < 0.5)

        # UV Loss
        loss_uv = F.binary_cross_entropy_with_logits(pred_vec[..., 0], gt_uv)

        if is_voiced.sum() == 0:
            return self.w_uv * loss_uv

        p_voiced = pred_vec[is_voiced]
        g_voiced = gt_note[is_voiced]

        # MSE Loss (归一化)
        gt_norm = (g_voiced - cfg.note_min) / (cfg.note_max - cfg.note_min)
        pred_norm = torch.sigmoid(p_voiced[..., 1])  # 用 sigmoid 约束到 [0,1]
        loss_mse = F.mse_loss(pred_norm, gt_norm)

        # Cyclic Loss
        loss_cos = 0.0
        start_idx = 2
        for period in cfg.periods:
            phase = (g_voiced / period) * 2 * np.pi
            gt_vec = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)

            pred_ring = F.normalize(p_voiced[..., start_idx:start_idx + 2], dim=-1)
            loss_cos += 1.0 - (pred_ring * gt_vec).sum(dim=-1).mean()
            start_idx += 2

        return self.w_uv * loss_uv + self.w_mse * loss_mse + self.w_cos * loss_cos


def decode_note_vernier(pred_vec, config: NoteVernierConfig):
    """Note Vernier 解码"""
    cfg = config

    uv_prob = torch.sigmoid(pred_vec[..., 0])
    pred_uv = (uv_prob > 0.5)


    note = torch.sigmoid(pred_vec[..., 1]) * (cfg.note_max - cfg.note_min) + cfg.note_min

    start_idx = 2
    for period in cfg.periods:
        ring = F.normalize(pred_vec[..., start_idx:start_idx + 2], dim=-1)
        phase_val = torch.atan2(ring[..., 0], ring[..., 1])
        pred_offset = (phase_val / (2 * np.pi)) * period

        k = torch.round((note - pred_offset) / period)
        note = k * period + pred_offset
        start_idx += 2

    return note, pred_uv


def decode_vernier_modulo(pred_vec, config):
    """
    版本 B: 取模解缠 (Modulo Unwrapping)
    核心: diff = remainder(diff + P/2, P) - P/2
    缺点: remainder 在边界处梯度不连续，偶尔会有数值精度坑。
    """
    # 1. MSE 基准
    cfg = config

    uv_prob = torch.sigmoid(pred_vec[..., 0])
    pred_uv = (uv_prob > 0.5)


    # norm_scale = cfg.note_max - cfg.note_min
    # curr_val = pred_vec[..., 1] * norm_scale + cfg.note_min
    note = torch.sigmoid(pred_vec[..., 1]) * (cfg.note_max - cfg.note_min) + cfg.note_min
    start_idx = 2
    for period in cfg.periods:
        v = F.normalize(pred_vec[..., start_idx:start_idx + 2], dim=-1)
        phase = torch.atan2(v[..., 0], v[..., 1])
        # 模型预测的圆环位置
        val_pred = (phase / (2 * np.pi)) * period

        # 当前猜测的圆环位置 (取模)
        # 注意：这里要处理 curr_val 为负数的情况，Python % 和 torch.remainder 行为略有不同
        # 建议用 torch.remainder
        curr_mod = torch.remainder(note, period)

        # 计算差值
        diff = val_pred - curr_mod

        # 核心逻辑：把差值折叠到 [-period/2, period/2]
        # 这是一个锯齿波函数
        diff = torch.remainder(diff + period / 2, period) - period / 2

        # 更新
        note = note + diff

        start_idx += 2

    return note, pred_uv


def decode_vernier_analytic(pred_vec, config):
    """
    版本 C: 几何解析解 (Geometric Projection)
    核心: delta = atan2(cross_product, dot_product)
    优点: 全程平滑可导 (Sin/Cos/Atan2)，无锯齿，无离散跳变。
    """

    cfg = config

    uv_prob = torch.sigmoid(pred_vec[..., 0])
    pred_uv = (uv_prob > 0.5)

    # 1. MSE 基准
    # norm_scale = config.note_max - config.note_min
    # curr_val = pred_vec[..., 1] * norm_scale + config.note_min
    note = torch.sigmoid(pred_vec[..., 1]) * (cfg.note_max - cfg.note_min) + cfg.note_min
    start_idx = 2
    for period in config.periods:
        # A. 预测向量 (Target)
        v_pred = F.normalize(pred_vec[..., start_idx:start_idx + 2], dim=-1)
        sin_p, cos_p = v_pred[..., 0], v_pred[..., 1]

        # B. 当前猜测向量 (Current Guess)
        # 把直线卷起来变成向量
        phase_curr = (note / period) * 2 * np.pi
        sin_c, cos_c = torch.sin(phase_curr), torch.cos(phase_curr)

        # C. 计算向量夹角 (Rotation)
        # sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
        sin_delta = sin_p * cos_c - cos_p * sin_c
        # cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
        cos_delta = cos_p * cos_c + sin_p * sin_c

        # atan2 自动找最短路径 [-pi, pi]
        delta_phase = torch.atan2(sin_delta, cos_delta)

        # D. 线性投影
        delta_val = (delta_phase / (2 * np.pi)) * period

        # 更新
        note = note + delta_val

        start_idx += 2

    return note, pred_uv

@dataclass
class NoteGaussianBinVernierConfig:
    """Gaussian Bin + Vernier 配置"""
    note_min: float = 0.0
    note_max: float = 127.0
    n_bins: int = 16
    sigma: float = 1.0
    periods: Tuple = (24,12.0, 4.0, 1.0)

    @property
    def bin_width(self):
        return (self.note_max - self.note_min) / self.n_bins


class NoteGaussianBinVernierLoss(nn.Module):
    """Gaussian Bin + Vernier Loss"""

    def __init__(self, config: NoteGaussianBinVernierConfig, w_bin=1.0, w_uv=1.0, w_cos=1.0):
        super().__init__()
        self.config = config
        self.w_bin = w_bin
        self.w_uv = w_uv
        self.w_cos = w_cos
        bin_centers = torch.arange(config.n_bins, dtype=torch.float32)
        self.register_buffer("bin_centers", bin_centers, persistent=False)

    def forward(self, pred_vec, gt_note, gt_uv):
        cfg = self.config
        is_voiced = (gt_uv < 0.5)

        pred_bins = pred_vec[..., :cfg.n_bins]
        pred_uv = pred_vec[..., cfg.n_bins]
        pred_rings = pred_vec[..., cfg.n_bins + 1:]

        loss_uv = F.binary_cross_entropy_with_logits(pred_uv, gt_uv)

        if is_voiced.sum() == 0:
            return self.w_uv * loss_uv

        gt_voiced = gt_note[is_voiced]
        pred_bins_voiced = pred_bins[is_voiced]
        pred_rings_voiced = pred_rings[is_voiced]

        target_pos = (gt_voiced - cfg.note_min) / cfg.bin_width
        diff = self.bin_centers - target_pos.unsqueeze(-1)
        gaussian = torch.exp(-0.5 * (diff / cfg.sigma) ** 2)
        gaussian = gaussian / (gaussian.sum(dim=-1, keepdim=True) + 1e-8)
        loss_bin = F.binary_cross_entropy_with_logits(pred_bins_voiced, gaussian, reduction='none')
        loss_bin = loss_bin.sum(dim=-1).mean()

        loss_cos = 0.0
        start_idx = 0
        for period in cfg.periods:
            phase = (gt_voiced / period) * 2 * np.pi
            gt_vec = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
            pred_ring = F.normalize(pred_rings_voiced[..., start_idx:start_idx + 2], dim=-1)
            loss_cos += 1.0 - (pred_ring * gt_vec).sum(dim=-1).mean()
            start_idx += 2

        return self.w_bin * loss_bin + self.w_uv * loss_uv + self.w_cos * loss_cos


def decode_note_gaussian_bin_vernier(pred_vec, config: NoteGaussianBinVernierConfig):
    """Gaussian Bin + Vernier 解码"""
    cfg = config

    pred_bins = pred_vec[..., :cfg.n_bins]
    pred_uv_logit = pred_vec[..., cfg.n_bins]
    pred_rings = pred_vec[..., cfg.n_bins + 1:]

    pred_uv = torch.sigmoid(pred_uv_logit) > 0.5

    probs = torch.softmax(pred_bins, dim=-1)
    bin_centers = torch.arange(cfg.n_bins, device=pred_bins.device, dtype=pred_bins.dtype)
    coarse_bin = (probs * bin_centers).sum(dim=-1)
    note = cfg.note_min + (coarse_bin + 0.5) * cfg.bin_width

    start_idx = 0
    for period in cfg.periods:
        ring = F.normalize(pred_rings[..., start_idx:start_idx + 2], dim=-1)
        phase_val = torch.atan2(ring[..., 0], ring[..., 1])
        pred_offset = (phase_val / (2 * np.pi)) * period
        k = torch.round((note - pred_offset) / period)
        note = k * period + pred_offset
        start_idx += 2

    return note, pred_uv