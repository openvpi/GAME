import math

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator


def similarity_to_figure(similarities, durations, title=None):
    dur_cumsum = np.cumsum(durations)
    fig = plt.figure(figsize=(9, 9))
    plt.pcolor(similarities, vmin=-1, vmax=1)
    for i in range(durations.shape[0]):
        rect = matplotlib.patches.Rectangle(
            xy=(dur_cumsum[i] - durations[i], dur_cumsum[i] - durations[i]),
            width=durations[i], height=durations[i],
            edgecolor="red", fill=False, linewidth=1.5,
        )
        plt.gca().add_patch(rect)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def distance_boundary_to_figure(
        distance_gt: np.ndarray, distance_pred: np.ndarray,
        threshold: float = None,
        boundaries_tp: np.ndarray = None,
        boundaries_fp: np.ndarray = None,
        boundaries_fn: np.ndarray = None,
        title=None
):
    figure_width = 12
    figure_height = 6
    fig = plt.figure(figsize=(12, 6))
    plt.plot(distance_gt, color="b", label="gt")
    plt.plot(distance_pred, color="r", label="pred")
    if threshold is not None:
        plt.plot([0, distance_gt.shape[0]], [threshold, threshold], color="black", linestyle="--")
    positions = np.arange(distance_gt.shape[0], dtype=np.int64)
    circle_radius = 10
    x_min = -1
    x_max = distance_gt.shape[0]
    y_min = min(0, distance_gt.min(), distance_pred.min()) - 1
    y_max = min(distance_gt.max(), distance_pred.max()) + 1
    ratio = (figure_width / figure_height) * (y_max - y_min) / (x_max - x_min)

    def _draw_circles(x_index, y_arr, color, label):
        label_added = False
        for pos in positions[x_index]:
            plt.gca().add_patch(
                matplotlib.patches.Ellipse(
                    xy=(pos, y_arr[pos]),
                    width=circle_radius, height=circle_radius * ratio,
                    edgecolor=color, fill=False,
                    linewidth=1.5, label=(label if not label_added else None)
                )
            )
            label_added = True

    if boundaries_tp is not None:
        _draw_circles(positions[boundaries_tp], distance_pred, "green", "match")
    if boundaries_fp is not None:
        _draw_circles(positions[boundaries_fp], distance_pred, "orange", "exceed")
    if boundaries_fn is not None:
        _draw_circles(positions[boundaries_fn], distance_gt, "grey", "miss")
    plt.xlim(-1, distance_gt.shape[0])
    plt.ylim(y_min, y_max)
    plt.grid(axis="y")
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def boundary_to_figure(
        bounds_gt: np.ndarray, bounds_pred: np.ndarray,
        dur_gt: np.ndarray = None, dur_pred: np.ndarray = None,
        title=None
):
    fig = plt.figure(figsize=(12, 6))
    bounds_acc_gt = np.cumsum(bounds_gt)
    bounds_acc_pred = np.cumsum(bounds_pred)
    plt.plot(bounds_acc_gt, color="b", label="gt")
    plt.plot(bounds_acc_pred, color="r", label="pred")
    if dur_gt is not None and dur_pred is not None:
        height = math.ceil(max(bounds_acc_gt[-1], bounds_acc_pred[-1]))
        dur_acc_gt = np.cumsum(dur_gt)
        dur_acc_pred = np.cumsum(dur_pred)
        plt.vlines(dur_acc_gt[:-1], 0, height / 2, colors="b", linestyles="--")
        plt.vlines(dur_acc_pred[:-1], height / 2, height, colors="r", linestyles="--")
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis="y")
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def spec_to_figure(spec, vmin=None, vmax=None, title=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def dur_to_figure(dur_gt, dur_pred, txt, title=None):
    if isinstance(dur_gt, torch.Tensor):
        dur_gt = dur_gt.cpu().numpy()
    if isinstance(dur_pred, torch.Tensor):
        dur_pred = dur_pred.cpu().numpy()
    dur_gt = dur_gt.astype(np.int64)
    dur_pred = dur_pred.astype(np.int64)
    dur_gt = np.cumsum(dur_gt)
    dur_pred = np.cumsum(dur_pred)
    width = max(12, min(48, len(txt) // 2))
    fig = plt.figure(figsize=(width, 8))
    plt.vlines(dur_pred, 12, 22, colors="r", label="pred")
    plt.vlines(dur_gt, 0, 10, colors="b", label="gt")
    for i in range(len(txt)):
        shift = (i % 8) + 1
        plt.text((dur_pred[i-1] + dur_pred[i]) / 2 if i > 0 else dur_pred[i] / 2, 12 + shift, txt[i],
                 size=16, horizontalalignment="center")
        plt.text((dur_gt[i-1] + dur_gt[i]) / 2 if i > 0 else dur_gt[i] / 2, shift, txt[i],
                 size=16, horizontalalignment="center")
        plt.plot([dur_pred[i], dur_gt[i]], [12, 10], color="black", linewidth=2, linestyle=":")
    plt.yticks([])
    plt.xlim(0, max(dur_pred[-1], dur_gt[-1]))
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def pitch_note_to_figure(pitch_gt, pitch_pred=None, note_midi=None, note_dur=None, note_rest=None, title=None):
    if isinstance(pitch_gt, torch.Tensor):
        pitch_gt = pitch_gt.cpu().numpy()
    if isinstance(pitch_pred, torch.Tensor):
        pitch_pred = pitch_pred.cpu().numpy()
    if isinstance(note_midi, torch.Tensor):
        note_midi = note_midi.cpu().numpy()
    if isinstance(note_dur, torch.Tensor):
        note_dur = note_dur.cpu().numpy()
    if isinstance(note_rest, torch.Tensor):
        note_rest = note_rest.cpu().numpy()
    fig = plt.figure()
    if note_midi is not None and note_dur is not None:
        note_dur_acc = np.cumsum(note_dur)
        if note_rest is None:
            note_rest = np.zeros_like(note_midi, dtype=np.bool_)
        for i in range(len(note_midi)):
            # if note_rest[i]:
            #     continue
            plt.gca().add_patch(
                plt.Rectangle(
                    xy=(note_dur_acc[i-1] if i > 0 else 0, note_midi[i] - 0.5),
                    width=note_dur[i], height=1,
                    edgecolor="grey", fill=False,
                    linewidth=1.5, linestyle="--" if note_rest[i] else "-"
                )
            )
    plt.plot(pitch_gt, color="b", label="gt")
    if pitch_pred is not None:
        plt.plot(pitch_pred, color="r", label="pred")
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis="y")
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def curve_to_figure(curve_gt, curve_pred=None, curve_base=None, grid=None, title=None):
    if isinstance(curve_gt, torch.Tensor):
        curve_gt = curve_gt.cpu().numpy()
    if isinstance(curve_pred, torch.Tensor):
        curve_pred = curve_pred.cpu().numpy()
    if isinstance(curve_base, torch.Tensor):
        curve_base = curve_base.cpu().numpy()
    fig = plt.figure()
    if curve_base is not None:
        plt.plot(curve_base, color="g", label="base")
    plt.plot(curve_gt, color="b", label="gt")
    if curve_pred is not None:
        plt.plot(curve_pred, color="r", label="pred")
    if grid is not None:
        plt.gca().yaxis.set_major_locator(MultipleLocator(grid))
    plt.grid(axis="y")
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def distribution_to_figure(title, x_label, y_label, items: list, values: list, zoom=0.8, rotate=False):
    fig = plt.figure(figsize=(int(len(items) * zoom), 10))
    plt.bar(x=items, height=values)
    plt.tick_params(labelsize=15)
    plt.xlim(-1, len(items))
    for a, b in zip(items, values):
        plt.text(a, b, b, ha="center", va="bottom", fontsize=15)
    plt.grid(axis="y")
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    if rotate:
        fig.autofmt_xdate(rotation=45)
    return fig
