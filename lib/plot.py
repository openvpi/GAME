import math

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
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


def probs_to_figure(
        probs_gt: np.ndarray, probs_pred: np.ndarray,
        title=None
):
    fig = plt.figure(figsize=(12, 6))
    probs_concat = np.concatenate([np.abs(probs_pred - probs_gt), probs_gt, probs_pred], axis=1)
    plt.pcolor(probs_concat.T, vmin=0, vmax=1)
    T, C = probs_gt.shape
    plt.yticks([2.5 * C, 1.5 * C, 0.5 * C], ["pred", "gt", "diff"])
    plt.hlines([C, 2 * C], xmin=0, xmax=T, color="white", linewidth=1.5)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def note_to_figure(
        note_dur, note_midi_gt, note_rest_gt,
        note_midi_pred=None, note_rest_pred=None,
        title=None
):
    fig = plt.figure(figsize=(12, 6))
    note_dur_acc = np.cumsum(note_dur)
    note_height = 0.5

    def draw_notes(note_midi, note_rest, color, label):
        if note_rest is None:
            note_rest = np.zeros_like(note_midi, dtype=np.bool_)
        ys = note_midi[~note_rest]
        x_mins = (note_dur_acc - note_dur)[~note_rest]
        x_maxs = note_dur_acc[~note_rest]
        for i in range(len(ys)):
            plt.gca().add_patch(plt.Rectangle(
                xy=(x_mins[i], ys[i] - note_height / 2),
                width=x_maxs[i] - x_mins[i], height=note_height,
                edgecolor=color, fill=False,
                linewidth=1.5, label=(label if i == 0 else None),
            ))
            plt.fill_between(
                [x_mins[i], x_maxs[i]], ys[i] - note_height / 2, ys[i] + note_height / 2,
                color="none", facecolor=color, alpha=0.2
            )

    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    draw_notes(note_midi_gt, note_rest_gt, color="b", label="gt")
    if note_midi_pred is not None:
        draw_notes(note_midi_pred, note_rest_pred, color="r", label="pred")
    plt.grid(axis="y")
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig
