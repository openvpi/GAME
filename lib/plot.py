import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def spectrogram_to_figure(spectrogram, title=None):
    fig = plt.figure(figsize=(12, 3))
    plt.pcolormesh(spectrogram.T, vmin=-14, vmax=4)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def similarity_to_figure(similarities, durations, title=None):
    dur_cumsum = np.cumsum(durations)
    fig = plt.figure(figsize=(9, 9))
    plt.pcolormesh(similarities, vmin=-1, vmax=1)
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


def boundary_to_figure(
        boundaries_gt: np.ndarray, boundaries_pred: np.ndarray,
        threshold: float = None,
        boundaries_tp: np.ndarray = None,
        boundaries_fp: np.ndarray = None,
        boundaries_fn: np.ndarray = None,
        title=None
):
    figure_width = 12
    figure_height = 6
    fig = plt.figure(figsize=(figure_width, figure_height))
    plt.plot(boundaries_gt, color="b", label="gt")
    plt.plot(boundaries_pred, color="r", label="pred")
    if threshold is not None:
        plt.plot([0, boundaries_gt.shape[0]], [threshold, threshold], color="black", linestyle="--")
    positions = np.arange(boundaries_gt.shape[0], dtype=np.int64)
    circle_radius = 10
    x_min = 0
    x_max = boundaries_gt.shape[0]
    y_min = 0
    y_max = 1.1
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
        _draw_circles(positions[boundaries_tp], boundaries_pred, "green", "match")
    if boundaries_fp is not None:
        _draw_circles(positions[boundaries_fp], boundaries_pred, "orange", "exceed")
    if boundaries_fn is not None:
        _draw_circles(positions[boundaries_fn], boundaries_gt, "grey", "miss")
    plt.xlim(-1, boundaries_gt.shape[0])
    plt.ylim(y_min, y_max)
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
    plt.pcolormesh(probs_concat.T, vmin=0, vmax=1)
    T, C = probs_gt.shape
    plt.yticks([2.5 * C, 1.5 * C, 0.5 * C], ["pred", "gt", "diff"])
    plt.hlines([C, 2 * C], xmin=0, xmax=T, color="white", linewidth=1.5)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def note_to_figure(
        note_midi_gt, note_rest_gt, note_dur_gt,
        note_midi_pred=None, note_rest_pred=None, note_dur_pred=None,
        title=None
):
    fig = plt.figure(figsize=(12, 6))
    note_height = 0.5

    def draw_notes(note_midi, note_rest, note_dur, color, label):
        note_dur_acc = np.cumsum(note_dur)
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

    draw_notes(note_midi_gt, note_rest_gt, note_dur_gt, color="b", label="gt")
    x_max = note_dur_gt.sum()
    if note_midi_pred is not None:
        draw_notes(note_midi_pred, note_rest_pred, note_dur_pred, color="r", label="pred")
        x_max = max(x_max, note_dur_pred.sum())

    plt.xlim(0, x_max)
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis="y")
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig
