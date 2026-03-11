from typing import Literal

import numpy as np


def validate_phones(
        ph_seq: list[str],
        ph_dur: list[float],
        ph_num: list[int],
) -> tuple[bool, str | None]:
    """
    Validate the phoneme sequence, durations, and word spans.
    :param ph_seq: list of phoneme symbols
    :param ph_dur: list of phoneme durations
    :param ph_num: list of word spans (number of phonemes in each word)
    :return: is_valid, error_message
    """
    if len(ph_seq) != len(ph_dur):
        return False, f"Length mismatch: {len(ph_seq)} phonemes vs {len(ph_dur)} durations."
    if sum(ph_num) != len(ph_seq):
        return False, f"Word span mismatch: sum of {ph_num} is {sum(ph_num)}, expected {len(ph_seq)}."
    return True, None


def parse_words(
        ph_seq: list[str],
        ph_dur: list[float],
        ph_num: list[int],
        uv_vocab: set[str] = None,
        uv_cond: Literal["lead", "all"] = "all",
        merge_consecutive_uv: bool = False,
) -> tuple[list[float], list[int]]:
    """
    Convert phoneme sequence to word durations and v/uv flags.
    :param ph_seq: list of phoneme symbols
    :param ph_dur: list of phoneme durations
    :param ph_num: list of word spans (number of phonemes in each word)
    :param uv_vocab: set of unvoiced phonemes
    :param uv_cond: condition for determining unvoiced words ("lead" for leading phonemes, "all" for all phonemes)
    :param merge_consecutive_uv: whether to merge consecutive unvoiced words into one
    :return: word_dur, word_vuv (1 for voiced, 0 for unvoiced)
    """
    word_dur = []
    word_vuv = []
    idx = 0
    for num in ph_num:
        dur_sum = sum(ph_dur[idx:idx + num])
        word_dur.append(dur_sum)
        vuv = 1
        if uv_vocab is not None:
            if uv_cond == "lead":
                if ph_seq[idx] in uv_vocab:
                    vuv = 0
            elif uv_cond == "all":
                if all(ph in uv_vocab for ph in ph_seq[idx:idx + num]):
                    vuv = 0
        else:
            vuv = 1
        word_vuv.append(vuv)
        idx += num
    if merge_consecutive_uv:
        word_dur, word_vuv = merge_consecutive_uv_words(word_dur, word_vuv)
    return word_dur, word_vuv


def merge_consecutive_uv_words(
        word_dur: list[float],
        word_vuv: list[int],
) -> tuple[list[float], list[int]]:
    """
    Merge consecutive unvoiced words into one.
    :param word_dur: list of word durations
    :param word_vuv: list of word v/uv flags (1 for voiced, 0 for unvoiced)
    :return: merged_word_dur, merged_word_vuv
    """
    if not word_dur:
        return [], []
    merged_dur = [word_dur[0]]
    merged_vuv = [word_vuv[0]]
    for dur, vuv in zip(word_dur[1:], word_vuv[1:]):
        if vuv == 0 and merged_vuv[-1] == 0:
            merged_dur[-1] += dur
        else:
            merged_dur.append(dur)
            merged_vuv.append(vuv)
    return merged_dur, merged_vuv


def align_notes_to_words(
        word_dur: list[float],
        word_vuv: list[int],
        note_seq: list[str],
        note_dur: list[float],
        tol: float = 0.01,
        apply_word_uv: bool = False
) -> tuple[list[str], list[float], list[int]]:
    """
    Align note sequence to word durations.
    :param word_dur: list of word durations
    :param word_vuv: list of word v/uv flags (1 for voiced, 0 for unvoiced)
    :param note_seq: list of note names (e.g. "C4", "D#4", "rest")
    :param note_dur: list of note durations
    :param tol: tolerance for alignment (in seconds)
    :param apply_word_uv: whether to set note pitch to "rest" for unvoiced words
    :return: new_note_seq, new_note_dur, note_slur (1 for slur, 0 for non-slur)
    """
    word_start = np.cumsum([0.0] + word_dur[:-1])
    word_end = np.cumsum(word_dur)
    note_start = np.cumsum([0.0] + note_dur[:-1])
    note_end = np.cumsum(note_dur)
    new_note_seq = []
    new_note_dur = []
    note_slur = []
    for word_idx in range(len(word_dur)):
        if apply_word_uv and word_vuv[word_idx] == 0:
            # unvoiced word, set all notes to rest
            new_note_seq.append("rest")
            new_note_dur.append(word_dur[word_idx])
            note_slur.append(0)
            continue
        # find the closest note start
        note_start_idx = np.argmin(np.abs(note_start - word_start[word_idx]))
        if word_start[word_idx] < note_start[note_start_idx] - tol:
            note_start_idx = max(0, note_start_idx - 1)
        # find the closest note end
        note_end_idx = np.argmin(np.abs(note_end[note_start_idx:] - word_end[word_idx])) + note_start_idx
        if word_end[word_idx] > note_end[note_end_idx] + tol:
            note_end_idx = min(len(note_end) - 1, note_end_idx + 1)
        # adjust note sequence and durations to fit the word duration
        word_note_seq = []
        word_note_dur = []
        for note_idx in range(note_start_idx, note_end_idx + 1):
            # adjust note start
            if note_idx == note_start_idx:
                start = word_start[word_idx]
            else:
                start = note_start[note_idx]
            # adjust note end
            if note_idx == note_end_idx:
                end = word_end[word_idx]
            else:
                end = note_end[note_idx]
            if word_note_seq and word_note_seq[-1] == note_seq[note_idx]:
                # same note as previous, merge durations
                word_note_dur[-1] += (end - start)
            else:
                word_note_seq.append(note_seq[note_idx])
                word_note_dur.append(end - start)
        new_note_seq.extend(word_note_seq)
        new_note_dur.extend(word_note_dur)
        note_slur.extend([0] + [1] * (len(word_note_seq) - 1))
    return new_note_seq, new_note_dur, note_slur
