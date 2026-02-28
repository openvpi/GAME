import abc
import csv
import json
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Literal

import librosa
import lightning.pytorch.callbacks
import matplotlib.pyplot as plt
import mido
import torch
import torch.nn.functional as F

from inference.me_infer_module import InferenceModule
from lib import logging

__all__ = [
    "SaveCombinedFileCallback",
    "SaveCombinedMidiFileCallback",
    "SaveCombinedTextFileCallback",
    "UpdateDiffSingerTranscriptionsCallback",
    "VisualizeNoteComparisonCallback",
    "ExportMetricSummaryCallback",
]

from lib.plot import note_to_figure


@dataclass
class _NoteInfo:
    onset: float
    offset: float
    pitch: float


class SaveCombinedFileCallback(lightning.pytorch.callbacks.Callback, abc.ABC):
    def __init__(self, output_dir: str | pathlib.Path):
        super().__init__()
        if isinstance(output_dir, str):
            output_dir = pathlib.Path(output_dir)
        self.output_dir = output_dir
        self.counters: dict[str, int] = {}
        self.notes: dict[str, list[_NoteInfo]] = {}

    def on_predict_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict[str, torch.Tensor],
            batch: dict[str, Any],
            *args, **kwargs
    ) -> None:
        batch_size = batch["size"]
        for i in range(batch_size):
            key: str = batch["key"][i]
            num_parts = batch["num_parts"][i]
            if key not in self.counters:
                self.counters[key] = 0
                self.notes[key] = []
            offset: float = batch["offset"][i]
            length: float = batch["length"][i]
            durations = outputs["durations"][i]
            scores = outputs["scores"][i]
            presence = outputs["presence"][i]
            note_onset = F.pad(
                durations, (1, 0), mode="constant", value=0
            ).cumsum(dim=0).clamp(max=length).add(offset)
            note_offset = durations.cumsum(dim=0).clamp(max=length).add(offset)
            for onset, offset, score, valid in zip(
                    note_onset.tolist(),
                    note_offset.tolist(),
                    scores.tolist(),
                    presence.tolist(),
            ):
                if offset - onset <= 0:
                    continue
                if not valid:
                    continue
                self.notes[key].append(_NoteInfo(
                    onset=onset,
                    offset=offset,
                    pitch=score,
                ))
            self.counters[key] += 1
            if self.counters[key] >= num_parts:
                self.save_file(key, logger_fn=trainer.progress_bar_callback.print)

    def on_predict_epoch_end(self, trainer: lightning.pytorch.Trainer, *args, **kwargs) -> None:
        for key in self.counters.keys():
            self.save_file(key, logger_fn=trainer.progress_bar_callback.print)

    def save_file(self, key: str, logger_fn: Callable) -> None:
        sorted_notes = sorted(self.notes[key], key=lambda x: (x.onset, x.offset, x.pitch))
        last_time = 0
        i = 0
        while i < len(sorted_notes):
            note = sorted_notes[i]
            note.onset = max(note.onset, last_time)
            note.offset = max(note.offset, note.onset)
            if note.offset <= note.onset:
                sorted_notes.pop(i)
            else:
                last_time = note.offset
                i += 1
        self.flush(key, sorted_notes, logger_fn)
        del self.counters[key]
        del self.notes[key]

    @abc.abstractmethod
    def flush(self, key: str, notes: list[_NoteInfo], logger_fn: Callable) -> None:
        pass


class SaveCombinedMidiFileCallback(SaveCombinedFileCallback):
    def __init__(
            self, output_dir: str | pathlib.Path,
            tempo: int = 120,
    ):
        super().__init__(output_dir)
        self.tempo = tempo

    def flush(self, key: str, notes: list[_NoteInfo], logger_fn: Callable) -> None:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(self.tempo), time=0))
        last_time = 0
        for note in notes:
            onset_ticks = round(note.onset * self.tempo * 8)
            offset_ticks = round(note.offset * self.tempo * 8)
            midi_pitch = round(note.pitch)
            if offset_ticks <= onset_ticks:
                continue
            track.append(mido.Message(
                "note_on", note=midi_pitch, time=onset_ticks - last_time
            ))
            track.append(mido.Message(
                "note_off", note=midi_pitch, time=offset_ticks - onset_ticks
            ))
            last_time = offset_ticks

        filepath = (self.output_dir / key).with_suffix(".mid")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with mido.MidiFile(charset="utf8") as midi_file:
            midi_file.tracks.append(track)
            midi_file.save(filepath)

        logging.info(f"Saved MIDI file: {filepath.as_posix()}", callback=logger_fn)


class SaveCombinedTextFileCallback(SaveCombinedFileCallback):
    def __init__(
            self, output_dir: str | pathlib.Path,
            file_format: Literal["txt", "csv"] = "csv",
            pitch_format: Literal["number", "name"] = "name",
            round_pitch: bool = False,
    ):
        super().__init__(output_dir)
        self.file_format = file_format
        self.pitch_format = pitch_format
        self.round_pitch = round_pitch

    def flush(self, key: str, notes: list[_NoteInfo], logger_fn: Callable) -> None:
        onset_list = [
            f"{note.onset:.3f}" for note in notes
        ]
        offset_list = [
            f"{note.offset:.3f}" for note in notes
        ]
        pitch_list = []
        for note in notes:
            pitch = note.pitch
            if self.round_pitch:
                pitch = round(pitch)
                pitch_txt = str(pitch)
            else:
                pitch_txt = f"{pitch:.3f}"
            if self.pitch_format == "name":
                pitch_txt = librosa.midi_to_note(pitch, unicode=False, cents=not self.round_pitch)
            pitch_list.append(pitch_txt)

        if self.file_format == "txt":
            filepath = (self.output_dir / key).with_suffix(".txt")
            with filepath.open(encoding="utf8", mode="w") as f:
                for onset, offset, pitch in zip(onset_list, offset_list, pitch_list):
                    f.write(f"{onset}\t{offset}\t{pitch}\n")
            logging.info(f"Saved text file: {filepath.as_posix()}", callback=logger_fn)
        elif self.file_format == "csv":
            filepath = (self.output_dir / key).with_suffix(".csv")
            with filepath.open(encoding="utf8", mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["onset", "offset", "pitch"])
                writer.writeheader()
                for onset, offset, pitch in zip(onset_list, offset_list, pitch_list):
                    writer.writerow({
                        "onset": onset,
                        "offset": offset,
                        "pitch": pitch,
                    })
            logging.info(f"Saved CSV file: {filepath.as_posix()}", callback=logger_fn)


class UpdateDiffSingerTranscriptionsCallback(lightning.pytorch.callbacks.Callback):
    def __init__(
            self, filelist: list[pathlib.Path],
            overwrite: bool = False,
            save_dir: str | pathlib.Path = None,
            save_filename: str = "transcriptions-midi.csv",
    ):
        super().__init__()
        self.overwrite = overwrite
        if isinstance(save_dir, str):
            save_dir = pathlib.Path(save_dir)
        self.save_dir = save_dir
        self.save_filename = save_filename
        self.index_map: dict[str, OrderedDict[str, dict[str, Any]]] = {}
        self.lengths: dict[str, int] = {}
        self.counters: dict[str, int] = {}
        for index in filelist:
            with open(index, "r", encoding="utf8") as f:
                items = list(csv.DictReader(f))
                o_dict = OrderedDict()
                for item in items:
                    o_dict[item["name"]] = item
                key = index.as_posix()
                self.index_map[key] = o_dict
                self.lengths[key] = len(items)
                self.counters[key] = 0

    def on_predict_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict[str, torch.Tensor],
            batch: dict[str, Any],
            *args, **kwargs
    ) -> None:
        batch_size = batch["size"]
        for i in range(batch_size):
            index: str = batch["index"][i]
            name: str = batch["name"][i]
            durations = outputs["durations"][i]
            scores = outputs["scores"][i]
            presence = outputs["presence"][i]
            valid = durations > 0
            durations = durations[valid].tolist()
            scores = scores[valid].tolist()
            presence = presence[valid].tolist()
            note_seq = [
                librosa.midi_to_note(score, unicode=False, cents=True) if presence else "rest"
                for score, presence in zip(scores, presence)
            ]
            note_dur = [f"{dur:.3f}" for dur in durations]
            item = self.index_map[index][name]
            item["note_seq"] = " ".join(note_seq)
            item["note_dur"] = " ".join(note_dur)
            item.pop("note_glide", None)
            self.counters[index] += 1
            if self.counters[index] >= self.lengths[index]:
                self.flush(index, logger_fn=trainer.progress_bar_callback.print)

    def on_predict_epoch_end(self, trainer: lightning.pytorch.Trainer, *args, **kwargs) -> None:
        for index in self.counters.keys():
            self.flush(index, logger_fn=trainer.progress_bar_callback.print)

    def flush(self, key: str, logger_fn: Callable):
        items = list(self.index_map[key].values())
        index = pathlib.Path(key)
        if self.overwrite:
            save_path = index
        elif self.save_dir is not None:
            save_path = self.save_dir / self.save_filename
        else:
            save_path = index.parent / self.save_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open(encoding="utf8", mode="w", newline="") as f:
            fieldnames = list(items[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(items)
        del self.index_map[key]
        del self.lengths[key]
        del self.counters[key]
        logging.info(f"Saved transcriptions: {save_path.as_posix()}", callback=logger_fn)


class VisualizeNoteComparisonCallback(lightning.pytorch.callbacks.Callback):
    def __init__(self, save_dir: str | pathlib.Path, digits: int = None):
        super().__init__()
        if isinstance(save_dir, str):
            save_dir = pathlib.Path(save_dir)
        self.save_dir = save_dir
        self.digits = digits

    def on_test_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict[str, torch.Tensor],
            batch: dict[str, Any],
            *args, **kwargs
    ) -> None:
        for i in range(len(batch["indices"])):
            idx = batch["indices"][i].item()
            title = batch["name"][i]
            N = batch["N"][i].item()
            scores_gt = batch["scores"][i, :N]
            presence_gt = batch["presence"][i, :N]
            durations_gt = batch["durations"][i, :N]
            N_pred = outputs["N"][i].item()
            scores_pred = outputs["scores"][i, :N_pred]
            presence_pred = outputs["presence"][i, :N_pred]
            durations_pred = outputs["durations_frame"][i, :N_pred]

            fig = note_to_figure(
                note_midi_gt=scores_gt.cpu().numpy(),
                note_rest_gt=(~presence_gt).cpu().numpy(),
                note_dur_gt=durations_gt.cpu().numpy(),
                note_midi_pred=scores_pred.cpu().numpy(),
                note_rest_pred=(~presence_pred).cpu().numpy(),
                note_dur_pred=durations_pred.cpu().numpy(),
                title=title
            )

            name = str(idx)
            if self.digits is not None:
                name = name.zfill(self.digits)

            save_path = self.save_dir / f"{name}.jpg"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)


class ExportMetricSummaryCallback(lightning.pytorch.callbacks.Callback):
    def __init__(self, save_path: str | pathlib.Path):
        super().__init__()
        if isinstance(save_path, str):
            save_path = pathlib.Path(save_path)
        self.save_path = save_path

    def on_test_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: InferenceModule,
            *args, **kwargs
    ) -> None:
        summary: dict[str, float] = {}
        for name, metric in pl_module.metrics.items():
            value = metric.compute()
            if isinstance(value, dict):
                summary.update(value)
            else:
                summary[name] = value
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with self.save_path.open(encoding="utf8", mode="w") as f:
            json.dump(summary, f, indent=4)
        logging.info(
            f"Saved metric summary: {self.save_path.as_posix()}",
            callback=trainer.progress_bar_callback.print
        )
