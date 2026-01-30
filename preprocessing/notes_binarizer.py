import csv
import pathlib
from dataclasses import dataclass

import dask
import librosa
import numpy
import torch
from scipy import interpolate

from lib import logging
from lib.feature.mel import StretchableMelSpectrogram
from .binarizer_base import BaseBinarizer, MetadataItem, DataSample, find_waveform_file

NOTES_ITEM_ATTRIBUTES = [
    "language_id",  # int64
    "spectrogram",  # float32 [T, C]
    "durations",  # int64 [N,] note durations in frames
    "regions",  # int64 [T,] note regions mapping frame -> note index
    "boundaries",  # bool [T,] note boundaries
    "scores",  # float32 [N,] MIDI pitch of notes
    "presence",  # bool [N,] presence of notes, 0 means rest
]


@dataclass
class NotesMetadataItem(MetadataItem):
    note_scores: list[str]
    note_durations: list[float]


class NotesBinarizer(BaseBinarizer):
    __data_attrs__ = NOTES_ITEM_ATTRIBUTES

    def load_metadata(self, subset_dir: pathlib.Path) -> list[MetadataItem]:
        with open(subset_dir / "index.csv", "r", encoding="utf8") as f:
            items = list(csv.DictReader(f))
        metadata_items = []
        for item in items:
            name = item["name"]
            language = item.get("language")
            waveform_fn = find_waveform_file(subset_dir, name)
            if waveform_fn is None:
                continue
            notes = item["notes"].split()
            durations = [float(dur) for dur in item["durations"].split()]
            if not (len(notes) == len(durations)):
                logging.error(
                    f"Length mismatch in raw dataset \'{subset_dir.as_posix()}\': "
                    f"item \'{name}\', notes({len(notes)}), durations({len(durations)})."
                )
                continue
            estimated_duration = sum(durations)
            metadata_items.append(NotesMetadataItem(
                name=name,
                language=language,
                waveform_fn=waveform_fn,
                estimated_duration=estimated_duration,
                note_scores=notes,
                note_durations=durations,
            ))
        return metadata_items

    def process_item(self, item: NotesMetadataItem) -> DataSample:
        language_id = numpy.array(self.lang_map.get(item.language, 0), dtype=numpy.int64)
        waveform = self.load_waveform(item.waveform_fn)
        spectrogram, length = self.get_mel(waveform)
        note_dur_sec = numpy.array(item.note_durations, dtype=numpy.float32)
        note_dur_frames = self.sec_dur_to_frame_dur(note_dur_sec, length)
        frame2note = self.length_regulator(note_dur_frames)
        note_boundaries = self.regions_to_boundaries(frame2note)
        note_midi = numpy.array(
            [(librosa.note_to_midi(n, round_midi=False) if n != "rest" else -1) for n in item.note_scores],
            dtype=numpy.float32
        )
        note_rest = note_midi < 0
        note_midi_interp = self.interpolate_rest(note_midi, note_rest)
        data = {
            "language_id": language_id,
            "spectrogram": spectrogram,
            "durations": note_dur_frames,
            "regions": frame2note,
            "boundaries": note_boundaries,
            "scores": note_midi_interp,
            "presence": ~note_rest,
        }
        data, length = dask.compute(data, length)
        return DataSample(
            path=item.waveform_fn.relative_to(self.data_dir).as_posix(),
            name=item.name,
            length=length,
            data=data,
        )

    @dask.delayed
    def load_waveform(self, wav_fn: pathlib.Path):
        waveform, _ = librosa.load(wav_fn, sr=self.config.features.audio_sample_rate, mono=True)
        return waveform

    @dask.delayed(nout=2)
    @torch.no_grad()
    def get_mel(self, waveform: numpy.ndarray):
        if self.mel_spec is None:
            self.mel_spec = StretchableMelSpectrogram(
                sample_rate=self.config.features.audio_sample_rate,
                n_mels=self.config.features.spectrogram.num_bins,
                n_fft=self.config.features.fft_size,
                win_length=self.config.features.win_size,
                hop_length=self.config.features.hop_size,
                fmin=self.config.features.spectrogram.fmin,
                fmax=self.config.features.spectrogram.fmax,
                clip_val=1e-9,
            ).eval().to(self.device)
        mel = self.mel_spec(
            torch.from_numpy(waveform).to(self.device).unsqueeze(0),
        ).squeeze(0).T.cpu().numpy()
        return mel, mel.shape[0]

    @dask.delayed
    def sec_dur_to_frame_dur(self, dur_sec: numpy.ndarray, length: int):
        dur_cumsum = numpy.round(numpy.cumsum(dur_sec, axis=0) / self.timestep).astype(numpy.int64)
        dur_cumsum = numpy.clip(dur_cumsum, a_min=0, a_max=length)
        dur_cumsum[-1] = length
        dur_frame = numpy.diff(dur_cumsum, axis=0, prepend=numpy.array([0]))
        return dur_frame

    @dask.delayed
    def length_regulator(self, dur_frames: numpy.ndarray):
        return self.lr(torch.from_numpy(dur_frames).long().unsqueeze(0)).squeeze(0).numpy()

    @dask.delayed
    def regions_to_boundaries(self, regions: numpy.ndarray):
        boundaries = numpy.diff(regions, axis=0, prepend=numpy.array([1])) > 0
        return boundaries

    @dask.delayed
    def interpolate_rest(self, note_midi: numpy.ndarray, note_rest: numpy.ndarray) -> numpy.ndarray:
        interp_func = interpolate.interp1d(
            numpy.where(~note_rest)[0], note_midi[~note_rest],
            kind='nearest', fill_value='extrapolate'
        )
        interp_midi = note_midi.copy()
        interp_midi[note_rest] = interp_func(numpy.where(note_rest)[0])
        return interp_midi
