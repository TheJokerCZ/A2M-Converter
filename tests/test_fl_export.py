"""Tests for muscriptor/utils/fl_export.py."""

import io
import wave
import zipfile

import mido
import numpy as np
import pytest
import torch

from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note
from muscriptor.utils.beats import BeatGrid, read_bar_offset
from muscriptor.utils.cleanup import C5_PITCH
from muscriptor.utils.fl_export import (
    align_wav_to_offset,
    build_fl_bundle,
    write_pcm16_wav_bytes,
)
from muscriptor.utils.midi import ExportOptions


def _read_midi_bytes(data: bytes) -> mido.MidiFile:
    return mido.MidiFile(file=io.BytesIO(data))


def _wav_frame_count(data: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnchannels(), wf.getframerate(), wf.getnframes()


def test_align_wav_to_offset_pads_and_trims_exact_samples():
    sr = 44100
    stereo = torch.ones(2, sr, dtype=torch.float32) * 0.5
    padded = align_wav_to_offset(stereo, sr, offset_s=0.5)
    assert padded.shape == (2, sr + sr // 2)
    assert float(padded[:, : sr // 2].abs().max().item()) == 0.0
    assert torch.allclose(padded[:, sr // 2 :], stereo)

    mono = torch.linspace(-0.5, 0.5, sr, dtype=torch.float32).unsqueeze(0)
    trimmed = align_wav_to_offset(mono, sr, offset_s=-0.25)
    cut = round(0.25 * sr)
    assert trimmed.shape == (1, sr - cut)
    assert torch.allclose(trimmed, mono[:, cut:])

    # Extreme negative offset beyond audio length must not crash or negative-slice
    empty = align_wav_to_offset(mono, sr, offset_s=-5.0)
    assert empty.shape == (1, 0)


def test_build_fl_bundle_structure_c5_drums_and_aligned_stems():
    sr = 44100
    notes = [
        Note(False, 0, 0.5, 1.0, 60, velocity=84),
        Note(False, 33, 0.5, 1.0, 40, velocity=96),
        Note(True, DRUM_PROGRAM, 0.5, 0.51, 36, velocity=120),  # kick
        Note(True, DRUM_PROGRAM, 1.0, 1.01, 38, velocity=105),  # snare
        Note(True, DRUM_PROGRAM, 1.25, 1.26, 45, velocity=90),  # tom
    ]
    program_names = {0: "acoustic piano", 33: "electric bass", DRUM_PROGRAM: "drums"}
    grid = BeatGrid(
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat=0.5,
        onset_delay=0.0,
        beat_subdivision=4,
        steady=True,
    )
    stems = {
        "drums": (torch.ones(2, sr * 2, dtype=torch.float32) * 0.25, sr),
        "bass": (torch.ones(2, sr * 2, dtype=torch.float32) * 0.2, sr),
    }

    zip_bytes = build_fl_bundle(
        notes,
        program_names=program_names,
        beat_grid=grid,
        options=ExportOptions(quantize=True, split_drums=True),
        stem_audio=stems,
        include_stems=True,
        normalize_drum_lanes_to_c5=True,
        song_name="my_banger",
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())
        assert "00_full_mix.mid" in names
        assert "README_FL_STUDIO.txt" in names
        assert "midi/01_acoustic_piano.mid" in names
        assert "midi/02_electric_bass.mid" in names
        assert "midi/drums_kick.mid" in names
        assert "midi/drums_snare.mid" in names
        assert "midi/drums_toms.mid" in names
        assert "stems/drums.wav" in names
        assert "stems/bass.wav" in names

        full_mid = _read_midi_bytes(zf.read("00_full_mix.mid"))
        kick_mid = _read_midi_bytes(zf.read("midi/drums_kick.mid"))
        snare_mid = _read_midi_bytes(zf.read("midi/drums_snare.mid"))
        toms_mid = _read_midi_bytes(zf.read("midi/drums_toms.mid"))
        piano_mid = _read_midi_bytes(zf.read("midi/01_acoustic_piano.mid"))

        # Single-hit lanes normalized to C5 (MIDI 60) on channel 0 for FL Sampler
        kick_ons = [m for t in kick_mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
        assert [(m.note, m.velocity, m.channel) for m in kick_ons] == [(C5_PITCH, 120, 0)]
        snare_ons = [m for t in snare_mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
        assert [(m.note, m.velocity, m.channel) for m in snare_ons] == [(C5_PITCH, 105, 0)]

        # Multi-pitch tom lane & combined MIDI preserve original GM pitches
        tom_ons = [m for t in toms_mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
        assert [(m.note, m.velocity) for m in tom_ons] == [(45, 90)]
        full_drum_ons = [
            (m.note, m.channel)
            for t in full_mid.tracks
            for m in t
            if m.type == "note_on" and m.velocity > 0 and m.channel == 9
        ]
        assert (36, 9) in full_drum_ons
        assert (38, 9) in full_drum_ons

        # Every MIDI shares exact same bar offset and tempo
        expected_offset = read_bar_offset(full_mid)
        assert expected_offset == pytest.approx(1.5, abs=0.001)
        assert read_bar_offset(piano_mid) == pytest.approx(expected_offset, abs=0.001)
        assert read_bar_offset(kick_mid) == pytest.approx(expected_offset, abs=0.001)

        # Aligned WAV stems have exact silence prepended
        ch, out_sr, frames = _wav_frame_count(zf.read("stems/drums.wav"))
        assert ch == 2
        assert out_sr == sr
        assert frames == sr * 2 + round(expected_offset * sr)


def test_build_fl_bundle_c5_disabled_keeps_gm_pitches():
    notes = [Note(True, DRUM_PROGRAM, 0.5, 0.51, 36, velocity=115)]
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.0, onset_delay=0.0)
    zip_bytes = build_fl_bundle(
        notes,
        beat_grid=grid,
        options=ExportOptions(split_drums=True),
        include_stems=False,
        normalize_drum_lanes_to_c5=False,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        kick_mid = _read_midi_bytes(zf.read("midi/drums_kick.mid"))
        ons = [m for t in kick_mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
        assert [(m.note, m.channel) for m in ons] == [(36, 9)]


def test_build_fl_bundle_variable_tempo_adds_warning_in_readme():
    beats = np.array([0.0, 0.5, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
    grid = BeatGrid(
        bpm=130.0,
        beats_per_bar=4,
        first_downbeat=0.0,
        onset_delay=0.0,
        beats=beats,
        steady=False,
    )
    notes = [Note(False, 0, 0.5, 1.0, 60, velocity=90)]
    zip_bytes = build_fl_bundle(
        notes,
        beat_grid=grid,
        options=ExportOptions(tempo_map=True),
        include_stems=False,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        readme = zf.read("README_FL_STUDIO.txt").decode("utf-8")
        assert "variable tempo" in readme.lower() or "tempo map" in readme.lower()


def test_build_fl_bundle_empty_notes_after_cleanup():
    notes = [Note(False, 0, 0.1, 0.4, 60, velocity=40)]
    grid = BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.0, onset_delay=0.0)
    zip_bytes = build_fl_bundle(
        notes,
        beat_grid=grid,
        options=ExportOptions(min_velocity=127, trim_leading_bars=True),
        include_stems=False,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())
        assert "00_full_mix.mid" in names
        assert "README_FL_STUDIO.txt" in names
        mid = _read_midi_bytes(zf.read("00_full_mix.mid"))
        assert isinstance(mid, mido.MidiFile)
