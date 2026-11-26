"""Tests for muscriptor/utils/velocity.py."""

import math

import torch

from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note
from muscriptor.utils.velocity import (
    apply_estimated_velocities,
    estimate_note_velocities,
)


def _sine_burst(sr: int, total_s: float, bursts: list[tuple[float, float, float, float]]) -> torch.Tensor:
    """Build mono waveform with (start_s, end_s, freq_hz, amplitude) sine bursts."""
    n = int(sr * total_s)
    wav = torch.zeros(1, n, dtype=torch.float32)
    t = torch.arange(n, dtype=torch.float32) / sr
    for start, end, freq, amp in bursts:
        i0 = int(start * sr)
        i1 = min(n, int(end * sr))
        wav[0, i0:i1] += amp * torch.sin(2.0 * math.pi * freq * t[i0:i1])
    return wav


def test_melodic_louder_note_gets_higher_velocity():
    sr = 16000
    amps = [0.05, 0.2, 0.5, 0.9]
    bursts = [(i * 0.5, i * 0.5 + 0.35, 440.0, a) for i, a in enumerate(amps)]
    wav = _sine_burst(sr, 2.5, bursts)
    notes = [
        Note(False, 0, i * 0.5, i * 0.5 + 0.35, 69)  # A4 = 440 Hz
        for i in range(len(amps))
    ]
    vels = estimate_note_velocities(notes, {"other": (wav, sr)}, ["other"] * len(notes))
    assert len(vels) == 4
    assert all(32 <= v <= 122 for v in vels)
    assert vels[0] < vels[1] < vels[2] < vels[3]
    assert vels[-1] - vels[0] >= 40


def test_drum_lane_band_energy_separates_kick_and_hihat():
    sr = 44100
    n = sr * 2
    wav = torch.zeros(1, n, dtype=torch.float32)
    t = torch.arange(n, dtype=torch.float32) / sr
    # Loud kick at 0.1s (70 Hz), soft kick at 0.5s (70 Hz)
    for onset, amp in [(0.1, 0.9), (0.5, 0.12), (0.9, 0.5)]:
        i0, i1 = int(onset * sr), int((onset + 0.06) * sr)
        wav[0, i0:i1] += amp * torch.sin(2.0 * math.pi * 70.0 * t[i0:i1])
    # Loud hihat at 1.1s (8 kHz), soft hihat at 1.4s (8 kHz), medium at 1.7s
    for onset, amp in [(1.1, 0.6), (1.4, 0.08), (1.7, 0.3)]:
        i0, i1 = int(onset * sr), int((onset + 0.04) * sr)
        wav[0, i0:i1] += amp * torch.sin(2.0 * math.pi * 8000.0 * t[i0:i1])

    notes = [
        Note(True, DRUM_PROGRAM, 0.1, 0.11, 36),
        Note(True, DRUM_PROGRAM, 0.5, 0.51, 36),
        Note(True, DRUM_PROGRAM, 0.9, 0.91, 36),
        Note(True, DRUM_PROGRAM, 1.1, 1.11, 42),
        Note(True, DRUM_PROGRAM, 1.4, 1.41, 42),
        Note(True, DRUM_PROGRAM, 1.7, 1.71, 42),
    ]
    vels = estimate_note_velocities(notes, {"drums": (wav, sr)}, ["drums"] * len(notes))
    assert vels[0] - vels[1] >= 25
    assert vels[3] - vels[4] >= 25


def test_uniform_loudness_track_does_not_amplify_noise():
    sr = 16000
    bursts = [(i * 0.4, i * 0.4 + 0.25, 440.0, 0.4) for i in range(6)]
    wav = _sine_burst(sr, 3.0, bursts)
    notes = [Note(False, 0, i * 0.4, i * 0.4 + 0.25, 69) for i in range(6)]
    vels = estimate_note_velocities(notes, {"mix": (wav, sr)})
    assert max(vels) - min(vels) <= 5


def test_empty_silent_and_few_notes_safe():
    sr = 16000
    silent = torch.zeros(2, sr, dtype=torch.float32)
    assert estimate_note_velocities([], {"mix": (silent, sr)}) == []

    silent_notes = [
        Note(False, 0, 0.1, 0.3, 60),
        Note(False, 0, 0.4, 0.6, 64),
        Note(True, DRUM_PROGRAM, 0.2, 0.21, 36),
    ]
    vels = estimate_note_velocities(silent_notes, {"mix": (silent, sr)})
    assert len(vels) == 3
    assert all(isinstance(v, int) and 1 <= v <= 127 for v in vels)

    # Fewer than 3 notes on active audio anchors cleanly against stem RMS
    active = _sine_burst(sr, 1.0, [(0.1, 0.4, 440.0, 0.5), (0.5, 0.8, 440.0, 0.25)])
    few_notes = [Note(False, 0, 0.1, 0.4, 69), Note(False, 0, 0.5, 0.8, 69)]
    applied = apply_estimated_velocities(few_notes, {"mix": (active, sr)})
    assert applied[0].velocity > applied[1].velocity
    assert all(32 <= n.velocity <= 122 for n in applied)
    assert few_notes[0].velocity == 100  # original not mutated
