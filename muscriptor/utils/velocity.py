"""Audio-derived per-note MIDI velocity estimation via STFT spectral energy."""

import dataclasses
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.cleanup import drum_lane_for_pitch

# Frequency bands (Hz) for transient drum energy estimation.
_DRUM_BANDS_HZ: dict[str, tuple[float, float]] = {
    "kick": (35.0, 140.0),
    "snare": (150.0, 2500.0),
    "toms": (80.0, 450.0),
    "hihat": (4000.0, 14000.0),
    "cymbals": (3000.0, 14000.0),
    "perc": (200.0, 6000.0),
}

_HARMONIC_WEIGHTS: tuple[tuple[int, float], ...] = (
    (1, 1.0),
    (2, 0.5),
    (3, 0.25),
)

_MIN_DYNAMIC_SPAN_DB = 24.0
_SILENCE_RMS_THRESHOLD = 1e-6
_EPS = 1e-8


def _to_mono_1d(wav: torch.Tensor) -> torch.Tensor:
    t = wav.detach().cpu().float()
    if t.dim() == 3:
        t = t.squeeze(0)
    if t.dim() == 2:
        t = t.mean(dim=0)
    return t.contiguous()


def _compute_stft_mag(
    mono: torch.Tensor, n_fft: int, hop_length: int
) -> np.ndarray:
    """Return amplitude-normalized magnitude spectrogram of shape [freq_bins, frames]."""
    if mono.numel() == 0:
        return np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)
    if mono.numel() < n_fft:
        mono = torch.nn.functional.pad(mono, (0, n_fft - mono.numel()))
    window = torch.hann_window(n_fft, dtype=torch.float32)
    spec = torch.stft(
        mono,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True,
    )
    scale = float(window.sum().item()) / 2.0
    mag = (spec.abs() / max(scale, _EPS)).numpy()
    return mag


@dataclasses.dataclass
class _StemAnalysis:
    sr: int
    rms_db: float
    is_silent: bool
    melodic_mag: np.ndarray | None = None
    melodic_n_fft: int = 2048
    melodic_hop: int = 160
    drum_mag: np.ndarray | None = None
    drum_n_fft: int = 512
    drum_hop: int = 80
    mono: torch.Tensor | None = None

    def get_melodic(self) -> tuple[np.ndarray, int, int]:
        if self.melodic_mag is None:
            self.melodic_n_fft = 2048 if self.sr <= 24000 else 4096
            self.melodic_hop = max(1, round(self.sr * 0.01))
            self.melodic_mag = _compute_stft_mag(
                self.mono, self.melodic_n_fft, self.melodic_hop
            )
        return self.melodic_mag, self.melodic_n_fft, self.melodic_hop

    def get_drum(self) -> tuple[np.ndarray, int, int]:
        if self.drum_mag is None:
            self.drum_n_fft = 512 if self.sr <= 24000 else 1024
            self.drum_hop = max(1, round(self.sr * 0.005))
            self.drum_mag = _compute_stft_mag(
                self.mono, self.drum_n_fft, self.drum_hop
            )
        return self.drum_mag, self.drum_n_fft, self.drum_hop


def _frame_slice(start_s: float, end_s: float, sr: int, hop: int, n_frames: int) -> slice:
    f0 = max(0, min(n_frames - 1, int(math.floor(start_s * sr / hop))))
    f1 = max(f0 + 1, min(n_frames, int(math.ceil(end_s * sr / hop)) + 1))
    return slice(f0, f1)


def _melodic_energy_db(note: Note, analysis: _StemAnalysis) -> float:
    mag, n_fft, hop = analysis.get_melodic()
    n_bins, n_frames = mag.shape
    nyquist = analysis.sr / 2.0
    bin_hz = analysis.sr / n_fft

    attack_end = max(note.onset + 0.02, min(note.offset, note.onset + 0.08))
    frames = _frame_slice(note.onset - 0.01, attack_end, analysis.sr, hop, n_frames)

    f0 = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
    semitone_factor = 2.0 ** (0.6 / 12.0)

    total = np.zeros(frames.stop - frames.start, dtype=np.float64)
    weight_sum = 0.0
    for harmonic, weight in _HARMONIC_WEIGHTS:
        freq = f0 * harmonic
        if freq >= nyquist * 0.95:
            continue
        lo_bin = max(1, int(math.floor((freq / semitone_factor) / bin_hz)))
        hi_bin = min(n_bins - 1, int(math.ceil((freq * semitone_factor) / bin_hz)))
        if hi_bin < lo_bin:
            continue
        # Peak bin magnitude per frame captures sinusoidal harmonic energy cleanly.
        band = mag[lo_bin : hi_bin + 1, frames]
        total += weight * band.max(axis=0)
        weight_sum += weight

    if weight_sum <= 0.0:
        return -120.0
    energy = float(np.percentile(total / weight_sum, 80))
    return float(20.0 * math.log10(energy + _EPS))


def _drum_energy_db(note: Note, analysis: _StemAnalysis) -> float:
    mag, n_fft, hop = analysis.get_drum()
    n_bins, n_frames = mag.shape
    nyquist = analysis.sr / 2.0
    bin_hz = analysis.sr / n_fft

    lane = drum_lane_for_pitch(note.pitch)
    lo_hz, hi_hz = _DRUM_BANDS_HZ[lane]
    hi_hz = min(hi_hz, nyquist * 0.95)
    lo_hz = min(lo_hz, hi_hz * 0.8)

    lo_bin = max(1, int(math.floor(lo_hz / bin_hz)))
    hi_bin = min(n_bins - 1, max(lo_bin, int(math.ceil(hi_hz / bin_hz))))

    frames = _frame_slice(
        note.onset - 0.005, note.onset + 0.035, analysis.sr, hop, n_frames
    )
    band = mag[lo_bin : hi_bin + 1, frames]
    # RMS across frequency bins per frame, then peak across the transient window.
    frame_rms = np.sqrt(np.mean(band.astype(np.float64) ** 2, axis=0))
    peak = float(frame_rms.max()) if frame_rms.size else 0.0
    return float(20.0 * math.log10(peak + _EPS))


def estimate_note_velocities(
    notes: Sequence[Note],
    audio_by_stem: Mapping[str, tuple[torch.Tensor, int]],
    note_stems: Sequence[str] | None = None,
    vel_min: int = 32,
    vel_max: int = 122,
    default_velocity: int = 100,
) -> list[int]:
    """Estimate MIDI velocity (1..127) for each note from its source audio stem."""
    if not notes:
        return []
    if not audio_by_stem:
        fallback = max(1, min(127, int(default_velocity)))
        return [fallback] * len(notes)

    vel_min = max(1, min(127, int(vel_min)))
    vel_max = max(vel_min, min(127, int(vel_max)))
    fallback_stem = "mix" if "mix" in audio_by_stem else next(iter(audio_by_stem))

    analyses: dict[str, _StemAnalysis] = {}
    for stem_name, (wav, sr) in audio_by_stem.items():
        mono = _to_mono_1d(wav)
        rms = float(torch.sqrt(torch.mean(mono * mono)).item()) if mono.numel() else 0.0
        is_silent = rms < _SILENCE_RMS_THRESHOLD
        rms_db = float(20.0 * math.log10(rms + _EPS)) if not is_silent else -120.0
        analyses[stem_name] = _StemAnalysis(
            sr=int(sr), rms_db=rms_db, is_silent=is_silent, mono=mono
        )

    raw_db: list[float] = []
    groups: dict[tuple[str, bool, int, str], list[int]] = defaultdict(list)

    for idx, note in enumerate(notes):
        stem_key = (
            note_stems[idx]
            if (note_stems is not None and idx < len(note_stems) and note_stems[idx] in analyses)
            else fallback_stem
        )
        analysis = analyses[stem_key]
        if analysis.is_silent:
            db = -120.0
        elif note.is_drum:
            db = _drum_energy_db(note, analysis)
        else:
            db = _melodic_energy_db(note, analysis)
        raw_db.append(db)
        lane = drum_lane_for_pitch(note.pitch) if note.is_drum else ""
        groups[(stem_key, note.is_drum, note.program, lane)].append(idx)

    velocities = [max(1, min(127, int(default_velocity)))] * len(notes)
    span = float(vel_max - vel_min)

    for (stem_key, _is_drum, _prog, _lane), indices in groups.items():
        analysis = analyses[stem_key]
        if analysis.is_silent:
            for i in indices:
                velocities[i] = max(vel_min, min(vel_max, int(default_velocity)))
            continue

        vals = np.array([raw_db[i] for i in indices], dtype=np.float64)
        if len(indices) >= 3:
            p5, p95 = np.percentile(vals, [5.0, 95.0])
            hi = float(p95)
            lo = min(float(p5), hi - _MIN_DYNAMIC_SPAN_DB)
            if (p95 - p5) < 1.0:
                # Nearly constant loudness: anchor against stem RMS so uniform tracks
                # stay steady without slamming everything to vel_max.
                hi = max(hi, analysis.rms_db + 12.0)
                lo = min(lo, analysis.rms_db - _MIN_DYNAMIC_SPAN_DB)
        else:
            # Fewer than 3 notes: anchor against the stem's overall RMS.
            hi = max(float(vals.max()) + 3.0, analysis.rms_db + 12.0)
            lo = min(float(vals.min()) - 6.0, analysis.rms_db - _MIN_DYNAMIC_SPAN_DB)

        denom = max(hi - lo, _MIN_DYNAMIC_SPAN_DB)
        for i in indices:
            norm = max(0.0, min(1.0, (raw_db[i] - lo) / denom))
            vel = round(vel_min + norm * span)
            velocities[i] = max(1, min(127, int(vel)))

    return velocities


def apply_estimated_velocities(
    notes: Sequence[Note],
    audio_by_stem: Mapping[str, tuple[torch.Tensor, int]],
    note_stems: Sequence[str] | None = None,
    vel_min: int = 32,
    vel_max: int = 122,
    default_velocity: int = 100,
) -> list[Note]:
    """Return shallow-copied `Note` objects with audio-derived velocities applied."""
    vels = estimate_note_velocities(
        notes,
        audio_by_stem,
        note_stems=note_stems,
        vel_min=vel_min,
        vel_max=vel_max,
        default_velocity=default_velocity,
    )
    return [dataclasses.replace(n, velocity=v) for n, v in zip(notes, vels)]
