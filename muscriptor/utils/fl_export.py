"""FL Studio ZIP bundle exporter: multi-track MIDI, per-instrument MIDIs,
optional C5-normalized drum lanes, and bar-1-aligned WAV stems."""

import dataclasses
import io
import re
import wave
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    Note,
    note2note_event,
    note_event2midi,
)
from muscriptor.utils.beats import BeatGrid
from muscriptor.utils.cleanup import (
    C5_PITCH,
    DRUM_LANE_LABELS,
    DRUM_LANE_ORDER,
    SINGLE_HIT_DRUM_LANES,
    cleanup_notes,
    drum_lane_for_pitch,
)
from muscriptor.utils.midi import (
    DRUM_HIT_BEATS,
    TICKS_PER_BEAT,
    ExportOptions,
    gaps_filled,
    leading_bars,
    quantized_notes,
    resolve_grid,
    shifted_notes,
    timeline_for,
)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "track"


def align_wav_to_offset(
    wav: torch.Tensor, sr: int, offset_s: float
) -> torch.Tensor:
    """Shift a waveform `[C, T]` or `[T]` so its sample 0 lands at MIDI tick 0.

    - `offset_s > 0`: prepends `round(offset_s * sr)` frames of silence so audio
      `t = 0` plays at the exact beat offset recorded in the MIDI.
    - `offset_s < 0`: trims `round(-offset_s * sr)` leading frames (clamped so
      trimming past the end safely returns `[C, 0]`).
    """
    t = wav.detach().cpu().float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    elif t.dim() == 3:
        t = t.squeeze(0)

    shift = round(float(offset_s) * int(sr))
    if shift > 0:
        pad = torch.zeros((t.shape[0], shift), dtype=torch.float32)
        return torch.cat([pad, t], dim=1)
    if shift < 0:
        cut = min(t.shape[1], -shift)
        return t[:, cut:].contiguous()
    return t.contiguous()


def write_pcm16_wav_bytes(wav: torch.Tensor, sr: int) -> bytes:
    """Encode float tensor `[C, T]` or `[T]` in `[-1, 1]` as 16-bit PCM WAV bytes."""
    t = wav.detach().cpu().float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    elif t.dim() == 3:
        t = t.squeeze(0)
    n_channels = int(t.shape[0])
    clipped = torch.clamp(t, -1.0, 1.0).numpy()
    pcm = np.round(clipped.T * 32767.0).astype("<i2")  # [T, C] little-endian int16

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _midi_to_bytes(midi) -> bytes:
    buf = io.BytesIO()
    midi.save(file=buf)
    return buf.getvalue()


def build_fl_bundle(
    notes: Sequence[Note],
    *,
    program_names: Mapping[int, str] | None = None,
    beat_grid: BeatGrid | None = None,
    options: ExportOptions | None = None,
    stem_audio: Mapping[str, tuple[torch.Tensor, int]] | None = None,
    include_stems: bool = True,
    normalize_drum_lanes_to_c5: bool = True,
    song_name: str = "transcription",
) -> bytes:
    """Pack combined MIDI, per-instrument MIDIs, optional C5 drum lanes, and
    bar-1-aligned WAV stems into a single uncompressed FL Studio ZIP archive."""
    options = options or ExportOptions()
    cleaned, updated_names = cleanup_notes(
        notes,
        min_note_duration_ms=options.min_note_duration_ms,
        min_velocity=options.min_velocity,
        deduplicate_ms=options.deduplicate_ms,
        excluded_instruments=options.excluded_instruments,
        instrument_remap=options.instrument_remap,
        use_velocities=options.use_velocities,
        velocity_scale=options.velocity_scale,
        program_names=program_names,
    )

    grid = resolve_grid(beat_grid, options, [n.onset for n in cleaned])
    delay = grid.onset_delay or 0.0
    shifted = shifted_notes(cleaned, -delay)
    earliest = min([-delay] + [n.onset for n in shifted])
    timeline = timeline_for(grid, earliest)

    beat_notes = [
        dataclasses.replace(
            n,
            onset=float(timeline.beats_at(n.onset)),
            offset=float(timeline.beats_at(n.offset)),
        )
        for n in shifted
    ]

    if options.quantize and grid.beat_subdivision is not None:
        beat_notes = quantized_notes(
            beat_notes, 1 / grid.beat_subdivision, options.strength
        )
    if options.fill_gaps > 0:
        beat_notes = gaps_filled(beat_notes, options.fill_gaps)
    if options.trim_leading_bars and beat_notes:
        empty = leading_bars(beat_notes, grid)
        timeline = timeline.shifted(-empty)
        beat_notes = shifted_notes(beat_notes, -empty)

    offset_s = float(-timeline.seconds_at(0.0))
    tempo_changes = timeline.tempo_changes(TICKS_PER_BEAT)

    def _render(
        track_notes: list[Note],
        names_map: dict[int, str],
        split_drums_flag: bool = False,
    ) -> bytes:
        midi = note_event2midi(
            note2note_event(track_notes),
            output_file=None,
            velocity=100,
            ticks_per_beat=TICKS_PER_BEAT,
            program_names=names_map,
            beats_per_bar=grid.beats_per_bar,
            offset_s=offset_s,
            to_tick=lambda beats: round(beats * TICKS_PER_BEAT),
            tempo_changes=tempo_changes,
            drum_length=DRUM_HIT_BEATS,
            split_drums=split_drums_flag,
        )
        return _midi_to_bytes(midi)

    written_files: list[str] = []
    archive_buf = io.BytesIO()
    with zipfile.ZipFile(archive_buf, "w", compression=zipfile.ZIP_STORED) as zf:
        # 1. Combined multi-track MIDI (always preserves GM drum pitches on ch 9).
        zf.writestr(
            "00_full_mix.mid",
            _render(beat_notes, updated_names, split_drums_flag=options.split_drums),
        )
        written_files.append("00_full_mix.mid")

        # 2. Individual per-instrument MIDIs (melodic/harmonic).
        by_program: dict[int, list[Note]] = defaultdict(list)
        drum_notes: list[Note] = []
        for n in beat_notes:
            if n.is_drum or n.program == DRUM_PROGRAM:
                drum_notes.append(n)
            else:
                by_program[n.program].append(n)

        for idx, (prog, prog_notes) in enumerate(by_program.items(), start=1):
            label = updated_names.get(prog, f"program {prog}")
            arcname = f"midi/{idx:02d}_{_slugify(label)}.mid"
            zf.writestr(arcname, _render(prog_notes, {prog: label}))
            written_files.append(arcname)

        # 3. Individual drum MIDIs (split by lane or combined).
        if drum_notes:
            if options.split_drums:
                by_lane: dict[str, list[Note]] = defaultdict(list)
                for n in drum_notes:
                    by_lane[drum_lane_for_pitch(n.pitch)].append(n)
                for lane in DRUM_LANE_ORDER:
                    lane_notes = by_lane.get(lane)
                    if not lane_notes:
                        continue
                    arcname = f"midi/drums_{lane}.mid"
                    label = DRUM_LANE_LABELS[lane]
                    if normalize_drum_lanes_to_c5 and lane in SINGLE_HIT_DRUM_LANES:
                        # Map single-hit lanes to MIDI 60 (FL Studio C5) on channel 0
                        # so dragging straight onto an FL Sampler plays at root pitch.
                        c5_notes = [
                            Note(
                                is_drum=False,
                                program=0,
                                onset=n.onset,
                                offset=n.onset + DRUM_HIT_BEATS,
                                pitch=C5_PITCH,
                                velocity=n.velocity,
                            )
                            for n in lane_notes
                        ]
                        zf.writestr(arcname, _render(c5_notes, {0: label}))
                    else:
                        zf.writestr(arcname, _render(lane_notes, {DRUM_PROGRAM: label}))
                    written_files.append(arcname)
            else:
                arcname = "midi/drums.mid"
                zf.writestr(arcname, _render(drum_notes, {DRUM_PROGRAM: "drums"}))
                written_files.append(arcname)

        # 4. Bar-1-aligned WAV stems (when present and requested).
        if include_stems and stem_audio:
            for stem_name, (wav, sr) in stem_audio.items():
                aligned = align_wav_to_offset(wav, sr, offset_s)
                arcname = f"stems/{_slugify(stem_name)}.wav"
                zf.writestr(arcname, write_pcm16_wav_bytes(aligned, sr))
                written_files.append(arcname)

        # 5. FL Studio import guide & metadata summary.
        meter_str = f"{grid.beats_per_bar}/4" if grid.beats_per_bar else "unknown"
        tempo_mode = "Fixed tempo" if grid.steady else "Variable tempo map"
        lines = [
            f"FL Studio MIDI & Stem Bundle — {song_name}",
            "=" * 54,
            f"Tempo:          {grid.bpm:.2f} BPM ({tempo_mode})",
            f"Time signature: {meter_str}",
            f"Bar-1 offset:   {offset_s:+.4f} s (all MIDIs and WAV stems share tick 0)",
            "",
            "Quick import into FL Studio:",
            f"  1. Set your FL Studio project tempo to {grid.bpm:.2f} BPM.",
            "  2. Drag stems/*.wav (if included) onto Playlist Bar 1 Beat 1.",
            "  3. Drag 00_full_mix.mid for all tracks at once, OR drag individual",
            "     midi/*.mid files directly onto specific Channel Rack VSTs/Samplers.",
        ]
        if options.split_drums and normalize_drum_lanes_to_c5:
            lines.append(
                "  4. Single-hit drum lanes (kick, snare, hihat) are normalized to C5"
                " (MIDI 60) for FL Studio Sampler channels; 00_full_mix.mid keeps GM pitches."
            )
        if not grid.steady:
            lines.extend(
                [
                    "",
                    "NOTE (Variable Tempo Map):",
                    "  This track's tempo varies over time. Bar 1 Beat 1 is aligned,",
                    "  but WAV stems do not automatically follow MIDI tempo changes in FL Studio.",
                    "  For sample-accurate full-song WAV+MIDI lock, either re-export with",
                    "  'One fixed tempo' or enable tempo-sync/stretching on the stems in FL Studio.",
                ]
            )
        lines.extend(["", "Included files:"] + [f"  - {name}" for name in written_files])
        zf.writestr("README_FL_STUDIO.txt", "\n".join(lines) + "\n")

    return archive_buf.getvalue()
