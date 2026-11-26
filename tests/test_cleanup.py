"""Tests for muscriptor/utils/cleanup.py and extended MIDI ExportOptions."""

from mido import MidiFile

from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note
from muscriptor.utils.beats import BeatGrid
from muscriptor.utils.cleanup import (
    C5_PITCH,
    DRUM_LANE_LABELS,
    DRUM_LANE_ORDER,
    SINGLE_HIT_DRUM_LANES,
    cleanup_notes,
    drum_lane_for_pitch,
)
from muscriptor.utils.midi import ExportOptions, notes_to_midi


def test_drum_lane_for_pitch_covers_gm_map():
    assert drum_lane_for_pitch(35) == "kick"
    assert drum_lane_for_pitch(36) == "kick"
    assert drum_lane_for_pitch(38) == "snare"
    assert drum_lane_for_pitch(39) == "snare"
    assert drum_lane_for_pitch(42) == "hihat"
    assert drum_lane_for_pitch(46) == "hihat"
    assert drum_lane_for_pitch(45) == "toms"
    assert drum_lane_for_pitch(50) == "toms"
    assert drum_lane_for_pitch(49) == "cymbals"
    assert drum_lane_for_pitch(51) == "cymbals"
    assert drum_lane_for_pitch(56) == "perc"  # cowbell
    assert drum_lane_for_pitch(80) == "perc"
    assert C5_PITCH == 60
    assert SINGLE_HIT_DRUM_LANES == frozenset({"kick", "snare", "hihat"})
    assert DRUM_LANE_ORDER == ("kick", "snare", "hihat", "toms", "cymbals", "perc")


def test_cleanup_filters_duration_velocity_and_deduplicates():
    notes = [
        # Too short melodic note (20 ms) -> dropped
        Note(False, 0, 0.0, 0.02, 60, velocity=90),
        # Drum hit with 10 ms duration -> kept despite min_note_duration_ms
        Note(True, DRUM_PROGRAM, 0.0, 0.01, 36, velocity=95),
        # Quiet bleed note (velocity 15 < 20) -> dropped
        Note(False, 0, 0.5, 1.0, 62, velocity=15),
        # Near-simultaneous duplicate (15 ms apart) -> merged keeping higher velocity & max offset
        Note(False, 0, 1.0, 1.4, 64, velocity=70),
        Note(False, 0, 1.015, 1.6, 64, velocity=110),
    ]
    cleaned, _ = cleanup_notes(
        notes,
        min_note_duration_ms=35.0,
        min_velocity=20,
        deduplicate_ms=25.0,
    )
    assert len(cleaned) == 2
    drum = [n for n in cleaned if n.is_drum][0]
    assert drum.pitch == 36
    melodic = [n for n in cleaned if not n.is_drum][0]
    assert melodic.pitch == 64
    assert melodic.velocity == 110
    assert melodic.offset == 1.6
    # Caller's original notes were not mutated
    assert len(notes) == 5
    assert notes[3].velocity == 70


def test_cleanup_excludes_and_remaps_instruments():
    notes = [
        Note(False, 52, 0.0, 0.5, 65, velocity=90),  # voice
        Note(False, 2, 0.0, 0.5, 60, velocity=85),  # electric_piano
        Note(False, 0, 0.5, 1.0, 64, velocity=95),  # acoustic_piano
    ]
    program_names = {52: "voice", 2: "electric piano", 0: "acoustic piano"}
    name_to_program = {"voice": 52, "electric_piano": 2, "acoustic_piano": 0}
    cleaned, updated_names = cleanup_notes(
        notes,
        excluded_instruments=("voice",),
        instrument_remap={"electric_piano": "acoustic_piano"},
        program_names=program_names,
        name_to_program=name_to_program,
    )
    assert len(cleaned) == 2
    assert all(n.program == 0 for n in cleaned)
    assert updated_names[0] == "acoustic piano"
    assert 52 not in updated_names


def test_cleanup_velocity_flatten_and_scale():
    notes = [
        Note(False, 0, 0.0, 0.5, 60, velocity=40),
        Note(False, 0, 0.5, 1.0, 64, velocity=110),
    ]
    flat, _ = cleanup_notes(notes, use_velocities=False)
    assert [n.velocity for n in flat] == [100, 100]

    scaled, _ = cleanup_notes(notes, use_velocities=True, velocity_scale=1.5)
    assert scaled[0].velocity < 40
    assert scaled[1].velocity > 110
    assert all(1 <= n.velocity <= 127 for n in scaled)


def test_notes_to_midi_split_drums_creates_named_drum_tracks():
    notes = [
        Note(True, DRUM_PROGRAM, 0.0, 0.01, 36, velocity=120),
        Note(True, DRUM_PROGRAM, 0.5, 0.51, 38, velocity=105),
        Note(True, DRUM_PROGRAM, 0.25, 0.26, 42, velocity=75),
    ]
    midi = notes_to_midi(
        notes,
        beat_grid=BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.0, onset_delay=0.0),
        options=ExportOptions(split_drums=True),
    )
    track_names = [
        msg.name
        for track in midi.tracks[1:]
        for msg in track
        if msg.type == "track_name"
    ]
    assert track_names == [
        DRUM_LANE_LABELS["kick"],
        DRUM_LANE_LABELS["snare"],
        DRUM_LANE_LABELS["hihat"],
    ]
    for track, expected_pitch, expected_vel in zip(
        midi.tracks[1:], [36, 38, 42], [120, 105, 75]
    ):
        ons = [m for m in track if m.type == "note_on" and m.velocity > 0]
        assert len(ons) == 1
        assert ons[0].channel == 9
        assert ons[0].note == expected_pitch
        assert ons[0].velocity == expected_vel


def test_notes_to_midi_all_notes_filtered_out_returns_valid_empty_midi():
    notes = [
        Note(False, 0, 0.0, 0.5, 60, velocity=50),
        Note(True, DRUM_PROGRAM, 0.5, 0.51, 36, velocity=60),
    ]
    midi = notes_to_midi(
        notes,
        beat_grid=BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.0, onset_delay=0.0),
        options=ExportOptions(min_velocity=127, trim_leading_bars=True),
    )
    assert isinstance(midi, MidiFile)
    tempos = [m for track in midi.tracks for m in track if m.type == "set_tempo"]
    assert len(tempos) >= 1
    note_ons = [m for track in midi.tracks for m in track if m.type == "note_on"]
    assert note_ons == []
