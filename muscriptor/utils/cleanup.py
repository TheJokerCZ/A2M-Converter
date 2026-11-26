"""Non-destructive MIDI note cleanup filters and GM drum-lane grouping."""

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence

from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES, get_group_program_map
from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note, sort_notes

# Standard FL Studio Sampler root note (C5 in FL Studio == MIDI note 60).
C5_PITCH = 60

# Deterministic GM drum pitch → FL Studio lane grouping (channel 9).
DRUM_LANES: dict[str, frozenset[int]] = {
    "kick": frozenset({35, 36}),
    "snare": frozenset({37, 38, 39, 40}),
    "hihat": frozenset({42, 44, 46}),
    "toms": frozenset({41, 43, 45, 47, 48, 50}),
    "cymbals": frozenset({49, 51, 52, 53, 55, 57, 59}),
}

DRUM_LANE_ORDER: tuple[str, ...] = (
    "kick",
    "snare",
    "hihat",
    "toms",
    "cymbals",
    "perc",
)

DRUM_LANE_LABELS: dict[str, str] = {
    "kick": "Drums - Kick",
    "snare": "Drums - Snare",
    "hihat": "Drums - Hi-Hat",
    "toms": "Drums - Toms",
    "cymbals": "Drums - Cymbals",
    "perc": "Drums - Perc",
}

# Single-hit lanes whose individual FL Studio MIDIs normalize cleanly to C5.
SINGLE_HIT_DRUM_LANES: frozenset[str] = frozenset({"kick", "snare", "hihat"})

# Default midpoint around which `velocity_scale` expands or compresses contrast.
_VELOCITY_MIDPOINT = 77.0


def drum_lane_for_pitch(pitch: int) -> str:
    """Map any GM percussion pitch (0..127) to one of the 6 FL Studio drum lanes."""
    for lane, pitches in DRUM_LANES.items():
        if pitch in pitches:
            return lane
    return "perc"


def _canonical_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def default_name_to_program() -> dict[str, int]:
    """Canonical instrument-group name → representative MIDI program number."""
    group_map = get_group_program_map("MT3_FULL_PLUS", "OMIT", include_drums=False)
    mapping: dict[str, int] = {"drums": DRUM_PROGRAM}
    for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items():
        if name == "drums":
            continue
        if gid in group_map and group_map[gid]:
            mapping[name] = group_map[gid][0]
    return mapping


def cleanup_notes(
    notes: Sequence[Note],
    *,
    min_note_duration_ms: float = 0.0,
    min_velocity: int = 1,
    deduplicate_ms: float = 0.0,
    excluded_instruments: Sequence[str] = (),
    instrument_remap: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    use_velocities: bool = True,
    velocity_scale: float = 1.0,
    program_names: Mapping[int, str] | None = None,
    name_to_program: Mapping[str, int] | None = None,
) -> tuple[list[Note], dict[int, str]]:
    """Apply non-destructive export filters and instrument remapping to `notes`.

    Returns shallow-copied filtered `Note` objects and an updated `program_names`
    dict without mutating the caller's inputs.
    """
    updated_names: dict[int, str] = dict(program_names or {})
    canonical_to_prog: dict[str, int] = default_name_to_program()
    if name_to_program:
        for k, v in name_to_program.items():
            canonical_to_prog[_canonical_name(k)] = v
    for prog, label in updated_names.items():
        canonical_to_prog[_canonical_name(label)] = prog

    prog_to_canonical: dict[int, str] = {v: k for k, v in canonical_to_prog.items()}
    for prog, label in updated_names.items():
        prog_to_canonical[prog] = _canonical_name(label)
    prog_to_canonical[DRUM_PROGRAM] = "drums"

    excluded_set = {_canonical_name(name) for name in excluded_instruments}
    remap_pairs = (
        instrument_remap.items()
        if isinstance(instrument_remap, Mapping)
        else (instrument_remap or ())
    )
    remap: dict[str, str] = {
        _canonical_name(src): _canonical_name(dst)
        for src, dst in remap_pairs
        if _canonical_name(src) != _canonical_name(dst)
    }

    min_dur_s = max(0.0, float(min_note_duration_ms)) / 1000.0
    vel_floor = max(1, int(min_velocity))

    kept: list[Note] = []
    for orig in notes:
        prog = DRUM_PROGRAM if orig.is_drum else orig.program
        inst_name = prog_to_canonical.get(prog, f"program_{prog}")

        # Apply remapping before exclusion so remapped targets can also be excluded.
        if inst_name in remap:
            target_name = remap[inst_name]
            target_prog = canonical_to_prog.get(target_name)
            if target_prog is not None:
                prog = target_prog
                inst_name = target_name
                if target_prog not in updated_names:
                    updated_names[target_prog] = target_name.replace("_", " ")

        if inst_name in excluded_set:
            continue

        vel = max(1, min(127, int(orig.velocity)))
        if vel < vel_floor:
            continue

        is_drum = prog == DRUM_PROGRAM or orig.is_drum
        if not is_drum and min_dur_s > 0.0 and (orig.offset - orig.onset) < min_dur_s:
            continue

        kept.append(
            dataclasses.replace(
                orig,
                is_drum=is_drum,
                program=DRUM_PROGRAM if is_drum else prog,
                velocity=vel,
            )
        )

    # Deduplicate near-simultaneous same-(instrument, pitch) hits.
    dedup_s = max(0.0, float(deduplicate_ms)) / 1000.0
    if dedup_s > 0.0 and kept:
        grouped: dict[tuple[bool, int, int], list[Note]] = defaultdict(list)
        for note in kept:
            grouped[(note.is_drum, note.program, note.pitch)].append(note)
        deduped: list[Note] = []
        for group in grouped.values():
            group.sort(key=lambda n: n.onset)
            merged: list[Note] = []
            for note in group:
                if merged and (note.onset - merged[-1].onset) <= dedup_s:
                    prev = merged[-1]
                    merged[-1] = dataclasses.replace(
                        prev,
                        offset=max(prev.offset, note.offset),
                        velocity=max(prev.velocity, note.velocity),
                    )
                else:
                    merged.append(note)
            deduped.extend(merged)
        kept = deduped

    # Velocity flatten or contrast scaling.
    if not use_velocities:
        kept = [dataclasses.replace(n, velocity=100) for n in kept]
    elif velocity_scale != 1.0:
        kept = [
            dataclasses.replace(
                n,
                velocity=max(
                    1,
                    min(
                        127,
                        round(
                            _VELOCITY_MIDPOINT
                            + (n.velocity - _VELOCITY_MIDPOINT) * velocity_scale
                        ),
                    ),
                ),
            )
            for n in kept
        ]

    # Prune excluded / emptied entries from updated_names.
    active_progs = {DRUM_PROGRAM if n.is_drum else n.program for n in kept}
    updated_names = {
        prog: label
        for prog, label in updated_names.items()
        if prog in active_progs and _canonical_name(label) not in excluded_set
    }

    sort_notes(kept)
    return kept, updated_names
