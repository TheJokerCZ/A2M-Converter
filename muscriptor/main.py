"""CLI for muscriptor: audio → MIDI transcription."""

import dataclasses
import json
import sys
import wave
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import typer

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.model_manager import ModelManager
from muscriptor.sessions import SessionStore
from muscriptor.tokenizer.mt3 import (
    MT3_FULL_PLUS_GROUP_NAMES,
    resolve_instrument_names,
)
from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    Note,
    trim_overlapping_notes,
    validate_notes,
)
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.audio import _read_non_wav_file, _read_wav_file
from muscriptor.utils.beats import BeatDetectionError, TempoDetection
from muscriptor.utils.cleanup import default_name_to_program
from muscriptor.utils.download import ModelDownloadError
from muscriptor.utils.fl_export import build_fl_bundle
from muscriptor.utils.midi import ExportOptions
from muscriptor.utils.sheets import (
    MuseScoreError,
    MuseScoreNotFoundError,
    find_musescore,
    prepare_output_dir,
    write_sheets,
)
from muscriptor.utils.stems import (
    SeparationMode,
    StemSeparationUnavailableError,
    is_stem_silent,
    plan_stem_conditioning,
    separate_stems,
)
from muscriptor.utils.velocity import estimate_note_velocities

app = typer.Typer(add_completion=False, help="muscriptor — audio-to-MIDI transcription")


def _load_model(
    model_path: str | None, device: str | None, dtype: str | None = None
) -> TranscriptionModel:
    """load_model with CLI-friendly failure: known download problems (missing
    HuggingFace authentication, …) print a plain message instead of a traceback."""
    try:
        return TranscriptionModel.load_model(
            weights_path=model_path, device=device, dtype=dtype
        )
    except ModelDownloadError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


class OutputFormat(str, Enum):
    midi = "midi"
    json = "json"
    jsonl = "jsonl"
    sheets = "sheets"
    fl_bundle = "fl-bundle"


def _transcribe(model, kwargs: dict, detect_tempo: str, export: ExportOptions):
    """transcribe_and_postprocess, with the CLI's --detect-tempo spelling and errors."""
    try:
        mode: TempoDetection = {
            "true": True,
            "false": False,
            "best-effort": "best-effort",
        }[detect_tempo]
        return model.transcribe_and_postprocess(
            **kwargs, detect_tempo=mode, export=export
        )
    except BeatDetectionError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Pass --detect-tempo best-effort or false to continue.", err=True)
        raise typer.Exit(1)


def _event_to_dict(ev: NoteStartEvent | NoteEndEvent) -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


@app.command()
def transcribe(
    audio_file: Annotated[
        Path, typer.Argument(help="Input audio file (wav, mp3, flac, …)")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output file path. Use '-' to write to stdout (all progress / "
                "timing info is sent to stderr in that case). "
                "Default: <audio_file>.<ext> where ext matches --format. "
                "With --format sheets this is a directory instead: it must be "
                "empty or not exist yet, and is created if missing."
            ),
        ),
    ] = None,
    format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Output format: midi (default), json (single array of events), "
                "jsonl (one event per line, streamed as transcription "
                "progresses), or sheets (a directory of engraved PDFs plus "
                "MusicXML and MIDI; requires MuseScore to be installed)"
            ),
            case_sensitive=False,
        ),
    ] = OutputFormat.midi,
    notes: Annotated[
        bool, typer.Option("--notes", help="Print decoded events to stdout")
    ] = False,
    sampling: Annotated[
        bool,
        typer.Option(
            "--sampling", help="Use temperature sampling instead of greedy decoding"
        ),
    ] = False,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature", "-t", help="Sampling temperature (only with --sampling)"
        ),
    ] = 1.0,
    cfg_coef: Annotated[
        float, typer.Option("--cfg-coef", help="Classifier-free guidance coefficient")
    ] = 1.0,  # todo: make it dynamic
    model_path: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model size ('small', 'medium', 'large'; default: medium), "
                "a local safetensors path, or an hf:// / http(s):// URL"
            ),
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps', …"
        ),
    ] = "auto",
    dtype: Annotated[
        str | None,
        typer.Option(
            "--dtype",
            help=(
                "Transformer dtype: 'float32', 'float16' or 'bfloat16'. "
                "Default: float16 on MPS, float32 elsewhere."
            ),
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            "-b",
            help=(
                "Chunks generated per forward pass (default: 1; with "
                "--no-prelude-forcing: 4 on GPU, 1 on CPU). Values > 1 lower "
                "quality at chunk boundaries and require --no-prelude-forcing."
            ),
        ),
    ] = None,
    strict_eos: Annotated[
        bool,
        typer.Option(
            "--strict-eos",
            help="Raise an error if a chunk fails to emit EOS within the generation budget (default: downgrade to a warning)",
        ),
    ] = False,
    beam_size: Annotated[
        int,
        typer.Option(
            "--beam-size",
            help="Beam search width (1 = greedy/sampling, ≥2 enables beam search)",
        ),
    ] = 1,
    prelude_forcing: Annotated[
        bool,
        typer.Option(
            "--prelude-forcing/--no-prelude-forcing",
            help=(
                "Teacher-force each chunk's tie prologue from the previous "
                "chunk's still-sounding notes, so chunks can't restart with "
                "the wrong instruments. Needs chunks generated in order, so "
                "it requires --batch-size 1 (the default)."
            ),
        ),
    ] = True,
    auralize: Annotated[
        Path | None,
        typer.Option(
            "--auralize",
            help=(
                "Write a stereo auralization (L=original audio, R=MIDI synthesis) to "
                "this path. Requires fluidsynth on PATH. Extension determines format: "
                ".wav (default) or .mp3. Only valid with --format midi."
            ),
        ),
    ] = None,
    soundfont: Annotated[
        Path | None,
        typer.Option(
            "--soundfont",
            help=(
                "Path to a .sf2 SoundFont for auralization. Defaults to "
                "MuseScore_General.sf2, downloaded once and cached locally."
            ),
        ),
    ] = None,
    instruments: Annotated[
        str | None,
        typer.Option(
            "--instruments",
            help=(
                "Comma-separated list of expected instrument group names. "
                "When given, every instrument not in the list is forbidden "
                "from being decoded at all. Case-insensitive; unambiguous "
                "abbreviations are accepted (e.g. 'timp,cello,dist'). Run "
                "'muscriptor list-instruments' to see all available names."
            ),
        ),
    ] = None,
    # typer can't take the bool | Literal["best-effort"] union the API uses, so
    # the CLI spells all three states as strings and converts below.
    detect_tempo: Annotated[
        Literal["true", "false", "best-effort"],
        typer.Option(
            help=(
                "Detect the beats and time signature from the audio and write "
                "them into the MIDI. 'true' fails if no beats are found, "
                "'best-effort' warns and uses a placeholder 120 BPM with no "
                "time signature, 'false' skips detection altogether."
            ),
        ),
    ] = "best-effort",
    bpm: Annotated[
        float | None,
        typer.Option(
            "--bpm",
            min=20,
            max=400,
            help="Write this tempo instead of the detected one.",
            rich_help_panel="MIDI timing",
        ),
    ] = None,
    beats_per_bar: Annotated[
        int | None,
        typer.Option(
            "--beats-per-bar",
            min=1,
            max=32,
            help="Write this many beats per bar (n/4) instead of the detected meter.",
            rich_help_panel="MIDI timing",
        ),
    ] = None,
    first_downbeat: Annotated[
        float | None,
        typer.Option(
            "--first-downbeat",
            help=(
                "Time in seconds of a bar line in the audio, instead of the "
                "detected first downbeat."
            ),
            rich_help_panel="MIDI timing",
        ),
    ] = None,
    tempo_map: Annotated[
        bool | None,
        typer.Option(
            "--tempo-map/--fixed-tempo",
            help=(
                "Follow the tracked beats with a tempo map, or write one constant "
                "tempo. Default: a tempo map only when the tempo isn't steady."
            ),
            show_default=False,
            rich_help_panel="MIDI timing",
        ),
    ] = None,
    beat_scale: Annotated[
        float,
        typer.Option(
            "--beat-scale",
            help=(
                "Beats per detected beat: 2 if the detected tempo is half what "
                "it should be, 0.5 if double. A power of two."
            ),
            rich_help_panel="MIDI timing",
        ),
    ] = 1.0,
    quantize: Annotated[
        bool,
        typer.Option(
            "--quantize/--no-quantize",
            help=(
                "Snap note starts and ends onto the beat grid. Always on for "
                "--format sheets."
            ),
            rich_help_panel="MIDI timing",
        ),
    ] = False,
    grid: Annotated[
        int | None,
        typer.Option(
            "--grid",
            min=1,
            max=32,
            help=(
                "Steps per beat to quantize to: 4 = sixteenths, 3 = eighth "
                "triplets, … Default: the one the notes sit on best."
            ),
            rich_help_panel="MIDI timing",
        ),
    ] = None,
    quantize_strength: Annotated[
        float,
        typer.Option(
            "--quantize-strength",
            min=0.0,
            max=1.0,
            help="How far to move notes onto the grid: 1 all the way, 0.5 halfway.",
            rich_help_panel="MIDI timing",
        ),
    ] = 1.0,
    fill_gaps: Annotated[
        float,
        typer.Option(
            "--fill-gaps",
            min=0.0,
            help=(
                "Hold notes on until the next note of their instrument when the "
                "silence in between is at most this many beats, to avoid "
                "spurious rests in notation software. 0 disables."
            ),
            rich_help_panel="MIDI timing",
        ),
    ] = 0.0,
    trim_leading_bars: Annotated[
        bool,
        typer.Option(
            "--trim-leading-bars",
            help=(
                "Drop the empty bars before the first note. The MIDI then no "
                "longer lines up with the start of the audio."
            ),
            rich_help_panel="MIDI timing",
        ),
    ] = False,
    separation: Annotated[
        SeparationMode,
        typer.Option(
            "--separation",
            help="Stem separation mode before transcribing: 'none', 'stems_4' (htdemucs_ft), or 'stems_6' (htdemucs_6s).",
            rich_help_panel="Studio & FL Export",
        ),
    ] = "none",
    stems: Annotated[
        str | None,
        typer.Option(
            "--stems",
            help="Comma-separated subset of stems to transcribe (e.g. 'drums,bass,other').",
            rich_help_panel="Studio & FL Export",
        ),
    ] = None,
    estimate_velocity: Annotated[
        bool | None,
        typer.Option(
            "--estimate-velocity/--no-estimate-velocity",
            help="Estimate per-note MIDI velocities from audio energy. Default: on for --separation != none or --format fl-bundle.",
            rich_help_panel="Studio & FL Export",
        ),
    ] = None,
    min_note_ms: Annotated[
        float,
        typer.Option(
            "--min-note-ms",
            min=0.0,
            help="Drop non-drum notes shorter than this many milliseconds (0 disables).",
            rich_help_panel="Studio & FL Export",
        ),
    ] = 0.0,
    min_velocity: Annotated[
        int,
        typer.Option(
            "--min-velocity",
            min=1,
            max=127,
            help="Drop notes with MIDI velocity below this threshold.",
            rich_help_panel="Studio & FL Export",
        ),
    ] = 1,
    deduplicate_ms: Annotated[
        float,
        typer.Option(
            "--deduplicate-ms",
            min=0.0,
            help="Merge same-(instrument, pitch) hits within this many milliseconds.",
            rich_help_panel="Studio & FL Export",
        ),
    ] = 0.0,
    split_drums: Annotated[
        bool,
        typer.Option(
            "--split-drums",
            help="Split GM drums into Kick, Snare, Hi-Hat, Toms, Cymbals, and Perc tracks.",
            rich_help_panel="Studio & FL Export",
        ),
    ] = False,
    c5_drums: Annotated[
        bool,
        typer.Option(
            "--c5-drums/--no-c5-drums",
            help="Normalize single-hit drum lane MIDIs (kick/snare/hihat) to C5 (MIDI 60) in FL bundles.",
            rich_help_panel="Studio & FL Export",
        ),
    ] = True,
    include_stems: Annotated[
        bool,
        typer.Option(
            "--include-stems/--no-include-stems",
            help="Include bar-1-aligned WAV stems inside --format fl-bundle archives when separation ran.",
            rich_help_panel="Studio & FL Export",
        ),
    ] = True,
) -> None:
    """Transcribe an audio file to MIDI."""
    instrument_names: list[str] | None = None
    if instruments is not None:
        tokens = [n for n in instruments.split(",") if n.strip()]
        try:
            instrument_names = resolve_instrument_names(tokens)
        except ValueError as e:
            typer.echo(
                f"Error: {e}. "
                "Run 'muscriptor list-instruments' to see available names.",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Instruments: {', '.join(instrument_names)}", err=True)

    if not audio_file.exists():
        typer.echo(f"Error: file not found: {audio_file}", err=True)
        raise typer.Exit(1)

    if prelude_forcing and batch_size is not None and batch_size != 1:
        typer.echo(
            f"Error: --batch-size {batch_size} requires --no-prelude-forcing: "
            "batching disables prelude forcing, which lowers transcription "
            "quality at chunk boundaries.",
            err=True,
        )
        raise typer.Exit(1)

    is_stdout = output is not None and str(output) == "-"

    if output is None:
        if format == OutputFormat.sheets:
            output = audio_file.parent / f"{audio_file.stem}_sheets"
        elif format == OutputFormat.fl_bundle:
            output = audio_file.parent / f"{audio_file.stem}_fl_studio.zip"
        else:
            suffix = {
                OutputFormat.midi: ".mid",
                OutputFormat.json: ".json",
                OutputFormat.jsonl: ".jsonl",
            }[format]
            output = audio_file.with_suffix(suffix)

    if format == OutputFormat.fl_bundle and is_stdout:
        typer.echo(
            "Error: --format fl-bundle writes a ZIP archive, so it cannot write to stdout. Pass a path with -o.",
            err=True,
        )
        raise typer.Exit(1)

    if format == OutputFormat.sheets:
        if is_stdout:
            typer.echo(
                "Error: --format sheets writes a directory, so it cannot write "
                "to stdout. Pass a directory path with -o.",
                err=True,
            )
            raise typer.Exit(1)
        # Both checked before the model loads: transcribing a song takes long
        # enough that discovering a missing MuseScore afterwards would sting.
        try:
            prepare_output_dir(output)
            find_musescore()
        except (ValueError, MuseScoreNotFoundError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    # Checked before the model loads, like the sheets output above.
    try:
        export = ExportOptions(
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            first_downbeat=first_downbeat,
            tempo_map=tempo_map,
            beat_scale=beat_scale,
            # Sheets are always quantized to get the "idealized" timing, otherwise
            # we might get very weird 1/64th rests etc.
            quantize=quantize or format == OutputFormat.sheets,
            subdivision=grid,
            strength=quantize_strength,
            fill_gaps=fill_gaps,
            trim_leading_bars=trim_leading_bars,
            min_note_duration_ms=min_note_ms,
            min_velocity=min_velocity,
            deduplicate_ms=deduplicate_ms,
            split_drums=split_drums,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    _device = None if device == "auto" else device
    do_velocity = (
        estimate_velocity
        if estimate_velocity is not None
        else (separation != "none" or format == OutputFormat.fl_bundle)
    )
    active_stem_list = (
        [s.strip().lower() for s in stems.split(",") if s.strip()]
        if stems
        else None
    )

    if auralize is not None and format != OutputFormat.midi:
        typer.echo("Error: --auralize requires --format midi", err=True)
        raise typer.Exit(1)

    # Studio pipeline path (stem separation, audio velocity, or FL bundle export).
    if separation != "none" or format == OutputFormat.fl_bundle or do_velocity:
        try:
            wav, sr = _read_wav_file(str(audio_file))
        except (wave.Error, EOFError):
            wav, sr = _read_non_wav_file(str(audio_file))

        stem_audio: dict[str, tuple] = {}
        if separation != "none":
            typer.echo(f"Separating stems ({separation})…", err=True)
            try:
                stem_audio = separate_stems(wav, sr, mode=separation, device=_device)
            except StemSeparationUnavailableError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1)

        typer.echo("Loading model…", err=True)
        model = _load_model(model_path, _device, dtype)
        typer.echo(f"Transcribing {audio_file} …", err=True)

        events: list[NoteStartEvent | NoteEndEvent] = []
        note_stems_by_index: dict[int, str] = {}
        next_idx = 0

        if separation != "none" and stem_audio:
            plans = plan_stem_conditioning(
                separation,
                active_stems=active_stem_list,
                user_instruments=instrument_names,
            )
            for plan in plans:
                if plan.stem not in stem_audio:
                    continue
                s_wav, s_sr = stem_audio[plan.stem]
                if is_stem_silent(s_wav, sr=s_sr):
                    typer.echo(f"Skipping silent {plan.stem} stem", err=True)
                    continue
                typer.echo(f"  Transcribing {plan.stem}…", err=True)
                local_map: dict[int, NoteStartEvent] = {}
                t_kw = dict(
                    use_sampling=sampling,
                    temperature=temperature,
                    cfg_coef=cfg_coef,
                    instruments=plan.instruments,
                    batch_size=batch_size,
                    no_eos_is_ok=not strict_eos,
                    beam_size=beam_size,
                    prelude_forcing=prelude_forcing,
                )
                if plan.allowed_instruments is not None:
                    t_kw["allowed_instruments"] = plan.allowed_instruments
                for ev in model.transcribe((s_wav, s_sr), **t_kw):
                    if isinstance(ev, ProgressEvent):
                        continue
                    if isinstance(ev, NoteStartEvent):
                        remapped = NoteStartEvent(
                            pitch=ev.pitch,
                            start_time=ev.start_time,
                            index=next_idx,
                            instrument=ev.instrument,
                        )
                        local_map[ev.index] = remapped
                        note_stems_by_index[next_idx] = plan.stem
                        next_idx += 1
                        events.append(remapped)
                    else:
                        st = local_map.get(ev.start_event_index)
                        if st is not None:
                            events.append(
                                NoteEndEvent(end_time=ev.end_time, start_event=st)
                            )
        else:
            for ev in model.transcribe(
                (wav, sr),
                use_sampling=sampling,
                temperature=temperature,
                cfg_coef=cfg_coef,
                instruments=instrument_names,
                batch_size=batch_size,
                no_eos_is_ok=not strict_eos,
                beam_size=beam_size,
                prelude_forcing=prelude_forcing,
            ):
                if isinstance(ev, ProgressEvent):
                    continue
                if isinstance(ev, NoteStartEvent):
                    note_stems_by_index[ev.index] = "mix"
                events.append(ev)

        mode: TempoDetection = {
            "true": True,
            "false": False,
            "best-effort": "best-effort",
        }[detect_tempo]
        try:
            beat_grid = model.detect_beat_grid_for((wav, sr), mode)
        except BeatDetectionError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        if beat_grid is not None:
            beat_grid = beat_grid.with_onset_delay(
                [ev.start_time for ev in events if isinstance(ev, NoteStartEvent)]
            )

        name_to_prog = default_name_to_program()
        open_starts: dict[int, NoteStartEvent] = {}
        paired_notes: list[Note] = []
        paired_indices: list[int] = []
        paired_stems: list[str] = []
        program_names: dict[int, str] = {}
        for ev in events:
            if isinstance(ev, NoteStartEvent):
                open_starts[ev.index] = ev
            elif isinstance(ev, NoteEndEvent):
                st = open_starts.pop(ev.start_event_index, None)
                if st is None:
                    continue
                is_drum = st.instrument == "drums"
                prog = DRUM_PROGRAM if is_drum else name_to_prog.get(st.instrument, 0)
                program_names[prog] = st.instrument.replace("_", " ")
                paired_notes.append(
                    Note(
                        is_drum=is_drum,
                        program=prog,
                        onset=st.start_time,
                        offset=ev.end_time,
                        pitch=st.pitch,
                    )
                )
                paired_indices.append(st.index)
                paired_stems.append(note_stems_by_index.get(st.index, "mix"))

        vel_by_index: dict[int, int] | None = None
        if do_velocity:
            vel_onset_delay = float(beat_grid.onset_delay or 0.0) if beat_grid else 0.0
            vel_notes = [
                dataclasses.replace(
                    n,
                    onset=max(0.0, n.onset - vel_onset_delay),
                    offset=max(max(0.0, n.onset - vel_onset_delay) + 0.01, n.offset - vel_onset_delay),
                )
                for n in paired_notes
            ]
            audio_lookup = dict(stem_audio) if stem_audio else {}
            audio_lookup.setdefault("mix", (wav, sr))
            vels = estimate_note_velocities(
                vel_notes, audio_lookup, note_stems=paired_stems
            )
            vel_by_index = {idx: v for idx, v in zip(paired_indices, vels)}
            paired_notes = [
                dataclasses.replace(n, velocity=v)
                for n, v in zip(paired_notes, vels)
            ]

        if format == OutputFormat.fl_bundle:
            paired_notes = trim_overlapping_notes(
                validate_notes(paired_notes, fix=True), sort=True
            )
            selected_stems = (
                {
                    k: v
                    for k, v in stem_audio.items()
                    if active_stem_list is None or k in active_stem_list
                }
                if (include_stems and stem_audio)
                else None
            )
            zip_bytes = build_fl_bundle(
                paired_notes,
                program_names=program_names,
                beat_grid=beat_grid,
                options=export,
                stem_audio=selected_stems,
                include_stems=include_stems,
                normalize_drum_lanes_to_c5=c5_drums,
                song_name=audio_file.stem,
            )
            output.write_bytes(zip_bytes)
            typer.echo(f"Saved FL Studio bundle to {output}", err=True)
            return

        if format == OutputFormat.sheets:
            midi_bytes = model.events_to_midi_bytes(
                iter(events),
                beat_grid=beat_grid,
                export=export,
                note_velocities=vel_by_index,
            )
            typer.echo(f"Engraving sheet music with MuseScore → {output} …", err=True)
            try:
                written = write_sheets(
                    midi_bytes,
                    output,
                    quantized=beat_grid is not None
                    and beat_grid.beat_subdivision is not None,
                )
            except MuseScoreError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1)
            for path in written:
                typer.echo(f"  {path.name}", err=True)
            typer.echo(f"Saved {len(written)} files to {output}", err=True)
            return

        if format == OutputFormat.midi:
            midi_bytes = model.events_to_midi_bytes(
                iter(events),
                beat_grid=beat_grid,
                export=export,
                note_velocities=vel_by_index,
            )
            if is_stdout:
                sys.stdout.buffer.write(midi_bytes)
                sys.stdout.buffer.flush()
            else:
                output.write_bytes(midi_bytes)
                typer.echo(f"Saved MIDI to {output}", err=True)
            if notes:
                typer.echo(
                    "Re-run with --format json to inspect the event stream.", err=True
                )
            if auralize is not None and not is_stdout:
                from muscriptor.utils.auralization import auralize as do_auralize

                typer.echo(f"Auralizing → {auralize} …", err=True)
                do_auralize(
                    midi_path=output,
                    original_audio_path=audio_file,
                    output_path=auralize,
                    soundfont_path=soundfont,
                )
                typer.echo(f"Saved auralization to {auralize}", err=True)
            return

        if format == OutputFormat.jsonl:
            if is_stdout:
                sink = sys.stdout
                close_after = False
            else:
                sink = output.open("w")
                close_after = True
            try:
                for e in events:
                    sink.write(json.dumps(_event_to_dict(e)) + "\n")
                    sink.flush()
                    if notes:
                        typer.echo(str(e), err=True)
            finally:
                if close_after:
                    sink.close()
            if not is_stdout:
                typer.echo(f"Saved JSONL to {output}", err=True)
            return

        # OutputFormat.json
        payload = json.dumps([_event_to_dict(e) for e in events], indent=2)
        if is_stdout:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()
        else:
            output.write_text(payload)
            typer.echo(f"Saved JSON to {output}", err=True)
        if notes:
            for e in events:
                typer.echo(str(e), err=True)
        return

    # All chatty progress/timing info goes to stderr — stdout is reserved for
    # the actual output when `-o -` is used.
    typer.echo("Loading model…", err=True)
    model = _load_model(model_path, _device, dtype)

    typer.echo(f"Transcribing {audio_file} …", err=True)

    kwargs = dict(
        audio=audio_file,
        use_sampling=sampling,
        temperature=temperature,
        cfg_coef=cfg_coef,
        instruments=instrument_names,
        batch_size=batch_size,
        no_eos_is_ok=not strict_eos,
        beam_size=beam_size,
        prelude_forcing=prelude_forcing,
    )

    if format == OutputFormat.sheets:
        midi_bytes, beat_grid = _transcribe(model, kwargs, detect_tempo, export)
        typer.echo(f"Engraving sheet music with MuseScore → {output} …", err=True)
        try:
            written = write_sheets(
                midi_bytes,
                output,
                quantized=beat_grid is not None
                and beat_grid.beat_subdivision is not None,
            )
        except MuseScoreError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        for path in written:
            typer.echo(f"  {path.name}", err=True)
        typer.echo(f"Saved {len(written)} files to {output}", err=True)
    elif format == OutputFormat.midi:
        midi_bytes, _ = _transcribe(model, kwargs, detect_tempo, export)
        if is_stdout:
            sys.stdout.buffer.write(midi_bytes)
            sys.stdout.buffer.flush()
        else:
            output.write_bytes(midi_bytes)
            typer.echo(f"Saved MIDI to {output}", err=True)
        if notes:
            typer.echo(
                "Re-run with --format json to inspect the event stream.", err=True
            )
        if auralize is not None and not is_stdout:
            from muscriptor.utils.auralization import auralize as do_auralize

            typer.echo(f"Auralizing → {auralize} …", err=True)
            do_auralize(
                midi_path=output,
                original_audio_path=audio_file,
                output_path=auralize,
                soundfont_path=soundfont,
            )
            typer.echo(f"Saved auralization to {auralize}", err=True)
    elif format == OutputFormat.jsonl:
        # Stream one JSON object per line, flushing after each event so the
        # file (or stdout pipe) can be consumed live.
        if is_stdout:
            sink = sys.stdout
            close_after = False
        else:
            sink = output.open("w")
            close_after = True
        try:
            for e in model.transcribe(**kwargs):
                if isinstance(e, ProgressEvent):
                    continue
                sink.write(json.dumps(_event_to_dict(e)) + "\n")
                sink.flush()
                if notes:
                    typer.echo(str(e), err=True)
        finally:
            if close_after:
                sink.close()
        if not is_stdout:
            typer.echo(f"Saved JSONL to {output}", err=True)
    else:  # json
        events = [
            e for e in model.transcribe(**kwargs) if not isinstance(e, ProgressEvent)
        ]
        payload = json.dumps([_event_to_dict(e) for e in events], indent=2)
        if is_stdout:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()
        else:
            output.write_text(payload)
            typer.echo(f"Saved JSON to {output}", err=True)
        if notes:
            for e in events:
                typer.echo(str(e), err=True)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on")] = 8222,
    model_path: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model size ('small', 'medium', 'large'; default: medium), "
                "a local safetensors path, or an hf:// / http(s):// URL"
            ),
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps', …"
        ),
    ] = "auto",
    dtype: Annotated[
        str | None,
        typer.Option(
            "--dtype",
            help=(
                "Transformer dtype: 'float32', 'float16' or 'bfloat16'. "
                "Default: float16 on MPS, bf16/fp16 for large on <=12GB CUDA, float32 elsewhere."
            ),
        ),
    ] = None,
    preload: Annotated[
        bool,
        typer.Option(
            "--preload/--no-preload",
            help="Eagerly download and load the default model before starting HTTP (default: lazy load on first request).",
        ),
    ] = False,
):
    """Run the HTTP transcription server (POST /transcribe → SSE event stream)."""
    import logging

    import uvicorn

    from muscriptor.server import create_app

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    _device = None if device == "auto" else device
    manager = ModelManager(default_model=model_path, device=_device, dtype=dtype)
    store = SessionStore()
    if preload:
        typer.echo("Loading model…")
        try:
            manager.get_model()
        except ModelDownloadError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
    web_dir = Path(__file__).resolve().parent / "web_dist"
    fastapi_app = create_app(
        model=None,
        web_dir=web_dir if web_dir.is_dir() else None,
        model_manager=manager,
        session_store=store,
    )
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def list_instruments():
    """List the instrument group names accepted by --instruments."""
    for name in MT3_FULL_PLUS_GROUP_NAMES:
        typer.echo(name)


def main():
    app()


if __name__ == "__main__":
    main()
