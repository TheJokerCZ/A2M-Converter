"""FastAPI server exposing transcription as an SSE event stream.

POST /transcribe with an audio file (multipart/form-data field `file`; WAV,
or any format soundfile/libsndfile can read — mp3, flac, ogg, m4a, …) returns
`text/event-stream`. Each event's data is a JSON dict tagged by `type`:
`start` / `end` note events (same shape as `muscriptor.main._event_to_dict`),
`progress` chunk anchors (`{completed, total}`), and a final
`transcription_complete` event carrying the base64-encoded .mid file (`data`)
plus the detected `beat_grid`
(`{bpm, beats_per_bar, first_downbeat, onset_delay, beat_subdivision, steady,
beats}`, or null if no beats were found). `steady` says whether the tracked
`beats` fit the constant `bpm`; when not, the MIDI follows them with a tempo
map. `onset_delay` is how late the streamed note times are against those
beats: the MIDI has it taken out already, an SSE consumer has to subtract it.

POST /export takes JSON instead: the streamed notes, that `beat_grid`, and
`options` (`muscriptor.utils.midi.ExportOptions`: tempo/meter/downbeat
corrections, quantization, gap filling), and returns the .mid written with
them. It runs no model, so the UI can re-export as the user changes options.

POST /transcribe/midi takes the same upload but blocks until transcription
completes and returns the raw `audio/midi` bytes directly (no SSE, no
base64), with a `Content-Disposition: attachment` header. Audio longer than
15 minutes is rejected with 413.

POST /sheets takes a MIDI upload instead of audio (a quantized one from
/export, with `quantized=true`) and returns every file
`muscriptor.utils.sheets.write_sheets` engraves from it — MusicXML, the full
score, one PDF per instrument — as a single uncompressed zip. It needs
MuseScore 4+ on the server, and answers 503 when there is none.
"""

import asyncio
import base64
import dataclasses
import io
import json
import os
import tempfile
import threading
import time
import wave
import zipfile
from pathlib import Path
from typing import Annotated, Callable, Literal

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent, StageEvent
from muscriptor.model_manager import ModelManager
from muscriptor.sessions import SessionStore
from muscriptor.soundfonts import SF3_URL
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note, trim_overlapping_notes, validate_notes
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.audio import _read_non_wav_file, _read_wav_file
from muscriptor.utils.beats import BeatDetectionError, BeatGrid, TempoDetection, detect_grid
from muscriptor.utils.cleanup import default_name_to_program
from muscriptor.utils.download import download_if_necessary
from muscriptor.utils.fl_export import build_fl_bundle
from muscriptor.utils.midi import ExportOptions, notes_to_midi
from muscriptor.utils.sheets import (
    MuseScoreError,
    MuseScoreNotFoundError,
    write_sheets,
)
from muscriptor.utils.stems import (
    SeparationMode,
    is_stem_silent,
    plan_stem_conditioning,
    separate_stems,
)
from muscriptor.utils.velocity import estimate_note_velocities


def _make_release_once(lock: threading.Lock):
    """Return a callable that releases `lock` at most once.

    Safe to call from multiple cleanup paths (generator finally + response
    background task), possibly from different threads, without risking a
    double-release RuntimeError.
    """
    guard = threading.Lock()
    released = False

    def release():
        nonlocal released
        with guard:
            if released:
                return
            released = True
        lock.release()

    return release


_MAX_TRANSCRIBE_MIDI_DURATION_S = 15 * 60

SHEETS_ZIP_NAME = "sheets.zip"


def engrave_to_zip(midi_bytes: bytes, quantized: bool = False) -> bytes:
    """Engrave `midi_bytes` and pack everything written into one zip.

    Runs `write_sheets` into a scratch directory that is thrown away once the
    archive is built, so the server keeps nothing on disk between requests.
    Member names are the bare filenames — the directory layout documented under
    "Sheet music" in the README, flattened by one level.

    Stored, not deflated: the client unpacks this archive in the browser to
    offer the files one at a time, and all but the MusicXML are PDFs, which
    carry compression of their own.
    """
    with tempfile.TemporaryDirectory(prefix="muscriptor-sheets-") as tmp:
        written = write_sheets(midi_bytes, Path(tmp), quantized=quantized)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in written:
                archive.write(path, arcname=path.name)
        return buf.getvalue()


class ExportNote(BaseModel):
    """A note as /transcribe streamed it (or edited in audio-aligned time)."""

    instrument: str
    pitch: int = Field(ge=0, le=127)
    start: float
    end: float
    velocity: int = Field(default=100, ge=1, le=127)


class ExportGrid(BaseModel):
    """The `beat_grid` /transcribe sent, passed back as it came."""

    bpm: float = Field(gt=0)
    beats_per_bar: int | None = None
    first_downbeat: float = 0.0
    onset_delay: float = 0.0
    beat_subdivision: int | None = None
    steady: bool = True
    beats: list[float] | None = None


class ExportSettings(BaseModel):
    """`ExportOptions`, with the bounds a client could get wrong."""

    bpm: float | None = Field(default=None, ge=20, le=400)
    beats_per_bar: int | None = Field(default=None, ge=1, le=32)
    first_downbeat: float | None = None
    tempo_map: bool | None = None
    beat_scale: Literal[0.25, 0.5, 1, 2, 4] = 1
    quantize: bool = False
    subdivision: int | None = Field(default=None, ge=1, le=32)
    strength: float = Field(default=1.0, ge=0, le=1)
    fill_gaps: float = Field(default=0.0, ge=0, le=16)
    trim_leading_bars: bool = False
    use_velocities: bool = True
    velocity_scale: float = Field(default=1.0, ge=0.1, le=3.0)
    min_note_duration_ms: float = Field(default=0.0, ge=0.0, le=2000.0)
    min_velocity: int = Field(default=1, ge=1, le=127)
    deduplicate_ms: float = Field(default=0.0, ge=0.0, le=500.0)
    split_drums: bool = False
    excluded_instruments: list[str] = Field(default_factory=list)
    instrument_remap: dict[str, str] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    notes: list[ExportNote]
    beat_grid: ExportGrid | None = None
    options: ExportSettings = ExportSettings()


class FlBundleExportRequest(ExportRequest):
    session_id: str | None = None
    include_stems: bool = True
    normalize_drum_lanes_to_c5: bool = True
    song_name: str = "transcription"


def _export_notes_to_model_notes(
    notes: list[ExportNote],
) -> tuple[list[Note], dict[int, str]]:
    name_to_prog = default_name_to_program()
    out: list[Note] = []
    program_names: dict[int, str] = {}
    for n in notes:
        is_drum = n.instrument == "drums"
        prog = DRUM_PROGRAM if is_drum else name_to_prog[n.instrument]
        program_names[prog] = n.instrument.replace("_", " ")
        out.append(
            Note(
                is_drum=is_drum,
                program=prog,
                onset=n.start,
                offset=n.end,
                pitch=n.pitch,
                velocity=n.velocity,
            )
        )
    out = validate_notes(out, fix=True)
    out = trim_overlapping_notes(out, sort=True)
    return out, program_names


def _grid_from_request(bg: ExportGrid | None) -> BeatGrid | None:
    if bg is None:
        return None
    fields = bg.model_dump()
    beats = fields.pop("beats")
    return BeatGrid(**fields, beats=None if beats is None else np.asarray(beats))


def _stage_payload(ev: StageEvent) -> str:
    data: dict[str, object] = {
        "type": "stage",
        "stage": ev.stage,
        "message": ev.message,
    }
    if ev.stem is not None:
        data["stem"] = ev.stem
    if ev.stem_index is not None:
        data["stem_index"] = ev.stem_index
    if ev.stem_total is not None:
        data["stem_total"] = ev.stem_total
    return f"data: {json.dumps(data)}\n\n"


def _serialize_beat_grid(grid: BeatGrid | None) -> dict | None:
    if grid is None:
        return None
    return {
        "bpm": grid.bpm,
        "beats_per_bar": grid.beats_per_bar,
        "first_downbeat": grid.first_downbeat,
        "onset_delay": grid.onset_delay,
        "beat_subdivision": grid.beat_subdivision,
        "steady": grid.steady,
        "beats": None
        if grid.beats is None
        else [round(float(b), 3) for b in grid.beats],
    }


def export_events(notes: list[ExportNote]) -> list[NoteStartEvent | NoteEndEvent]:
    """`notes` as the event stream `events_to_midi_bytes` reassembles them from."""
    events: list[NoteStartEvent | NoteEndEvent] = []
    for index, note in enumerate(notes):
        start = NoteStartEvent(
            pitch=note.pitch,
            start_time=note.start,
            index=index,
            instrument=note.instrument,
        )
        events += [start, NoteEndEvent(end_time=note.end, start_event=start)]
    return events


def event_to_dict(ev: NoteStartEvent | NoteEndEvent) -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


def create_app(
    model: TranscriptionModel | None = None,
    web_dir: str | Path | None = None,
    *,
    model_manager: ModelManager | None = None,
    session_store: SessionStore | None = None,
    separator_fn: Callable[..., dict] | None = None,
) -> FastAPI:
    app = FastAPI(title="muscriptor")
    sessions = session_store if session_store is not None else SessionStore()
    run_separator = separator_fn if separator_fn is not None else separate_stems

    def _resolve_model(model_size: str | None = None) -> TranscriptionModel:
        if model_manager is not None:
            return model_manager.get_model(model_size)
        if model is None:
            raise RuntimeError("No transcription model or ModelManager configured.")
        return model

    def _release_gpu() -> None:
        if model_manager is not None:
            model_manager.release_gpu()

    transcribe_lock = threading.Lock()
    # State of the run currently holding the lock (or the last one to have held
    # it), guarded by `cancel_guard`: its cancel event and the id of the client
    # that started it. A new /transcribe from the SAME client (a resubmit in
    # the same browser tab) sets the cancel event so the in-flight run stops at
    # its next event boundary instead of transcribing to completion for a client
    # that has moved on. A request from a DIFFERENT client never preempts — it
    # waits for the lock like any other contender, so two independent browser
    # windows don't kill each other's transcription. This scoping must be done
    # here rather than by watching the socket: browser aborts don't always reach
    # us (e.g. port forwards / proxies keep the upstream connection open after
    # the browser aborts), so a same-tab resubmit can't be detected as a
    # disconnect.
    cancel_guard = threading.Lock()
    current_cancel: threading.Event | None = None
    current_client: str | None = None
    # Bound on how long a same-client resubmit waits for the run it just
    # cancelled to unwind and release the lock. That run stops within one
    # chunk boundary, so this only needs to cover that latency — unlike
    # genuine cross-client contention, which now fails instantly instead of
    # sitting on the connection (see below).
    preempt_wait_s = 5.0

    async def acquire_transcribe_lock(
        client_id: str | None, cancellable: bool
    ) -> tuple[threading.Event | None, Callable[[], None]]:
        """Acquire the single-transcription lock, preempting only a run started
        by this same `client_id` (a resubmit): that run is signalled to stop
        and this call waits up to `preempt_wait_s` for it to release the lock.
        A different client — or an anonymous API caller with no id — never
        preempts and is never preempted; it gets an immediate 503 instead of
        waiting, so a caller retrying against another machine (e.g. behind a
        load balancer like Traefik) doesn't have to hold the connection open.

        Returns `(cancel, release)`: `cancel` is the new run's cancel event
        (`None` when `cancellable` is False, e.g. the blocking /transcribe/midi
        render, which can't be stopped mid-flight); `release` frees the lock at
        most once, from whichever cleanup path runs first.
        """
        nonlocal current_cancel, current_client
        deadline: float | None = None
        while True:
            with cancel_guard:
                # Re-check each iteration so that even a same-client run which
                # became "current" while we were already waiting (a resubmit
                # that beat us to the lock) gets cancelled too — the newest
                # request from a given client always wins.
                preempting = (
                    current_cancel is not None
                    and client_id is not None
                    and current_client == client_id
                )
                if preempting:
                    current_cancel.set()
            if not preempting:
                acquired = await asyncio.to_thread(transcribe_lock.acquire, False)
                if not acquired:
                    raise HTTPException(
                        status_code=503,
                        detail="server busy: another transcription is in progress",
                    )
                break
            if deadline is None:
                deadline = time.monotonic() + preempt_wait_s
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPException(
                    status_code=503,
                    detail="server busy: another transcription is in progress",
                )
            acquired = await asyncio.to_thread(
                transcribe_lock.acquire, True, min(0.1, remaining)
            )
            if acquired:
                break
        cancel = threading.Event() if cancellable else None
        with cancel_guard:
            current_cancel = cancel
            current_client = client_id
        return cancel, _make_release_once(transcribe_lock)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/capabilities")
    async def capabilities():
        if model_manager is not None:
            return model_manager.capabilities()
        return ModelManager(device="cpu").capabilities()

    @app.get("/instruments")
    async def list_instruments():
        return {"instruments": list(MT3_FULL_PLUS_GROUP_NAMES.keys())}

    @app.get("/sessions/{session_id}/stems/{stem_name}.wav")
    async def get_session_stem(session_id: str, stem_name: str) -> FileResponse:
        if not SessionStore.is_valid_session_id(session_id) or not SessionStore.is_valid_stem_name(stem_name):
            raise HTTPException(status_code=400, detail="invalid session id or stem name")
        path = sessions.get_stem_path(session_id, stem_name)
        if path is None:
            raise HTTPException(status_code=404, detail="stem not found or session expired")
        return FileResponse(path, media_type="audio/wav", filename=f"{stem_name}.wav")

    @app.get("/soundfonts/MuseScore_General.sf3")
    async def soundfont() -> FileResponse:
        """Compressed soundfont for the web UI's in-browser synthesizer.

        Fetched from SF3_URL on first request (in a worker thread, so the
        event loop keeps serving) and cached locally.
        """
        path = await asyncio.to_thread(download_if_necessary, SF3_URL)
        return FileResponse(path, media_type="application/octet-stream")

    @app.post("/transcribe")
    async def transcribe(
        file: Annotated[UploadFile, File()],
        instruments: Annotated[list[str], Form(default_factory=list)],
        stems: Annotated[list[str], Form(default_factory=list)],
        # "true" fails loudly on tempo detection errors, "false" doesn't even try
        detect_tempo: Annotated[TempoDetection, Form()] = "best-effort",
        studio: Annotated[bool, Form()] = False,
        separation: Annotated[SeparationMode, Form()] = "none",
        model_size: Annotated[str | None, Form()] = None,
        beam_size: Annotated[int, Form(ge=1, le=4)] = 1,
        x_client_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        data = await file.read()
        try:
            wav, sr = _read_wav_file(io.BytesIO(data))
        except (wave.Error, EOFError):
            try:
                wav, sr = _read_non_wav_file(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"could not decode audio file '{file.filename}': {e}",
                ) from e

        unknown = [n for n in instruments if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown instrument name(s): {', '.join(unknown)}",
            )

        cancel, release_lock = await acquire_transcribe_lock(
            x_client_id, cancellable=True
        )

        # Legacy wire path: preserves exact SSE events and keys for existing callers/tests.
        if not studio and separation == "none" and model_size is None and beam_size == 1:
            def legacy_gen():
                try:
                    active_model = _resolve_model(None)
                    events: list[NoteStartEvent | NoteEndEvent] = []
                    for ev in active_model.transcribe(
                        (wav, sr),
                        instruments=instruments or None,
                        batch_size=1,
                        no_eos_is_ok=True,
                    ):
                        if cancel.is_set():
                            return
                        if isinstance(ev, ProgressEvent):
                            payload = json.dumps(
                                {
                                    "type": "progress",
                                    "completed": ev.completed,
                                    "total": ev.total,
                                }
                            )
                            yield f"data: {payload}\n\n"
                            continue
                        events.append(ev)
                        payload = json.dumps(event_to_dict(ev))
                        yield f"data: {payload}\n\n"
                    if cancel.is_set():
                        return
                    grid = active_model.detect_beat_grid_for((wav, sr), detect_tempo)
                    if grid:
                        grid = grid.with_onset_delay(
                            [
                                ev.start_time
                                for ev in events
                                if isinstance(ev, NoteStartEvent)
                            ]
                        )
                    midi_bytes = active_model.events_to_midi_bytes(
                        iter(events), beat_grid=grid
                    )
                    midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
                    payload = json.dumps(
                        {
                            "type": "transcription_complete",
                            "data": midi_b64,
                            "beat_grid": _serialize_beat_grid(grid),
                        }
                    )
                    yield f"data: {payload}\n\n"
                finally:
                    release_lock()

            return StreamingResponse(
                legacy_gen(),
                media_type="text/event-stream",
                background=BackgroundTask(release_lock),
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Studio SSE path: supports sequential GPU stem separation, model switching,
        # stage updates, audio-derived velocities, and structured SSE error reporting.
        def studio_gen():
            try:
                stem_audio: dict[str, tuple] = {}
                session_id: str | None = None
                stem_names: list[str] = []

                if separation != "none":
                    _release_gpu()
                    yield _stage_payload(
                        StageEvent(
                            stage="separating",
                            message=f"Separating stems ({'6 stems' if separation == 'stems_6' else '4 stems'})…",
                        )
                    )
                    if cancel.is_set():
                        return
                    stem_audio = run_separator(wav, sr, mode=separation)
                    if cancel.is_set():
                        return
                    if stem_audio:
                        session_id = sessions.create_session(stem_audio)
                        stem_names = list(stem_audio.keys())

                yield _stage_payload(
                    StageEvent(
                        stage="loading_model",
                        message=f"Loading MuScriptor ({model_size or (model_manager.default_model if model_manager else 'default')})…",
                    )
                )
                active_model = _resolve_model(model_size)
                if cancel.is_set():
                    return

                events: list[NoteStartEvent | NoteEndEvent] = []
                note_stems_by_index: dict[int, str] = {}
                next_global_idx = 0

                if separation != "none" and stem_audio:
                    plans = plan_stem_conditioning(
                        separation,
                        active_stems=stems or None,
                        user_instruments=instruments or None,
                    )
                    active_plans = []
                    for p in plans:
                        if p.stem not in stem_audio:
                            continue
                        stem_wav, stem_sr = stem_audio[p.stem]
                        if is_stem_silent(stem_wav, sr=stem_sr):
                            yield _stage_payload(
                                StageEvent(
                                    stage="transcribing",
                                    message=f"Skipping silent {p.stem} stem",
                                    stem=p.stem,
                                )
                            )
                            continue
                        active_plans.append(p)

                    stem_total = max(1, len(active_plans))
                    for stem_idx, plan in enumerate(active_plans, start=1):
                        if cancel.is_set():
                            return
                        yield _stage_payload(
                            StageEvent(
                                stage="transcribing",
                                message=f"Transcribing {plan.stem} ({stem_idx}/{stem_total})…",
                                stem=plan.stem,
                                stem_index=stem_idx,
                                stem_total=stem_total,
                            )
                        )
                        stem_wav, stem_sr = stem_audio[plan.stem]
                        local_to_global: dict[int, NoteStartEvent] = {}
                        t_kwargs: dict[str, object] = {
                            "instruments": plan.instruments,
                            "batch_size": 1,
                            "no_eos_is_ok": True,
                            "beam_size": beam_size,
                        }
                        if plan.allowed_instruments is not None:
                            t_kwargs["allowed_instruments"] = plan.allowed_instruments

                        for ev in active_model.transcribe(
                            (stem_wav, stem_sr), **t_kwargs
                        ):
                            if cancel.is_set():
                                return
                            if isinstance(ev, ProgressEvent):
                                total_all = max(1, ev.total * stem_total)
                                completed_all = (stem_idx - 1) * ev.total + ev.completed
                                payload = json.dumps(
                                    {
                                        "type": "progress",
                                        "completed": completed_all,
                                        "total": total_all,
                                    }
                                )
                                yield f"data: {payload}\n\n"
                                continue
                            if isinstance(ev, NoteStartEvent):
                                remapped = NoteStartEvent(
                                    pitch=ev.pitch,
                                    start_time=ev.start_time,
                                    index=next_global_idx,
                                    instrument=ev.instrument,
                                )
                                local_to_global[ev.index] = remapped
                                note_stems_by_index[next_global_idx] = plan.stem
                                next_global_idx += 1
                                events.append(remapped)
                                yield f"data: {json.dumps(event_to_dict(remapped))}\n\n"
                            else:
                                start_ev = local_to_global.get(ev.start_event_index)
                                if start_ev is None:
                                    continue
                                end_ev = NoteEndEvent(
                                    end_time=ev.end_time, start_event=start_ev
                                )
                                events.append(end_ev)
                                yield f"data: {json.dumps(event_to_dict(end_ev))}\n\n"
                else:
                    yield _stage_payload(
                        StageEvent(
                            stage="transcribing",
                            message="Transcribing full mix…",
                            stem="mix",
                            stem_index=1,
                            stem_total=1,
                        )
                    )
                    for ev in active_model.transcribe(
                        (wav, sr),
                        instruments=instruments or None,
                        batch_size=1,
                        no_eos_is_ok=True,
                        beam_size=beam_size,
                    ):
                        if cancel.is_set():
                            return
                        if isinstance(ev, ProgressEvent):
                            payload = json.dumps(
                                {
                                    "type": "progress",
                                    "completed": ev.completed,
                                    "total": ev.total,
                                }
                            )
                            yield f"data: {payload}\n\n"
                            continue
                        if isinstance(ev, NoteStartEvent):
                            note_stems_by_index[ev.index] = "mix"
                        events.append(ev)
                        yield f"data: {json.dumps(event_to_dict(ev))}\n\n"

                if cancel.is_set():
                    return

                yield _stage_payload(
                    StageEvent(stage="beats", message="Detecting tempo and bar lines…")
                )
                grid = active_model.detect_beat_grid_for((wav, sr), detect_tempo)
                if grid:
                    grid = grid.with_onset_delay(
                        [
                            ev.start_time
                            for ev in events
                            if isinstance(ev, NoteStartEvent)
                        ]
                    )

                yield _stage_payload(
                    StageEvent(
                        stage="velocity",
                        message="Estimating per-note velocities from audio…",
                    )
                )
                # Pair start/end events into Notes for velocity estimation.
                name_to_prog = default_name_to_program()
                open_starts: dict[int, NoteStartEvent] = {}
                paired_notes: list[Note] = []
                paired_indices: list[int] = []
                paired_stems: list[str] = []
                for ev in events:
                    if isinstance(ev, NoteStartEvent):
                        open_starts[ev.index] = ev
                    elif isinstance(ev, NoteEndEvent):
                        st = open_starts.pop(ev.start_event_index, None)
                        if st is None:
                            continue
                        is_drum = st.instrument == "drums"
                        prog = (
                            DRUM_PROGRAM
                            if is_drum
                            else name_to_prog.get(st.instrument, 0)
                        )
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

                audio_lookup = dict(stem_audio) if stem_audio else {}
                audio_lookup.setdefault("mix", (wav, sr))
                estimated_vels = estimate_note_velocities(
                    paired_notes, audio_lookup, note_stems=paired_stems
                )
                vel_by_index = {
                    idx: vel for idx, vel in zip(paired_indices, estimated_vels)
                }

                midi_bytes = active_model.events_to_midi_bytes(
                    iter(events), beat_grid=grid, note_velocities=vel_by_index
                )
                midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
                payload = json.dumps(
                    {
                        "type": "transcription_complete",
                        "data": midi_b64,
                        "beat_grid": _serialize_beat_grid(grid),
                        "session_id": session_id,
                        "stems": stem_names,
                        "velocities": {str(k): v for k, v in vel_by_index.items()},
                    }
                )
                yield f"data: {payload}\n\n"
            except Exception as e:
                err_payload = json.dumps({"type": "error", "message": str(e)})
                yield f"data: {err_payload}\n\n"
            finally:
                release_lock()

        return StreamingResponse(
            studio_gen(),
            media_type="text/event-stream",
            background=BackgroundTask(release_lock),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/transcribe/midi")
    async def transcribe_midi(
        file: Annotated[UploadFile, File()],
        instruments: Annotated[list[str], Form(default_factory=list)],
        # "true" fails loudly on tempo detection errors, "false" doesn't even try
        detect_tempo: Annotated[TempoDetection, Form()] = "best-effort",
        x_client_id: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Transcribe an audio file and return the .mid file directly.

        Unlike /transcribe, this blocks until transcription finishes and
        returns the raw MIDI bytes (no SSE, no base64) with a
        Content-Disposition header, so a plain HTTP client can save the
        response straight to disk.
        """
        data = await file.read()
        try:
            wav, sr = _read_wav_file(io.BytesIO(data))
        except (wave.Error, EOFError):
            try:
                wav, sr = _read_non_wav_file(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"could not decode audio file '{file.filename}': {e}",
                ) from e

        duration_s = wav.shape[-1] / sr
        if duration_s > _MAX_TRANSCRIBE_MIDI_DURATION_S:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"audio file is {duration_s / 60:.1f} minutes long; "
                    f"the limit is {_MAX_TRANSCRIBE_MIDI_DURATION_S // 60:.0f} minutes"
                ),
            )

        unknown = [n for n in instruments if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown instrument name(s): {', '.join(unknown)}",
            )

        # Same mutual exclusion as /transcribe, via the shared helper. This run
        # is not cancellable (it doesn't stream, so there's nothing to stop
        # mid-flight); cancellable=False records that, so nothing tries to
        # preempt a blocking MIDI render. A non-browser API caller sends no
        # client id and so neither preempts nor is preempted — it just waits for
        # the lock like any other contender.
        _cancel, release_lock = await acquire_transcribe_lock(
            x_client_id, cancellable=False
        )
        try:
            active_model = await asyncio.to_thread(_resolve_model, None)
            midi_bytes, _ = await asyncio.to_thread(
                active_model.transcribe_and_postprocess,
                (wav, sr),
                instruments=instruments or None,
                detect_tempo=detect_tempo,
            )
        except BeatDetectionError as e:
            # Only reachable with detect_tempo=true, where the caller wants an error
            # if tempo detection fails.
            raise HTTPException(status_code=422, detail=str(e)) from e
        finally:
            release_lock()

        return Response(
            content=midi_bytes,
            media_type="audio/midi",
            headers={"Content-Disposition": 'attachment; filename="result.mid"'},
        )

    @app.post("/export")
    async def export(request: ExportRequest) -> Response:
        """Write the notes /transcribe streamed as MIDI, with export options.

        No model and no lock: this only rebuilds the file, so it is cheap
        enough to call on every download.
        """
        unknown = sorted(
            {n.instrument for n in request.notes} - MT3_FULL_PLUS_GROUP_NAMES.keys()
        )
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown instrument name(s): {', '.join(unknown)}",
            )
        grid = _grid_from_request(request.beat_grid)
        export_opts = ExportOptions(**request.options.model_dump())

        if model is not None:
            vel_map = {idx: n.velocity for idx, n in enumerate(request.notes)}
            has_custom_vel = any(v != 100 for v in vel_map.values())
            if has_custom_vel:
                midi_bytes = await asyncio.to_thread(
                    model.events_to_midi_bytes,
                    iter(export_events(request.notes)),
                    beat_grid=grid,
                    export=export_opts,
                    note_velocities=vel_map,
                )
            else:
                midi_bytes = await asyncio.to_thread(
                    model.events_to_midi_bytes,
                    iter(export_events(request.notes)),
                    beat_grid=grid,
                    export=export_opts,
                )
        else:
            def _build_direct() -> bytes:
                notes_list, prog_names = _export_notes_to_model_notes(request.notes)
                midi = notes_to_midi(
                    notes_list,
                    program_names=prog_names,
                    beat_grid=grid,
                    options=export_opts,
                )
                buf = io.BytesIO()
                midi.save(file=buf)
                return buf.getvalue()

            midi_bytes = await asyncio.to_thread(_build_direct)
        return Response(content=midi_bytes, media_type="audio/midi")

    @app.post("/export/fl-bundle")
    async def export_fl_bundle(request: FlBundleExportRequest) -> Response:
        """Build an FL Studio ZIP bundle (combined MIDI, per-instrument MIDIs,
        optional C5 drum lanes, and bar-1-aligned WAV stems)."""
        unknown = sorted(
            {n.instrument for n in request.notes} - MT3_FULL_PLUS_GROUP_NAMES.keys()
        )
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown instrument name(s): {', '.join(unknown)}",
            )
        grid = _grid_from_request(request.beat_grid)
        export_opts = ExportOptions(**request.options.model_dump())

        def _build() -> bytes:
            notes_list, prog_names = _export_notes_to_model_notes(request.notes)
            stem_audio = (
                sessions.load_stems(request.session_id)
                if (request.include_stems and request.session_id)
                else None
            )
            return build_fl_bundle(
                notes_list,
                program_names=prog_names,
                beat_grid=grid,
                options=export_opts,
                stem_audio=stem_audio,
                include_stems=request.include_stems,
                normalize_drum_lanes_to_c5=request.normalize_drum_lanes_to_c5,
                song_name=request.song_name,
            )

        zip_bytes = await asyncio.to_thread(_build)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="fl_studio_bundle.zip"'
            },
        )

    @app.post("/auralize")
    async def auralize(
        midi: Annotated[UploadFile, File()],
        audio: Annotated[UploadFile | None, File()] = None,
        mode: Annotated[str, Form()] = "mix",
    ):
        """Render a transcription as WAV.

        mode="mix": stereo, original audio (L) + FluidSynth synthesis (R);
        requires the `audio` upload. mode="synth": mono, just the synthesis.
        """
        from muscriptor.utils.auralization import auralize as do_auralize
        from muscriptor.utils.auralization import synthesize

        if mode not in ("mix", "synth"):
            raise HTTPException(status_code=400, detail=f"unknown mode: {mode!r}")
        if mode == "mix" and audio is None:
            raise HTTPException(
                status_code=400, detail="mode='mix' requires an audio file"
            )

        midi_data = await midi.read()
        tmp_paths: list[str] = []

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
            tmp_midi.write(midi_data)
            midi_tmp = tmp_midi.name
            tmp_paths.append(midi_tmp)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_tmp = tmp_out.name
            tmp_paths.append(out_tmp)

        try:
            if mode == "synth":
                synthesize(midi_path=midi_tmp, output_path=out_tmp)
            else:
                audio_data = await audio.read()
                suffix = Path(audio.filename or "audio.wav").suffix.lower() or ".wav"
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as tmp_audio:
                    tmp_audio.write(audio_data)
                    tmp_paths.append(tmp_audio.name)
                do_auralize(
                    midi_path=midi_tmp,
                    original_audio_path=tmp_audio.name,
                    output_path=out_tmp,
                )
            with open(out_tmp, "rb") as f:
                wav_bytes = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.unlink(p)

        return Response(content=wav_bytes, media_type="audio/wav")

    @app.post("/sheets")
    async def sheets(
        midi: Annotated[UploadFile, File()],
        quantized: Annotated[bool, Form()] = False,
    ) -> Response:
        """Engrave a MIDI file as sheet music, returned as one zip.

        The whole set is rendered in one go — MuseScore is slow enough that a
        round trip per file would be worse — so the caller gets every PDF, the
        MusicXML and the MIDI in a single uncompressed archive and picks from it
        locally. Requires MuseScore 4+ on the server (503 without it).

        `quantized` says the upload is already snapped to a beat grid — a
        quantized MIDI from /export — which is what the notation should be
        engraved from. Without it the engraving keeps the timing jitter, so this
        does not quantize anything itself.
        """
        midi_bytes = await midi.read()

        try:
            zip_bytes = await asyncio.to_thread(engrave_to_zip, midi_bytes, quantized)
        except MuseScoreNotFoundError as e:
            # A deployment problem, not a bad request: the same 503 the UI
            # already knows how to report, with the install hint as its detail.
            raise HTTPException(status_code=503, detail=str(e)) from e
        except MuseScoreError as e:
            # MuseScore ran but wrote nothing usable — most often because the
            # upload wasn't a MIDI file it could import.
            raise HTTPException(status_code=500, detail=str(e)) from e

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{SHEETS_ZIP_NAME}"'
            },
        )

    if web_dir is not None:
        web_path = Path(web_dir)
        if web_path.is_dir():
            app.mount("/", StaticFiles(directory=web_path, html=True), name="web")

    return app
