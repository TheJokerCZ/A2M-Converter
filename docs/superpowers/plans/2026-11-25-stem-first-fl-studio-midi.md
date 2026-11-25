# Stem-First Studio for FL Studio MIDI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade this local MuScriptor Python/React application into a Stem-First Audio-to-MIDI studio with audio-derived per-note velocities, interactive Piano Roll editing, and one-click FL Studio bundle export (multi-track MIDI, per-instrument MIDIs, C5-normalized drum lanes, and bar-1-aligned WAV stems).

**Architecture:** The backend adds modular post-processing (`cleanup.py`, `velocity.py`, `fl_export.py`), GPU-safe sequential stem separation (`stems.py`), bounded session storage (`sessions.py`), lazy VRAM-aware model lifecycle (`model_manager.py`), and additive Studio SSE / FL-export endpoints in `server.py` and `main.py` while keeping all legacy wire formats and existing unit tests untouched. The React/Vite frontend adds a Studio setup panel, live multi-stage progress and SSE error recovery, an audio-aligned interactive Piano Roll editor with undo/redo and instrument merging, stem reference auditioning, and an FL Studio export dialog.

**Tech Stack:** Python 3.13, PyTorch (CUDA), `mido`, `soundfile`, `demucs-infer` (optional `demucs` fallback), FastAPI, Typer, pytest; React 19, TypeScript 6, Vite 8, Tailwind CSS 4, `spessasynth_lib`, `tone`, `fflate`.

**Spec:** [docs/superpowers/specs/2026-11-25-stem-first-fl-studio-midi-design.md](../../specs/2026-11-25-stem-first-fl-studio-midi-design.md)

## Global Constraints

- Hardware target: Windows with NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM), Python 3.13.5, Node v24.18.0 (`uv`/`pnpm` absent from PATH; use project `.venv` + `pip` and `npm` while keeping `pyproject.toml` and `web/package.json` valid).
- Strict sequential GPU memory lifecycle on 8 GB VRAM: release MuScriptor from CUDA before Demucs, unload Demucs + `torch.cuda.empty_cache()` after separation, auto-select `bfloat16` (if CUDA bf16 supported) or `float16` when loading `large` (`dim >= 1536`) on CUDA GPUs with `<= 12 GiB` VRAM and `dtype is None`.
- Zero regressions on existing unit tests: every pre-existing test in `tests/` must pass completely unmodified (`NoteStartEvent` fields untouched, `create_app(model)` signature preserved, `POST /transcribe` legacy path unchanged when `studio=False`, CLI default `--separation none` and legacy `--format midi` deterministic).
- Optional separator dependency: app must boot and work cleanly in `Full Mix` mode when `demucs-infer` / `demucs` is not installed.
- Fast deterministic tests: all `pytest` unit tests use synthetic tensors and stubbed separator/model backends with zero network or HuggingFace weight downloads.

## Review Focus

1. **Silent/instrumental stems or all-silent input:** Silent stems (`peak < -40 dBFS` or active ratio `< 0.5%`) are skipped cleanly, and an empty note list after separation or cleanup produces a valid empty Type 1 MIDI and ZIP without crashing on `min()` or empty percentile calls (pinned in Tasks 2, 3, 4, 5, 6).
2. **Edited notes double-shifting by `onset_delay`:** Interactive Piano Roll notes live in audio-aligned time (already shifted by `-onset_delay`), so exporting edited notes passes `onset_delay: 0.0` to `/export` and `/export/fl-bundle` instead of subtracting `onset_delay` a second time (pinned in Tasks 9 & 10).
3. **Missing HuggingFace auth during lazy Studio startup:** `muscriptor serve` boots without crashing, `/capabilities` reports `hf_auth_configured: false`, and if a user submits `/transcribe` (`studio=true`), `ModelDownloadError` emits a structured SSE `error` event and releases the transcription lock so subsequent requests never hang with 503 (pinned in Task 6).
4. **Non-44.1 kHz or mono input audio in stem separation and WAV alignment:** `separate_stems` handles mono `[1, T]` and stereo `[2, T]` at arbitrary sample rates (e.g., 48 kHz), and `align_wav_to_offset` pads/trims exact sample counts for both `offset_s > 0` and `offset_s < 0` (including when `trim_leading_bars` exceeds clip length, clamping cleanly to empty/zero-safe audio rather than negative slicing) (pinned in Tasks 4 & 5).
5. **All notes filtered out or excluded before export:** When `excluded_instruments`, `min_velocity`, or `min_note_duration_ms` filters out every note, `notes_to_midi` and `build_fl_bundle` still produce a valid MIDI (conductor track + tempo) and valid ZIP archive with `00_full_mix.mid` and `README_FL_STUDIO.txt` (pinned in Tasks 2 & 5).

---

## File Structure

### Backend (`muscriptor/`)
- Modify: `pyproject.toml` — add optional `[project.optional-dependencies] stems = ["demucs-infer"]`.
- Modify: `muscriptor/events.py` — add `StageEvent` dataclass (leave `NoteStartEvent` unchanged).
- Modify: `muscriptor/tokenizer/notes.py` — add `velocity: int = 100` to `Note`, `midi_velocity: int | None = None` to `NoteEvent`, preserve velocities in `note2note_event`, support `midi_velocity` and `split_drums: bool = False` in `note_event2midi`.
- Create: `muscriptor/utils/cleanup.py` — GM drum lane constants/mapping (`DRUM_LANES`, `SINGLE_HIT_DRUM_LANES`, `C5_PITCH`, `drum_lane_for_pitch`) and non-destructive `cleanup_notes()`.
- Modify: `muscriptor/utils/midi.py` — extend `ExportOptions` with velocity/cleanup/drum-split/exclude/remap fields and apply `cleanup_notes()` + `split_drums` in `notes_to_midi()`.
- Create: `muscriptor/utils/velocity.py` — vectorized STFT melodic & drum-band audio velocity estimator (`estimate_note_velocities()`).
- Modify: `muscriptor/transcription_model.py` — add `allowed_instruments: list[str] | None = None` to `transcribe()`, auto-select bf16/fp16 for `large` on `<= 12 GiB` CUDA GPUs in `load_model()`, carry note velocities through `events_to_midi_bytes(..., note_velocities=...)`.
- Create: `muscriptor/utils/stems.py` — `SeparationMode`, `StemPlan`, `is_separation_available()`, `available_separation_modes()`, `is_stem_silent()`, `plan_stem_conditioning()`, `separate_stems()` with strict GPU teardown.
- Create: `muscriptor/utils/fl_export.py` — `align_wav_to_offset()` and `build_fl_bundle()` producing `<song>_fl_studio.zip` (`00_full_mix.mid`, `midi/*.mid`, optional C5 drum lanes, optional bar-aligned `stems/*.wav`, `README_FL_STUDIO.txt`).
- Create: `muscriptor/sessions.py` — bounded LRU `SessionStore` for separated `44.1 kHz` WAV stems with strict UUID + allowlist validation.
- Create: `muscriptor/model_manager.py` — lazy `ModelManager` with `hf_auth_configured()`, `capabilities()`, `get_model()`, `release_gpu()`, and custom-checkpoint lock support.
- Modify: `muscriptor/server.py` — `GET /capabilities`, Studio SSE flow on `POST /transcribe`, `GET /sessions/{session_id}/stems/{stem_name}.wav`, extended `POST /export`, and `POST /export/fl-bundle`.
- Modify: `muscriptor/main.py` — CLI flags for separation, velocity estimation, cleanup, `--format fl-bundle`, and lazy `muscriptor serve` (`--preload/--no-preload`).

### Frontend (`web/`)
- Modify: `web/vite.config.ts` — proxy `/capabilities` and `/sessions` to backend in dev server.
- Create: `web/src/capabilities.ts` — `/capabilities` client & types (`ServerCapabilities`, `SeparationMode`, `StudioConfig`).
- Create: `web/src/components/StudioSettingsPanel.tsx` — separation mode, active stems, model size, decoding quality, and HF auth notice.
- Create: `web/src/editor.ts` — canonical audio-aligned editable note state (`EditableNote`), undo/redo history (50 snapshots), note edit operations (delete, move, resize, add, setVelocity, mergeInstrument), and `toExportPayload()` (sets `onset_delay: 0`).
- Modify: `web/src/sse.ts` & `web/src/hooks/useTranscription.ts` — send `studio=true` + Studio config, handle `stage` and `error` SSE events, apply `velocities` map, store `sessionId`/`stems`, convert notes to audio-aligned `EditableNote[]`.
- Modify: `web/src/audio.ts` — real per-note `velocity` playback in `spessasynth_lib`, `replaceNotes()` live sync, `loadWavUrl()` + stem buffer cache for switching reference audio between Original Mix and separated stems.
- Modify: `web/src/pianoroll.ts` & `web/src/components/PianoRollCanvas.tsx` — velocity-modulated rendering, selection highlights, marquee box, `Scrub`/`Edit` mode toolbar, hit-testing (body vs right edge), drag move/resize, double-click add, keyboard shortcuts, selected-note velocity slider, Undo/Redo controls.
- Modify: `web/src/components/InstrumentList.tsx` — note count badges, lifted mute state, and per-row **"Merge into…"** action.
- Modify: `web/src/components/Controls.tsx` — reference audio dropdown (`Original Mix` / separated stems).
- Modify: `web/src/export.ts`, create `web/src/components/FlExportDialog.tsx`, modify `web/src/components/OutputBar.tsx`, `web/src/components/WelcomeScreen.tsx`, `web/src/App.tsx` — wire Studio setup, stage progress, interactive editor, stem auditioning, and FL Studio Bundle / enhanced MIDI export.

---

### Task 1: Dev Environment Setup & Core Velocity / Event Data Model

**Files:**
- Modify: `pyproject.toml`
- Modify: `muscriptor/events.py:20-55`
- Modify: `muscriptor/tokenizer/notes.py:19-45,246-413`
- Test: `tests/test_notes.py` (append new tests only; keep existing tests unmodified)

**Interfaces:**
- Produces:
  - `muscriptor.events.StageEvent(stage: str, message: str, stem: str | None = None, stem_index: int | None = None, stem_total: int | None = None)`
  - `muscriptor.tokenizer.notes.Note(is_drum: bool, program: int, onset: float, offset: float, pitch: int, velocity: int = 100)`
  - `muscriptor.tokenizer.notes.NoteEvent(is_drum: bool, program: int, time: float, velocity: int, pitch: int, midi_velocity: int | None = None)`
  - `muscriptor.tokenizer.notes.note_event2midi(..., split_drums: bool = False) -> MidiFile` (existing params unchanged)

- [ ] **Step 1: Set up local Python `.venv` (CUDA PyTorch + editable package + pytest) and verify green baseline**

Run:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e . pytest
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: All pre-existing tests pass (0 failures) and `torch.cuda.is_available()` is `True` on the RTX 4070.

- [ ] **Step 2: Write failing tests for per-note MIDI velocity and `StageEvent`**

Append to `tests/test_notes.py`:
```python
def test_note_velocity_preserved_in_midi():
    notes = [
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=60, velocity=42),
        Note(is_drum=False, program=0, onset=0.5, offset=1.0, pitch=64, velocity=118),
        Note(is_drum=True, program=DRUM_PROGRAM, onset=0.25, offset=0.26, pitch=36, velocity=125),
    ]
    midi = note_event2midi(note2note_event(notes), velocity=100)
    note_ons = [
        (msg.note, msg.velocity)
        for track in midi.tracks
        for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    ]
    assert (60, 42) in note_ons
    assert (64, 118) in note_ons
    assert (36, 125) in note_ons


def test_legacy_note_event_without_midi_velocity_uses_default_velocity():
    events = [
        NoteEvent(is_drum=False, program=0, time=0.0, velocity=1, pitch=60),
        NoteEvent(is_drum=False, program=0, time=0.5, velocity=0, pitch=60),
    ]
    midi = note_event2midi(events, velocity=88)
    note_ons = [msg.velocity for t in midi.tracks for msg in t if msg.type == "note_on"]
    assert note_ons == [88]


def test_stage_event_fields():
    from muscriptor.events import StageEvent

    ev = StageEvent(stage="transcribing", message="Transcribing drums", stem="drums", stem_index=1, stem_total=4)
    assert ev.stage == "transcribing"
    assert ev.stem == "drums"
    assert ev.stem_index == 1
    assert ev.stem_total == 4
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_notes.py -k "velocity_preserved or legacy_note_event or stage_event" -v`
Expected: FAIL (`TypeError` on `Note(..., velocity=...)` / `ImportError` on `StageEvent`).

- [ ] **Step 4: Implement `StageEvent`, `Note.velocity`, `NoteEvent.midi_velocity`, and optional `[project.optional-dependencies] stems`**

- In `pyproject.toml`: add `[project.optional-dependencies]` with `stems = ["demucs-infer"]`.
- In `muscriptor/events.py`: add `@dataclass class StageEvent` with `stage: str`, `message: str`, `stem: str | None = None`, `stem_index: int | None = None`, `stem_total: int | None = None`. Do **not** alter `NoteStartEvent`.
- In `muscriptor/tokenizer/notes.py`:
  - Add `velocity: int = 100` to `Note`.
  - Add `midi_velocity: int | None = None` to `NoteEvent`.
  - In `note2note_event(notes)`, pass `midi_velocity=max(1, min(127, int(note.velocity)))` on onset `NoteEvent`s (and `midi_velocity=None` on offset `NoteEvent`s).
  - In `note_event2midi()`, compute `msg_velocity = max(1, min(127, int(ne.midi_velocity if ne.midi_velocity is not None else velocity))) if ne.velocity > 0 else 0`.

- [ ] **Step 5: Run full test suite to verify pass and zero regressions**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: All existing and new tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml muscriptor/events.py muscriptor/tokenizer/notes.py tests/test_notes.py
git commit -m "feat: add per-note velocity fields and StageEvent without changing wire contract"
```

---

### Task 2: Non-Destructive MIDI Cleanup, Drum Lanes & Extended `ExportOptions`

**Files:**
- Create: `muscriptor/utils/cleanup.py`
- Modify: `muscriptor/tokenizer/notes.py:283-413`
- Modify: `muscriptor/utils/midi.py:26-66,353-420`
- Create: `tests/test_cleanup.py`

**Interfaces:**
- Consumes: `Note`, `DRUM_PROGRAM`, `note_event2midi` (Task 1); `MT3_FULL_PLUS_GROUP_NAMES` (`muscriptor/tokenizer/mt3.py`).
- Produces:
  - `muscriptor.utils.cleanup.DRUM_LANES: dict[str, frozenset[int]]` (`kick`, `snare`, `hihat`, `toms`, `cymbals`, `perc`)
  - `muscriptor.utils.cleanup.DRUM_LANE_ORDER: tuple[str, ...]` = `("kick", "snare", "hihat", "toms", "cymbals", "perc")`
  - `muscriptor.utils.cleanup.DRUM_LANE_LABELS: dict[str, str]` (`"Drums - Kick"`, `"Drums - Snare"`, `"Drums - Hi-Hat"`, `"Drums - Toms"`, `"Drums - Cymbals"`, `"Drums - Perc"`)
  - `muscriptor.utils.cleanup.SINGLE_HIT_DRUM_LANES: frozenset[str]` = `frozenset({"kick", "snare", "hihat"})`
  - `muscriptor.utils.cleanup.C5_PITCH = 60`
  - `muscriptor.utils.cleanup.drum_lane_for_pitch(pitch: int) -> str`
  - `muscriptor.utils.cleanup.cleanup_notes(notes: list[Note], *, min_note_duration_ms: float = 0.0, min_velocity: int = 1, deduplicate_ms: float = 0.0, excluded_instruments: Sequence[str] = (), instrument_remap: Mapping[str, str] | None = None, use_velocities: bool = True, velocity_scale: float = 1.0, program_names: Mapping[int, str] | None = None, name_to_program: Mapping[str, int] | None = None) -> tuple[list[Note], dict[int, str]]`
  - Extended `muscriptor.utils.midi.ExportOptions` fields: `use_velocities: bool = True`, `velocity_scale: float = 1.0`, `min_note_duration_ms: float = 0.0`, `min_velocity: int = 1`, `deduplicate_ms: float = 0.0`, `split_drums: bool = False`, `excluded_instruments: tuple[str, ...] = ()`, `instrument_remap: dict[str, str] | None = None`.

- [ ] **Step 1: Write failing tests in `tests/test_cleanup.py`**

Include tests covering:
- `test_drum_lane_for_pitch_covers_gm_map`: `36 -> "kick"`, `38 -> "snare"`, `42 -> "hihat"`, `45 -> "toms"`, `49 -> "cymbals"`, `56 -> "perc"` (cowbell), `80 -> "perc"`.
- `test_cleanup_filters_duration_velocity_and_deduplicates`: drops melodic note `< 35 ms` (keeps drum hits regardless of short duration), drops notes with `velocity < 20`, deduplicates two identical `(program=0, pitch=60)` hits 15 ms apart keeping higher velocity and `max(offset)`.
- `test_cleanup_excludes_and_remaps_instruments`: drops `"voice"`, remaps `"electric_piano" -> "acoustic_piano"` updating `note.program` and `program_names`.
- `test_cleanup_velocity_flatten_and_scale`: `use_velocities=False` sets all velocities to `100`; `velocity_scale=1.5` widens contrast around midpoint `77` clamped to `[1, 127]`.
- `test_notes_to_midi_split_drums_creates_named_drum_tracks`: kick (`36`), snare (`38`), hi-hat (`42`) with `ExportOptions(split_drums=True)` produce 3 named drum tracks (`"Drums - Kick"`, `"Drums - Snare"`, `"Drums - Hi-Hat"`) all on MIDI channel `9` with original GM pitches intact.
- `test_notes_to_midi_all_notes_filtered_out_returns_valid_empty_midi` *(Review Focus #1/#5)*: `ExportOptions(min_velocity=127)` filtering out every note produces a valid Type 1 `MidiFile` with conductor track and `set_tempo` without raising.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cleanup.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'muscriptor.utils.cleanup'`).

- [ ] **Step 3: Implement `muscriptor/utils/cleanup.py`, drum-track splitting in `note_event2midi`, and extended `ExportOptions` in `muscriptor/utils/midi.py`**

- In `muscriptor/utils/cleanup.py`:
  - Define GM drum sets from Spec Section 5.4: `kick={35,36}`, `snare={37,38,39,40}`, `hihat={42,44,46}`, `toms={41,43,45,47,48,50}`, `cymbals={49,51,52,53,55,57,59}`, `perc=everything else`.
  - Implement `cleanup_notes()` operating on shallow copies (never mutating caller's `Note` list): normalize instrument names (`name.replace(" ", "_")`) when matching `excluded_instruments` and `instrument_remap`, apply `min_velocity`, apply `min_note_duration_ms` to non-drums only (`(offset - onset) * 1000.0 >= min_note_duration_ms`), deduplicate within `(is_drum, program, pitch)` when `next.onset - prev.onset <= deduplicate_ms / 1000.0`, apply `use_velocities` / `velocity_scale`.
- In `muscriptor/tokenizer/notes.py`:
  - Add `split_drums: bool = False` to `note_event2midi()`. When `split_drums=True` and `ne.is_drum`, key drum tracks by `("drum_lane", drum_lane_for_pitch(ne.pitch))` on channel `9`, ordered by `DRUM_LANE_ORDER`, named with `DRUM_LANE_LABELS[lane]`.
- In `muscriptor/utils/midi.py`:
  - Add new fields to `ExportOptions` (note: keep `frozen=True`; use `tuple[tuple[str, str], ...] = ()` or normalize `dict` in `ExportOptions` — use `excluded_instruments: tuple[str, ...] = ()` and `instrument_remap: tuple[tuple[str, str], ...] = ()` or allow `dict` in `__post_init__` via `object.__setattr__` so both `dict` and `tuple` work cleanly with frozen dataclass!).
  - In `notes_to_midi()`: run `cleanup_notes()` **before** `timeline_for` (so `min_note_duration_ms` and `deduplicate_ms` are measured accurately in seconds!), handle empty `notes` safely when computing `earliest = -delay` and `leading_bars`, and pass `split_drums=options.split_drums` to `note_event2midi()`.

- [ ] **Step 4: Run tests to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cleanup.py tests/test_midi.py tests/test_notes.py -v`
Expected: PASS (0 failures).

- [ ] **Step 5: Commit**

```powershell
git add muscriptor/utils/cleanup.py muscriptor/tokenizer/notes.py muscriptor/utils/midi.py tests/test_cleanup.py
git commit -m "feat: add non-destructive MIDI cleanup filters and split drum lanes"
```

---

### Task 3: Audio-Derived Per-Note Velocity Estimator

**Files:**
- Create: `muscriptor/utils/velocity.py`
- Create: `tests/test_velocity.py`

**Interfaces:**
- Consumes: `Note`, `DRUM_PROGRAM` (`muscriptor/tokenizer/notes.py`); `drum_lane_for_pitch` (`muscriptor/utils/cleanup.py`).
- Produces:
  - `muscriptor.utils.velocity.estimate_note_velocities(notes: list[Note], audio_by_stem: Mapping[str, tuple[torch.Tensor, int]], note_stems: Sequence[str] | None = None, vel_min: int = 32, vel_max: int = 122, default_velocity: int = 100) -> list[int]`
  - `muscriptor.utils.velocity.apply_estimated_velocities(notes: list[Note], audio_by_stem: Mapping[str, tuple[torch.Tensor, int]], note_stems: Sequence[str] | None = None, vel_min: int = 32, vel_max: int = 122) -> list[Note]`

- [ ] **Step 1: Write failing tests in `tests/test_velocity.py`**

Include tests covering:
- `test_melodic_louder_note_gets_higher_velocity`: synthetic 440 Hz (A4, pitch 69) sine sequence with 4 notes at amplitudes `[0.05, 0.2, 0.5, 0.9]` on `"other"` stem → estimated velocities are strictly increasing, bounded within `[32, 122]`, with spread `>= 40`.
- `test_drum_lane_band_energy_separates_kick_and_hihat`: synthetic 44.1 kHz `"drums"` stem with loud kick burst (`70 Hz`, amp `0.9`), soft kick burst (`70 Hz`, amp `0.15`), loud hi-hat noise burst (`8 kHz`, amp `0.6`), soft hi-hat (`8 kHz`, amp `0.1`) → loud kick > soft kick by `>= 25` AND loud hi-hat > soft hi-hat by `>= 25`.
- `test_uniform_loudness_track_does_not_amplify_noise`: 6 identical-amplitude notes produce velocities within `<= 5` of each other (minimum `24 dB` dynamic span anchor prevents wild noise amplification).
- `test_empty_and_silent_audio_safe` *(Review Focus #1)*: empty `notes=[]` returns `[]`; notes over all-zero silent tensor return valid ints in `[1, 127]` without `NaN` or exceptions; `< 3` notes on non-silent audio anchor cleanly against stem RMS (~`60..115`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_velocity.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'muscriptor.utils.velocity'`).

- [ ] **Step 3: Implement `muscriptor/utils/velocity.py`**

Follow Spec Section 5.3:
- Downmix each stem waveform to 1D CPU `float32` mono tensor (preserving its native `sr` — e.g., `44.1 kHz` for crisp hi-hat/cymbal bands, `16 kHz` fallback).
- Cache one melodic STFT (`hop ≈ 10 ms`, `n_fft = 2048` if `sr <= 24000` else `4096`) and one drum transient STFT (`hop ≈ 5 ms`, `n_fft = 512` if `sr <= 24000` else `1024`) per stem.
- Melodic energy: attack window `[onset - 0.01, min(offset, onset + 0.08)]`, sum weighted magnitudes across `±0.6` semitone bins around `f0` (`1.0`), `2*f0` (`0.5`), `3*f0` (`0.25`), take 80th percentile frame energy → dB.
- Drum energy: transient window `[onset - 0.005, onset + 0.035]`, peak band RMS across lane bands (`kick: 35–140 Hz`, `snare: 150–2500 Hz`, `toms: 80–450 Hz`, `hihat: 4000–14000 Hz`, `cymbals: 3000–14000 Hz`, `perc: 200–6000 Hz`, upper bound clamped to `0.95 * sr / 2`) → dB.
- Group by `(program, is_drum, lane_if_drum)`:
  - If group size `>= 3`: `hi = p95`, `lo = min(p5, hi - 24.0)`; map linearly `[lo, hi] -> [vel_min, vel_max]`. If all values are within `1e-3 dB` (flat/silent), map to midpoint or RMS-anchored value.
  - If group size `< 3`: anchor relative to overall stem RMS dB (`[rms_db - 24, rms_db + 12] -> [vel_min, vel_max]`).
- Clamp every output to `[1, 127]`.

- [ ] **Step 4: Run tests to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_velocity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add muscriptor/utils/velocity.py tests/test_velocity.py
git commit -m "feat: add audio-derived per-note STFT velocity estimation"
```

---

### Task 4: Decoupled Logit Masking (`allowed_instruments`), Low-VRAM `large` Auto-Dtype & Stem Separation

**Files:**
- Modify: `muscriptor/transcription_model.py:273-465,704-757`
- Create: `muscriptor/utils/stems.py`
- Modify: `tests/test_strict_instruments.py` (append new tests only; keep existing tests unmodified)
- Create: `tests/test_stems.py`

**Interfaces:**
- Consumes: `MT3_FULL_PLUS_GROUP_NAMES` (`muscriptor/tokenizer/mt3.py`); `resample` (`muscriptor/utils/audio.py`).
- Produces:
  - `TranscriptionModel.transcribe(..., allowed_instruments: list[str] | None = None)`
  - `TranscriptionModel.events_to_midi_bytes(events, beat_grid=None, export=None, note_velocities: Mapping[int, int] | None = None) -> bytes`
  - `muscriptor.utils.stems.SeparationMode = Literal["none", "stems_4", "stems_6"]`
  - `muscriptor.utils.stems.STEM_NAMES_BY_MODE: dict[str, tuple[str, ...]]`
  - `muscriptor.utils.stems.StemPlan(stem: str, instruments: list[str] | None, allowed_instruments: list[str] | None)`
  - `muscriptor.utils.stems.is_separation_available() -> bool`
  - `muscriptor.utils.stems.available_separation_modes() -> list[str]`
  - `muscriptor.utils.stems.is_stem_silent(wav: torch.Tensor, sr: int = 16000, peak_db: float = -40.0, active_ratio: float = 0.005) -> bool`
  - `muscriptor.utils.stems.plan_stem_conditioning(mode: SeparationMode, active_stems: Sequence[str] | None = None, user_instruments: Sequence[str] | None = None) -> list[StemPlan]`
  - `muscriptor.utils.stems.separate_stems(wav: torch.Tensor, sr: int, mode: SeparationMode, device: str | torch.device | None = None, segment: float = 7.5) -> dict[str, tuple[torch.Tensor, int]]`

- [ ] **Step 1: Write failing tests in `tests/test_strict_instruments.py` and `tests/test_stems.py`**

- Append to `tests/test_strict_instruments.py`:
  - `test_allowed_instruments_masks_logits_without_class_conditioning`: calling `transcribe(audio, instruments=None, allowed_instruments=["acoustic_piano", "synth_lead"])` sets `instrument_group=None` in `ConditioningAttributes.text` AND sets `forbidden_tokens` equal to `tokenizer.forbidden_token_ids(["acoustic_piano", "synth_lead"])`.
- Create `tests/test_stems.py`:
  - `test_plan_stem_conditioning_auto_4_stems`: `drums` gets `instruments=["drums"]`, `vocals` gets `instruments=["voice"]`, `bass` gets `instruments=None, allowed_instruments=["electric_bass", "acoustic_bass", "contrabass"]`, `other` gets `instruments=None, allowed_instruments=[...]` with `"drums"`, `"voice"`, `"electric_bass"`, `"acoustic_bass"`, `"contrabass"` absent!
  - `test_plan_stem_conditioning_auto_6_stems`: `guitar` and `piano` get their respective `allowed_instruments` (with `instruments=None`), and `other` excludes drums, vocals, bass, guitar, and piano.
  - `test_plan_stem_conditioning_with_explicit_user_instruments`: `user_instruments=["drums", "acoustic_piano"]` in `stems_4` produces plans ONLY for `drums` (`instruments=["drums"]`) and `other` (`instruments=["acoustic_piano"]`), skipping `bass` and `vocals`.
  - `test_is_stem_silent_detects_silence_and_activity` *(Review Focus #1)*: zero/noise tensor at `-60 dBFS` is silent (`True`); 440 Hz tone at `-12 dBFS` for 0.2s in a 2s clip is active (`False`).
  - `test_separate_stems_handles_mono_and_48khz_and_clears_cuda` *(Review Focus #4)*: with a monkeypatched stub Demucs separator accepting `[2, T]` stereo at `44100 Hz`, passing mono `[1, 48000]` at `48000 Hz` resamples to `44100 Hz`, duplicates mono to stereo, returns 4 stereo CPU tensors of matching length, and calls GPU cleanup in `finally`.
  - `test_auto_dtype_selects_half_for_large_on_8gb_cuda`: monkeypatching CUDA `total_memory = 8 * 1024**3` resolves `bfloat16` (when bf16 supported) or `float16` for `large` (`dim=1536`), while keeping `float32` for `medium` (`dim=1024`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_strict_instruments.py tests/test_stems.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement decoupled masking & auto-dtype in `muscriptor/transcription_model.py` and stem separation in `muscriptor/utils/stems.py`**

- In `muscriptor/transcription_model.py`:
  - Extract helper `_auto_dtype(device: torch.device, cfg: _ModelConfig) -> torch.dtype`: `float16` on `mps`; on `cuda` with `cfg.dim >= 1536` and `torch.cuda.get_device_properties(device).total_memory <= 12 * 1024**3`, return `torch.bfloat16` if `torch.cuda.is_bf16_supported()` else `torch.float16`; else `torch.float32`. Call in `load_model()` when `dtype is None` (after resolving `cfg`!).
  - In `transcribe()`: add `allowed_instruments: list[str] | None = None`. Keep `instrument_group = instrument_group_from_names(instruments) if instruments else None`. Compute `mask_names = instruments if instruments else allowed_instruments`; if `mask_names`, build `forbidden_tokens = tokenizer.forbidden_token_ids(mask_names)`.
  - In `events_to_midi_bytes()`: add optional `note_velocities: Mapping[int, int] | None = None`; when provided, set `note.velocity = note_velocities.get(ev.index, 100)` on `NoteStartEvent`.
- In `muscriptor/utils/stems.py`:
  - Implement `STEM_NAMES_BY_MODE`, `plan_stem_conditioning()`, `is_stem_silent()`, `is_separation_available()`, `available_separation_modes()`.
  - Implement `separate_stems()`: supports `demucs-infer` (`demucs_infer.pretrained.get_model`, `demucs_infer.apply.apply_model`) with fallback to `demucs` (`demucs.pretrained.get_model`, `demucs.apply.apply_model`). Handles arbitrary input `sr` and mono/stereo `[C, T]`, runs in `torch.inference_mode()` with `segment=segment, split=True`, moves outputs to CPU `.float()`, and deletes model + `torch.cuda.empty_cache()` in `finally`.

- [ ] **Step 4: Install `demucs-infer` in `.venv` (if wheels compatible on Python 3.13) and run tests**

Run:
```powershell
.\.venv\Scripts\python.exe -m pip install demucs-infer
.\.venv\Scripts\python.exe -m pytest tests/test_strict_instruments.py tests/test_stems.py tests/test_transcription_model.py -v
```
Expected: All tests PASS. (Verify `is_separation_available()` returns `True` in `.venv`.)

- [ ] **Step 5: Commit**

```powershell
git add muscriptor/transcription_model.py muscriptor/utils/stems.py tests/test_strict_instruments.py tests/test_stems.py
git commit -m "feat: add decoupled allowed_instruments masking, 8GB large auto-dtype, and stem separation"
```

---

### Task 5: FL Studio Bundle Exporter & Bar-1 WAV Stem Alignment

**Files:**
- Create: `muscriptor/utils/fl_export.py`
- Create: `tests/test_fl_export.py`

**Interfaces:**
- Consumes: `Note`, `DRUM_PROGRAM`, `note2note_event`, `note_event2midi` (`muscriptor/tokenizer/notes.py`); `ExportOptions`, `resolve_grid`, `timeline_for`, `shifted_notes`, `quantized_notes`, `gaps_filled`, `leading_bars`, `TICKS_PER_BEAT`, `DRUM_HIT_BEATS` (`muscriptor/utils/midi.py`); `cleanup_notes`, `drum_lane_for_pitch`, `DRUM_LANE_ORDER`, `SINGLE_HIT_DRUM_LANES`, `C5_PITCH` (`muscriptor/utils/cleanup.py`); `BeatGrid` (`muscriptor/utils/beats.py`).
- Produces:
  - `muscriptor.utils.fl_export.align_wav_to_offset(wav: torch.Tensor, sr: int, offset_s: float) -> torch.Tensor`
  - `muscriptor.utils.fl_export.write_pcm16_wav_bytes(wav: torch.Tensor, sr: int) -> bytes`
  - `muscriptor.utils.fl_export.build_fl_bundle(notes: list[Note], *, program_names: Mapping[int, str] | None = None, beat_grid: BeatGrid | None = None, options: ExportOptions | None = None, stem_audio: Mapping[str, tuple[torch.Tensor, int]] | None = None, include_stems: bool = True, normalize_drum_lanes_to_c5: bool = True, song_name: str = "transcription") -> bytes`

- [ ] **Step 1: Write failing tests in `tests/test_fl_export.py`**

Include tests covering:
- `test_align_wav_to_offset_pads_and_trims_exact_samples` *(Review Focus #4)*:
  - `[2, 44100]` at `sr=44100`, `offset_s=+0.5` → shape `[2, 66150]`, first `22050` frames are `0.0`, tail equals original tensor.
  - `[1, 44100]` at `sr=44100`, `offset_s=-0.25` → shape `[1, 33075]`, equals original `[:, 11025:]`.
  - `offset_s=-5.0` exceeding 1s clip → returns safe non-crashing empty/minimal `[1, 0]` tensor instead of negative indexing.
- `test_build_fl_bundle_structure_c5_drums_and_aligned_stems`:
  - Notes across `acoustic_piano` (program 0), `electric_bass` (program 33), and drums (`36` kick, `38` snare, `45` tom) + steady `BeatGrid(bpm=120.0, beats_per_bar=4, first_downbeat=0.5, onset_delay=0.0)` + `ExportOptions(split_drums=True, quantize=True)` + 2 stereo stems (`drums`, `bass`).
  - Asserts ZIP contains: `00_full_mix.mid`, `midi/01_acoustic_piano.mid`, `midi/02_electric_bass.mid`, `midi/drums_kick.mid`, `midi/drums_snare.mid`, `midi/drums_toms.mid`, `stems/drums.wav`, `stems/bass.wav`, `README_FL_STUDIO.txt`.
  - Parses `midi/drums_kick.mid` and `midi/drums_snare.mid`: all note pitches are `60` (C5)!
  - Parses `midi/drums_toms.mid` and `00_full_mix.mid`: tom pitch remains `45`, combined MIDI kick remains `36` on channel `9`!
  - Verifies every individual `.mid` file has the exact same `set_tempo` and `BAR_OFFSET_MARKER` as `00_full_mix.mid`, and `stems/drums.wav` length matches `original_frames + round(offset_s * sr)`.
- `test_build_fl_bundle_c5_disabled_keeps_gm_pitches`: `normalize_drum_lanes_to_c5=False` keeps pitch `36` in `midi/drums_kick.mid`.
- `test_build_fl_bundle_variable_tempo_adds_warning_in_readme`: unsteady `BeatGrid(steady=False, beats=...)` writes a tempo-drift note in `README_FL_STUDIO.txt`.
- `test_build_fl_bundle_empty_notes_after_cleanup` *(Review Focus #5)*: returns valid ZIP with `00_full_mix.mid` and `README_FL_STUDIO.txt` and zero exceptions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_fl_export.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'muscriptor.utils.fl_export'`).

- [ ] **Step 3: Implement `muscriptor/utils/fl_export.py`**

- Compute ONE shared `grid`, `timeline`, and `offset_s = -timeline.seconds_at(0.0)` across all cleaned notes (exactly mirroring `notes_to_midi` so `00_full_mix.mid`, every `midi/*.mid`, and every `stems/*.wav` share identical bar-1 alignment!).
- Write `00_full_mix.mid` with `split_drums=options.split_drums`.
- Write one single-track `.mid` per non-drum program (`midi/<idx:02d>_<slug>.mid`) and either `midi/drums.mid` (`split_drums=False`) or `midi/drums_<lane>.mid` (`split_drums=True`), remapping pitches to `C5_PITCH` (`60`) on `SINGLE_HIT_DRUM_LANES` when `normalize_drum_lanes_to_c5=True` (using channel `0` when normalized to C5 so FL Sampler channels don't get routed to GM channel 9!).
- Align each stem in `stem_audio` with `align_wav_to_offset(wav, sr, offset_s)` and encode 16-bit PCM WAV via `wave` stdlib (`write_pcm16_wav_bytes`).
- Generate concise `README_FL_STUDIO.txt` documenting BPM, meter, alignment offset, files, and variable-tempo note when `not grid.steady`.
- Pack with `zipfile.ZipFile(..., compression=zipfile.ZIP_STORED)`.

- [ ] **Step 4: Run tests to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_fl_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add muscriptor/utils/fl_export.py tests/test_fl_export.py
git commit -m "feat: add FL Studio ZIP bundle exporter with C5 drum lanes and bar-1 aligned WAV stems"
```

---

### Task 6: Session Store, Lazy `ModelManager`, and Studio Server API

**Files:**
- Create: `muscriptor/sessions.py`
- Create: `muscriptor/model_manager.py`
- Modify: `muscriptor/server.py`
- Create: `tests/test_studio_server.py`

**Interfaces:**
- Consumes: `StageEvent` (Task 1); `ExportOptions` (Task 2); `estimate_note_velocities` (Task 3); `separate_stems`, `plan_stem_conditioning`, `is_stem_silent`, `is_separation_available`, `available_separation_modes` (Task 4); `build_fl_bundle`, `write_pcm16_wav_bytes` (Task 5).
- Produces:
  - `muscriptor.sessions.SessionStore(root_dir: Path | None = None, max_sessions: int = 3)` with `create_session(stems: Mapping[str, tuple[torch.Tensor, int]]) -> str`, `get_stem_path(session_id: str, stem_name: str) -> Path | None`, `load_stems(session_id: str) -> dict[str, tuple[torch.Tensor, int]]`, `cleanup() -> None`.
  - `muscriptor.model_manager.ModelManager(default_model: str | None = None, device: str | None = None, dtype: str | None = None)` with `hf_auth_configured() -> bool`, `capabilities() -> dict`, `get_model(model_size: str | None = None) -> TranscriptionModel`, `release_gpu() -> None`.
  - `muscriptor.server.create_app(model=None, *, web_dir=None, model_manager=None, session_store=None, separator_fn=None) -> FastAPI`
  - Endpoints: `GET /capabilities`, extended `POST /transcribe` (`studio`, `separation`, `stems`, `model_size`, `beam_size`), `GET /sessions/{session_id}/stems/{stem_name}.wav`, extended `POST /export`, `POST /export/fl-bundle`.

- [ ] **Step 1: Write failing tests in `tests/test_studio_server.py`**

Include tests covering:
- `test_capabilities_reports_models_separation_and_hf_auth`: checks keys `device`, `gpu_name`, `vram_mb`, `loaded_model`, `default_model`, `custom_model_locked`, `available_models`, `hf_auth_configured`, `separation_available`, `separation_modes`.
- `test_studio_transcribe_runs_sequential_gpu_stems_and_streams_stages`: stub `ModelManager` + stub `separator_fn` returning 4 stems (`drums`, `bass`, `vocals` silent, `other`). Asserts `release_gpu()` is called **before** `separator_fn`, silent `vocals` stem is skipped, `stage` events (`separating`, `transcribing` per active stem, `beats`, `velocity`) are emitted, streamed `start` events have strictly unique increasing `index` values across stems, `transcription_complete` includes `session_id`, `stems`, and `velocities` map (`str(idx) -> 1..127`), and `GET /sessions/{session_id}/stems/drums.wav` returns 200 `audio/wav`.
- `test_studio_transcribe_all_silent_stems_returns_valid_complete` *(Review Focus #1)*: when all stems are silent, `transcription_complete` is still emitted with valid base64 MIDI and `velocities == {}`.
- `test_studio_transcribe_model_download_error_emits_sse_error_and_releases_lock` *(Review Focus #3)*: stub `ModelManager.get_model` raising `ModelDownloadError("HF login required")` emits `{type: "error", message: "...HF login required..."}` over SSE, and a subsequent request succeeds (lock was not leaked).
- `test_session_stem_endpoint_blocks_path_traversal_and_unknown_stems`: `/sessions/../../etc/stems/drums.wav` and `/sessions/{valid}/stems/evil.wav` return 400/404.
- `test_session_store_lru_evicts_oldest_beyond_limit`: creating 4 sessions with `max_sessions=3` evicts session 1's directory on disk.
- `test_export_with_velocities_and_cleanup_options`: `POST /export` with `velocity` per note + `split_drums: true`, `min_velocity: 50` drops quiet notes and writes expected velocities and split drum tracks.
- `test_export_fl_bundle_returns_zip_with_aligned_stems`: `POST /export/fl-bundle` with `session_id`, `include_stems: true`, `normalize_drum_lanes_to_c5: true` returns 200 `application/zip` containing `00_full_mix.mid`, `midi/*.mid`, `stems/*.wav`, and `README_FL_STUDIO.txt`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_studio_server.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `muscriptor/sessions.py`, `muscriptor/model_manager.py`, and extend `muscriptor/server.py`**

- In `muscriptor/sessions.py`:
  - `SAFE_STEM_NAMES = frozenset({"drums", "bass", "vocals", "guitar", "piano", "other"})`, strict UUID4 validation regex.
  - Persist stems as 16-bit PCM WAV (`write_pcm16_wav_bytes`) in `<root>/<session_id>/<stem>.wav`.
  - Maintain `OrderedDict[str, Path]` with thread lock and LRU eviction (`shutil.rmtree(..., ignore_errors=True)`).
- In `muscriptor/model_manager.py`:
  - Detect `custom_model_locked`: `True` when `default_model` is not `None` and not in `("small", "medium", "large")`.
  - `hf_auth_configured()`: checks `os.environ.get("HF_TOKEN")` or `huggingface_hub.get_token() is not None`.
  - `get_model(model_size)`: if already loaded with same target, move back to target device if offloaded; if switching size, unload old model (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) and load new variant via `TranscriptionModel.load_model()`.
  - `release_gpu()`: if model is loaded on CUDA, unload/move to CPU + `torch.cuda.empty_cache()`.
- In `muscriptor/server.py`:
  - Support both `create_app(model)` (legacy direct model) and `create_app(model=None, model_manager=...)`.
  - Add `ExportNote.velocity: int = Field(default=100, ge=1, le=127)`.
  - Extend `ExportSettings` with `use_velocities`, `velocity_scale`, `min_note_duration_ms`, `min_velocity`, `deduplicate_ms`, `split_drums`, `excluded_instruments`, `instrument_remap`.
  - In `POST /transcribe`:
    - When `studio=False` and `separation == "none"` and `model_size is None` and `beam_size == 1`: execute the exact legacy streaming loop (preserving 100% compatibility with `tests/test_server.py`).
    - When `studio=True`: wrap generator in `try ... except Exception as e:` emitting `{type: "error", message: str(e)}`, emit `stage` events, manage GPU sequencing via `model_manager.release_gpu()` before separation, offset streamed `NoteStartEvent.index` cleanly across stems so every note across all stems has a unique global index, run `beat_this` once on full mix, run `estimate_note_velocities()` using the stem waveforms, save stems to `session_store`, build MIDI with `note_velocities`, and emit enriched `transcription_complete`.
  - Add `GET /capabilities`, `GET /sessions/{session_id}/stems/{stem_name}.wav`, and `POST /export/fl-bundle`.

- [ ] **Step 4: Run both new and existing server tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_server.py tests/test_studio_server.py -v`
Expected: ALL tests in both files PASS (`tests/test_server.py` completely unmodified!).

- [ ] **Step 5: Commit**

```powershell
git add muscriptor/sessions.py muscriptor/model_manager.py muscriptor/server.py tests/test_studio_server.py
git commit -m "feat: add lazy ModelManager, SessionStore, Studio SSE pipeline, and FL bundle endpoint"
```

---

### Task 7: CLI Parity (`muscriptor transcribe` & Lazy `muscriptor serve`)

**Files:**
- Modify: `muscriptor/main.py:46-590`
- Create: `tests/test_cli_studio.py`

**Interfaces:**
- Consumes: `ExportOptions` (Task 2), `estimate_note_velocities` (Task 3), `separate_stems`, `plan_stem_conditioning`, `is_stem_silent` (Task 4), `build_fl_bundle` (Task 5), `ModelManager`, `SessionStore` (Task 6).
- Produces:
  - `OutputFormat.fl_bundle = "fl-bundle"`
  - CLI options on `muscriptor transcribe`: `--separation`, `--stems`, `--estimate-velocity/--no-estimate-velocity`, `--min-note-ms`, `--min-velocity`, `--deduplicate-ms`, `--split-drums`, `--c5-drums/--no-c5-drums`, `--include-stems/--no-include-stems`.
  - CLI option on `muscriptor serve`: `--preload/--no-preload` (default `False`, boots via `ModelManager` lazily).

- [ ] **Step 1: Write failing tests in `tests/test_cli_studio.py`**

Include tests covering:
- `test_cli_fl_bundle_format_writes_zip`: invoking `transcribe audio.wav --format fl-bundle --split-drums -o out.zip` with stubbed model creates a valid ZIP containing `00_full_mix.mid`, `midi/*.mid`, and `README_FL_STUDIO.txt`.
- `test_cli_separation_stems_4_transcribes_active_stems_with_velocity`: invoking `transcribe audio.wav --separation stems_4 --stems drums,bass --format fl-bundle -o out.zip` with stubbed separator & model transcribes only `drums` and `bass`, estimates non-flat velocities, and embeds aligned `stems/drums.wav` + `stems/bass.wav` in the ZIP.
- `test_cli_serve_starts_lazily_without_loading_model`: invoking `serve` with monkeypatched `uvicorn.run` does NOT call `TranscriptionModel.load_model()` before `uvicorn.run` starts, and passes a `ModelManager` into `create_app`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli_studio.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement CLI Studio options and lazy `serve` in `muscriptor/main.py`**

- Add `fl_bundle = "fl-bundle"` to `OutputFormat` (default output filename `<audio_file.stem>_fl_studio.zip`; reject `-o -` with clear error).
- Add `--separation`, `--stems`, `--estimate-velocity/--no-estimate-velocity` (`bool | None = None`: defaults to `True` when `separation != "none"` or `format == OutputFormat.fl_bundle`, else `False` so existing CLI tests remain 100% deterministic!), `--min-note-ms`, `--min-velocity`, `--deduplicate-ms`, `--split-drums`, `--c5-drums/--no-c5-drums`, `--include-stems/--no-include-stems`.
- When `separation != "none"` or `format == OutputFormat.fl_bundle` or `estimate_velocity` is True: run stem separation (if requested) before loading MuScriptor on GPU, transcribe planned stems, compute `beat_grid` + `onset_delay` + `estimate_note_velocities`, and write either `.mid` or FL bundle `.zip`.
- Update `serve` to construct `ModelManager(default_model=model_path, device=_device, dtype=dtype)` and `SessionStore()`, preloading only when `--preload` is passed, and passing them to `create_app(model=None, web_dir=..., model_manager=manager, session_store=store)`.

- [ ] **Step 4: Run all CLI tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli.py tests/test_cli_studio.py -v`
Expected: PASS (`tests/test_cli.py` unmodified!).

- [ ] **Step 5: Commit**

```powershell
git add muscriptor/main.py tests/test_cli_studio.py
git commit -m "feat: add CLI stem separation, FL bundle export, and lazy serve startup"
```

---

### Task 8: Frontend Studio Setup, Stage Progress, SSE Error Recovery & Stem Auditioning

**Files:**
- Modify: `web/vite.config.ts:10-25`
- Create: `web/src/capabilities.ts`
- Create: `web/src/components/StudioSettingsPanel.tsx`
- Modify: `web/src/sse.ts`
- Modify: `web/src/hooks/useTranscription.ts`
- Modify: `web/src/audio.ts`
- Modify: `web/src/components/WelcomeScreen.tsx`
- Modify: `web/src/components/Controls.tsx`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes: Backend `GET /capabilities`, `POST /transcribe` (`studio=true`), `GET /sessions/{id}/stems/{stem}.wav` (Task 6).
- Produces:
  - `web/src/capabilities.ts`: `ServerCapabilities`, `SeparationMode`, `StudioConfig`, `fetchCapabilities()`
  - `AudioEngine.replaceNotes(notes: Array<{ instrument: string; pitch: number; start: number; end: number; velocity?: number }>)`
  - `AudioEngine.loadReferenceUrl(sourceKey: string, url: string | null)` (caches decoded stem `AudioBuffer`s and restores original mix when `sourceKey === "mix"`).
  - Updated `TranscriptionResult` carrying `sessionId: string | null`, `stems: string[]`, and notes with `id: number` and `velocity: number`.

- [ ] **Step 1: Install frontend dependencies in `web/` and verify clean baseline build**

Run:
```powershell
cd web
npm install
npm run build
```
Expected: `tsc --noEmit && vite build` succeeds cleanly.

- [ ] **Step 2: Add dev proxies, `/capabilities` client, and `StudioSettingsPanel.tsx`**

- Update `web/vite.config.ts` proxy list to include `"/capabilities"` and `"/sessions"`.
- Create `web/src/capabilities.ts` defining `SeparationMode = "none" | "stems_4" | "stems_6"`, `StudioConfig = { separation: SeparationMode; stems: string[]; modelSize: "small" | "medium" | "large"; beamSize: 1 | 2 }`, `ServerCapabilities`, and `fetchCapabilities()`.
- Create `web/src/components/StudioSettingsPanel.tsx`:
  - Renders separation mode segmented picker (`4 Stems — Best`, `6 Stems`, `Full Mix — Fast`), disabling stem options with hint when `!capabilities.separation_available`.
  - Renders toggle pills for active stems (`drums`, `bass`, `vocals`, `other` [+ `guitar`, `piano` in `stems_6`]).
  - Renders model selector (`small`, `medium`, `large`) with VRAM/speed badge, locked if `capabilities.custom_model_locked`.
  - Renders decoding quality toggle (`Standard` vs `Beam Search (2)`).
  - Shows a clear amber setup banner when `capabilities.hf_auth_configured === false` explaining `HF_TOKEN` / `hf auth login` + accepting CC BY-NC 4.0 on HuggingFace.

- [ ] **Step 3: Wire Studio SSE events (`stage`, `error`, `velocities`, `session_id`, `stems`) and velocity-aware stem playback**

- Update `web/src/hooks/useTranscription.ts`:
  - Send `studio: "true"`, `separation`, `stems`, `model_size`, `beam_size` in `extra` form fields.
  - Handle `ev.type === "stage"` via `onStage(ev)` callback so `App.tsx` / `OutputBar.tsx` displays the live stage text (`"Separating 4 stems…"`, `"Transcribing drums (1/4)…"`, `"Estimating note velocities…"`).
  - Handle `ev.type === "error"` by throwing `new TranscribeError(ev.message, { userMessage: ev.message })` so `onError(message)` surfaces the exact backend error on the Welcome screen!
  - On `transcription_complete`: apply `ev.velocities` (`Record<string, number>`) onto every streamed note, convert notes into audio-aligned times (`start - onsetDelay`, `end - onsetDelay`), update `rollRef` + `audio.replaceNotes(...)`, and store `sessionId` & `stems` on `TranscriptionResult`.
- Update `web/src/audio.ts`:
  - Add optional `velocity?: number` to `NoteOpts` (default `100`); pass `Math.max(1, Math.min(127, Math.round(opts.velocity ?? NOTE_VELOCITY)))` to `synth.noteOn(channel, opts.pitch, vel, { time })`!
  - Add `replaceNotes(notes: NoteOpts[])` that replaces `allNotes`, updates `autoStopAt`, and re-seeks at current transport position so live edits immediately affect playback.
  - Cache original decoded `mixBuffer` alongside a `stemBuffers = new Map<string, AudioBuffer>()` and add `setReferenceAudio(source: string, url?: string)` so switching between `Original Mix` and any stem (`Drums`, `Bass`, `Vocals`, `Other`…) swaps `wavBuffer` seamlessly (even mid-playback by restarting the buffer source at the current transport offset!).
- Update `web/src/components/Controls.tsx`: when `stems.length > 0`, show a compact Reference Audio `<select>` (`Original Mix` + each separated stem) right beside the Mix slider!
- Update `web/src/components/WelcomeScreen.tsx` and `web/src/App.tsx` to load `/capabilities`, render `StudioSettingsPanel`, pass `studioConfig` to `useTranscription`, and display stage progress text in `OutputBar`.

- [ ] **Step 4: Verify TypeScript typecheck and Vite build**

Run: `cd web && npm run build`
Expected: `tsc --noEmit && vite build` succeeds with 0 errors.

- [ ] **Step 5: Commit**

```powershell
git add web/vite.config.ts web/src/capabilities.ts web/src/components/StudioSettingsPanel.tsx web/src/hooks/useTranscription.ts web/src/audio.ts web/src/components/WelcomeScreen.tsx web/src/components/Controls.tsx web/src/components/OutputBar.tsx web/src/App.tsx
git commit -m "feat(web): add Studio setup panel, SSE stage/error handling, velocity synth playback, and stem auditioning"
```

---

### Task 9: Interactive Piano Roll Editor, Undo/Redo & Instrument Merging

**Files:**
- Create: `web/src/editor.ts`
- Create: `web/src/editor.test.mjs` (or Node built-in test runner test verifying pure TS/JS logic via `npx tsx --test` or node test — let's use Node's built-in `node --test` on pure logic or compiled check!)
- Modify: `web/src/pianoroll.ts`
- Modify: `web/src/components/PianoRollCanvas.tsx`
- Modify: `web/src/components/InstrumentList.tsx`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes: `BeatGrid`, `PianoRoll`, `RollNote` (`web/src/pianoroll.ts`); `AudioEngine.replaceNotes` (Task 8).
- Produces:
  - `web/src/editor.ts`:
    - `EditableNote = { id: number; instrument: string; pitch: number; start: number; end: number; velocity: number }` (always in **audio-aligned time**!)
    - `EditorHistory` class (max 50 snapshots: `push`, `undo`, `redo`, `canUndo`, `canRedo`)
    - Pure transforms: `deleteNotes`, `moveNotes`, `resizeNote`, `addNote`, `setNotesVelocity`, `mergeInstrument`, `toExportPayload(notes: EditableNote[], beatGrid: BeatGrid | null)` (returns `{ notes, beatGrid }` with `onset_delay: 0` so edited audio-aligned notes are NEVER double-shifted!).

- [ ] **Step 1: Write failing unit tests for `web/src/editor.ts` (including `onset_delay: 0` export invariant)**

Create `web/src/editor.test.ts` (runnable directly with Node 24's built-in TypeScript stripping: `node --test web/src/editor.test.ts`!):
- `test_toExportPayload_zeros_onset_delay_to_prevent_double_shift` *(Review Focus #2)*: passing audio-aligned `EditableNote[]` + `BeatGrid` with `onset_delay: 0.025` returns payload with `beatGrid.onset_delay === 0` and exact unchanged `start`/`end`/`velocity`.
- `test_editor_history_undo_redo_and_cap`: push edits, undo restores prior notes, redo reapplies, capped at 50 snapshots.
- `test_merge_instrument_consolidates_notes_and_preserves_velocities`: merging `"electric_piano"` into `"acoustic_piano"` updates every matching note's `instrument` without losing pitch/time/velocity.
- `test_move_resize_delete_and_velocity_clamping`: moving clamps pitch to `[21, 108]` and `start >= 0`; resizing enforces minimum duration `0.02s`; `setNotesVelocity` clamps to `[1, 127]`.

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test web/src/editor.test.ts`
Expected: FAIL (`Cannot find module './editor.ts'`).

- [ ] **Step 3: Implement `web/src/editor.ts` and verify unit tests pass**

Run: `node --test web/src/editor.test.ts`
Expected: PASS.

- [ ] **Step 4: Implement interactive editing in `pianoroll.ts`, `PianoRollCanvas.tsx`, `InstrumentList.tsx`, and `App.tsx`**

- In `web/src/pianoroll.ts`:
  - Extend `RollNote` with `id?: number` and `velocity?: number`.
  - Add `replaceNotes(notes: RollNote[])` (preserves viewport, beat grid, mute/hide sets, recalculates `stackOffset`).
  - Add `setSelectedIds(ids: ReadonlySet<number>)`, `setMarquee(rect: { x0: number; y0: number; x1: number; y1: number } | null)`, `yToPitch(y: number): number`, `hitTestNote(x: number, y: number): { id: number; edge: "body" | "right" } | null`, `notesInRect(x0: number, y0: number, x1: number, y1: number): number[]`.
  - In `render()`: scale note fill opacity/brightness subtly by `velocity` (`0.42 + 0.58 * (vel / 127)` multiplied by mute/highlight alpha), draw a crisp `#ffffff` 1.5px stroke around selected notes, and draw the translucent marquee box when active.
- In `web/src/components/PianoRollCanvas.tsx`:
  - Add a sleek compact editor bar directly above the `<canvas>` (visible once transcription finishes):
    - Mode toggle: `Scrub` / `Edit`
    - Active instrument selector (for double-click add)
    - When `selectedIds.size > 0`: selected count, Velocity slider (`1..127`), `-12` / `+12` octave buttons, `Delete` button
    - `Undo` / `Redo` buttons with disabled state
  - In `Edit` mode: click / `Shift+Click` note select, drag body to move time+pitch, drag right edge (last `6px` of bar) to resize duration, drag empty canvas for marquee select, double-click empty canvas to add a `0.25s` note on the active instrument; wire keyboard shortcuts (`Delete`/`Backspace`, `ArrowUp`/`ArrowDown` ±1 / `Shift` ±12, `Ctrl+Z` / `Ctrl+Shift+Z` / `Ctrl+Y`) when not focused in a form input.
- In `web/src/components/InstrumentList.tsx`:
  - Lift `muted: Set<string>` state to `App.tsx` (so export dialog can offer **"Exclude muted instruments"**!).
  - Display per-instrument note count badge and a compact **"Merge →"** `<select>` allowing one-click merging of any instrument track into another detected instrument (recorded in Undo/Redo history!).
- In `web/src/App.tsx`:
  - Maintain `editedNotes: EditableNote[]` + `EditorHistory` ref; every edit updates `editedNotes`, `rollRef.current.replaceNotes(...)`, `audio.replaceNotes(...)`, and `instruments` list in sync.

- [ ] **Step 5: Verify editor tests, TypeScript check, and Vite build**

Run:
```powershell
node --test web/src/editor.test.ts
cd web && npm run build
```
Expected: All editor tests PASS and `tsc --noEmit && vite build` succeeds with 0 errors.

- [ ] **Step 6: Commit**

```powershell
git add web/src/editor.ts web/src/editor.test.ts web/src/pianoroll.ts web/src/components/PianoRollCanvas.tsx web/src/components/InstrumentList.tsx web/src/App.tsx
git commit -m "feat(web): add interactive Piano Roll editor, velocity visualization, undo/redo, and instrument merging"
```

---

### Task 10: Frontend FL Studio Export Dialog, Production Build, Docs & Full Verification

**Files:**
- Modify: `web/src/export.ts`
- Create: `web/src/components/FlExportDialog.tsx`
- Modify: `web/src/components/MidiExportDialog.tsx`
- Modify: `web/src/components/OutputBar.tsx`
- Modify: `web/src/App.tsx`
- Modify: `README.md`

**Interfaces:**
- Consumes: `POST /export`, `POST /export/fl-bundle` (Task 6); `EditableNote`, `toExportPayload` (Task 9).
- Produces:
  - `web/src/export.ts`: extended `ExportNote` (with `velocity: number`), `StudioCleanupOptions` (`use_velocities`, `velocity_scale`, `min_note_duration_ms`, `min_velocity`, `deduplicate_ms`, `split_drums`, `excluded_instruments`), `exportFlBundle(...) -> Promise<Blob>`.
  - `FlExportDialog.tsx` wired as primary FL Studio export in `OutputBar.tsx`.

- [ ] **Step 1: Extend `web/src/export.ts` with velocity, cleanup options, and `exportFlBundle()`**

- Add `velocity?: number` to `ExportNote`.
- Add `StudioExportOptions` extending `NoteCleanup` with `use_velocities: boolean`, `velocity_scale: number`, `min_note_duration_ms: number`, `min_velocity: number`, `deduplicate_ms: number`, `split_drums: boolean`, `excluded_instruments: string[]`.
- Add `FL_PRESET_DEFAULTS`: `quantize: true`, `strength: 1`, `fill_gaps: 0.25`, `trim_leading_bars: false`, `use_velocities: true`, `velocity_scale: 1.0`, `min_note_duration_ms: 35`, `min_velocity: 18`, `deduplicate_ms: 25`, `split_drums: true`, `include_stems: true`, `normalize_drum_lanes_to_c5: true`, `exclude_muted: true`.
- Implement `exportFlBundle()` calling `POST /export/fl-bundle` using `toExportPayload(notes, beatGrid)` (so `onset_delay` is `0` for already-aligned edited notes!). Also update `exportMidi()` to use `toExportPayload` when notes are audio-aligned.

- [ ] **Step 2: Create `web/src/components/FlExportDialog.tsx` and wire into `OutputBar.tsx` & `App.tsx`**

- In `FlExportDialog.tsx`:
  - Remembers user's FL export settings in `localStorage`.
  - Sections:
    1. **Format:** `FL Studio Bundle (.zip)` vs `Single Multi-Track MIDI (.mid)`.
    2. **Timing & Grid:** Quantize toggle, Grid resolution, Strength slider, Fill gaps, Trim leading bars. If `followsBeats(beatGrid, timing)` is true AND `include_stems` is checked, show a clear inline warning with a 1-click **"Switch to fixed tempo"** button (`onSwitchToFixedTempo`) so WAV stems stay locked across the whole song in FL Studio!
    3. **Dynamics & Cleanup:** Use audio velocities checkbox + dynamic contrast slider (`0.5×–1.5×`), Min note duration (`0 / 25 / 35 / 50 ms`), Min velocity floor (`1–50`), Deduplicate window (`0 / 25 / 40 ms`).
    4. **FL Studio Layout:** Split drums into 6 lanes (`Kick / Snare / Hi-Hat / Toms / Cymbals / Perc`), Normalize single-hit drum lanes to C5 (FL Sampler), Exclude muted instruments (showing count of muted tracks), Include bar-1-aligned WAV stems (enabled when `result.stems.length > 0`).
- In `OutputBar.tsx`:
  - Show prominent **Export for FL Studio** button opening `FlExportDialog`, plus keep **Download ▾** dropdown for quick Single MIDI, Synthesized WAV, Original+MIDI Mix WAV, and Sheet Music (PDF/MusicXML).
  - Pass `editedNotes`, `mutedInstruments`, `timing`, and `onSwitchToFixedTempo` from `App.tsx`.

- [ ] **Step 3: Build production frontend bundle (`muscriptor/web_dist`) and update `README.md`**

- Update `README.md` with a concise **"FL Studio / Stem-First Studio"** section covering Windows CUDA setup (`.venv`), HuggingFace auth, stem separation (`demucs-infer`), Piano Roll editing, and FL Studio bundle import.
- Run: `cd web && npm run build`
Expected: Production bundle built cleanly into `muscriptor/web_dist/` with 0 TypeScript or Vite errors.

- [ ] **Step 4: Run complete verification suite (backend pytest + frontend tests + build + live server smoke test)**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
node --test web/src/editor.test.ts
.\.venv\Scripts\python.exe -c "from fastapi.testclient import TestClient; from pathlib import Path; from muscriptor.model_manager import ModelManager; from muscriptor.sessions import SessionStore; from muscriptor.server import create_app; app = create_app(model=None, web_dir=Path('muscriptor/web_dist'), model_manager=ModelManager(), session_store=SessionStore()); c = TestClient(app); print(c.get('/health').json()); print(c.get('/capabilities').json()); assert c.get('/').status_code == 200"
```
Expected:
- Every Python test in `tests/` PASSES (including all original unmodified tests).
- Every frontend editor test PASSES.
- Smoke test prints `{'status': 'ok'}` and capabilities JSON with `gpu_name` (RTX 4070), `separation_available: True`, and `/` returning HTTP `200` serving the built Studio UI.

- [ ] **Step 5: Commit**

```powershell
git add web/src/export.ts web/src/components/FlExportDialog.tsx web/src/components/MidiExportDialog.tsx web/src/components/OutputBar.tsx web/src/App.tsx README.md
git commit -m "feat(web): add FL Studio export dialog, production build, and studio documentation"
```

---

## Self-Review Checklist Verification
- **Spec coverage:** Every requirement in Sections 2, 3, and 5 of `docs/superpowers/specs/2026-11-25-stem-first-fl-studio-midi-design.md` maps directly to Tasks 1–10 (stems & decoupled masking → T4; velocity → T1, T3, T6, T8; cleanup & split drums → T2; FL bundle + C5 + aligned WAVs → T5, T10; lazy `ModelManager`, `SessionStore`, `/capabilities`, SSE stage/error → T6; CLI parity → T7; Web UI Studio panel, stem auditioning, Piano Roll editor, instrument merging, FL dialog → T8, T9, T10).
- **Type & interface consistency:** `Note.velocity`, `NoteEvent.midi_velocity`, `StageEvent`, `allowed_instruments`, `SeparationMode`, `StemPlan`, `DRUM_LANES`, `SINGLE_HIT_DRUM_LANES`, `C5_PITCH`, `ExportOptions`, `ModelManager`, `SessionStore`, `EditableNote`, `toExportPayload` match verbatim across every producer and consumer task.
- **Review Focus:** All 5 high-risk failure modes (silent stems, `onset_delay` double-shift on edited notes, missing HF auth SSE error + lock release, mono/48 kHz WAV alignment & negative-trim safety, all-filtered empty MIDI/ZIP) have explicit tests pinned in their owning tasks.
