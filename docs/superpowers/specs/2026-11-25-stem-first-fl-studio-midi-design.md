# Stem-First Studio for FL Studio MIDI — Design Document

**Date:** 2026-11-25  
**Target Repo:** `muscriptor` (`c:\Moje Složka\Vibe Coding\Audio to MIDI`)  
**Scope:** Full project enhancement (Approach A — Stem-First Studio)

---

## 1. Intent & Why

The user wants a personal, local, high-accuracy Audio-to-MIDI studio built on top of this repository (`muscriptor` Python backend + React web UI) to convert self-generated full-mix songs (e.g., AI-generated tracks or bounced stereo mixes) into **clean, expressive, production-ready MIDI** (plus optionally aligned WAV stems) for further editing, sound design, and arrangement in **FL Studio**.

Because this is strictly for personal, non-commercial use:
- MuScriptor's model weights (`CC BY-NC 4.0`) and MIT source code can be used directly without commercial licensing barriers.
- Features aimed at sheet-music SaaS platforms (accounts, billing, OMR photo scanning, natural-language chat editing) are unnecessary; every engineering dollar should go toward **transcription accuracy on dense mixes, realistic MIDI dynamics, fast manual note cleanup, and zero-friction FL Studio import**.

### Why Vanilla MuScriptor Falls Short Today
Inspection of the current codebase reveals three main gaps for an FL Studio workflow:
1. **Flat MIDI velocity (`100` everywhere):** `muscriptor/tokenizer/notes.py` and `muscriptor/utils/midi.py` hardcode `velocity = 100` for every note. Drums, piano, guitar, and bass lose all groove and dynamics.
2. **Full-mix bleed & crosstalk:** `TranscriptionModel.transcribe()` processes the entire 16 kHz mono mix in a single pass. In heavily compressed or reverberant full mixes, vocals bleed into guitar/synth tracks, kick masks bass, and dense drums obscure quiet chords.
3. **Limited DAW export & no note editing:** The web UI hardcodes one loaded model (default `medium`, `beam_size=1`), exports only a single combined `.mid` file, and its `PianoRollCanvas` is read-only (playback/scrubbing only — no deleting false notes, merging split instrument tracks, or exporting per-instrument / split-drum FL Studio files).

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
1. **Stem-First Transcription Pipeline:** Separate full mixes on GPU into clean stems (`4 Stems` via `htdemucs_ft` as default in the Studio UI; `6 Stems` via `htdemucs_6s`; or `Full Mix` fast single-pass) before running MuScriptor with targeted per-stem instrument constraints. Allow toggling individual stems on/off (e.g., skipping vocals).
2. **Audio-Derived Per-Note Velocity:** Estimate realistic MIDI velocity (`1–127`) for every note directly from its source stem (or full mix) waveform around the onset, preserving drum groove and melodic dynamics in preview playback, Piano Roll display, and MIDI export.
3. **Non-Destructive Artifact Cleanup:** Filter micro-duration ghost notes, quiet bleed notes below a velocity floor, and accidental double-triggers at export time without destroying the underlying transcribed notes.
4. **FL Studio–Ready Export Pack:**
   - Single combined Type 1 multi-track `.mid` (with real velocities, tempo/meter, optional split drum lanes), AND
   - One-click **FL Studio Bundle (`.zip`)** containing:
     - `00_full_mix.mid` (combined multi-track MIDI),
     - `midi/<NN>_<instrument>.mid` (individual single-track MIDI files ready to drag directly onto FL Studio Channel Rack instruments),
     - optional split drum-lane MIDI files (`Kick`, `Snare`, `Hi-Hat`, `Toms`, `Cymbals`, `Perc`), with an option to normalize single-hit drum-lane notes to **C5 (MIDI 60)** for FL Studio Sampler channels,
     - optional bar-1-aligned `stems/<stem>.wav` files synchronized to the exact same MIDI start offset,
     - `README_FL_STUDIO.txt` with detected BPM, meter, alignment offset, and import tips.
5. **Interactive Piano Roll Editor & Studio Controls (Web UI):**
   - Select separation mode, active stems, model size (`small` / `medium` / `large`), and decoding quality (`Standard` / `Beam Search`) from the Welcome screen.
   - Stream live multi-stage progress (Loading model → Separating → Transcribing per stem → Beat grid → Velocity estimation) and structured SSE error recovery.
   - Edit notes in the Piano Roll before export: select (click / Shift-click / marquee), delete false notes, move in time/pitch, resize duration, add notes, adjust velocity of selected notes, undo/redo (`Ctrl+Z` / `Ctrl+Shift+Z`).
   - Reassign/merge instrument tracks (e.g., merge stray `electric_piano` into `acoustic_piano`) and optionally exclude muted instruments from export.
   - Audition the original full mix **or** any individual separated stem against the velocity-aware browser synth.
6. **CLI Parity:** Expose stem separation, velocity estimation, cleanup options, and FL Studio bundle export in `muscriptor transcribe` for batch workflows.

### Non-Goals
- Commercial SaaS features (auth, user accounts, billing, cloud multi-tenancy).
- Optical Music Recognition (scanning PDFs/photos of sheet music).
- Natural-language AI chatbot score editor.
- In-browser sheet music engraving/editing (preserve existing MuseScore 4 `/sheets` export as-is).
- Forking or modifying C++ NeuralNote VST3.

### Anti-Goals (Failure Conditions)
- **GPU OOM on 8 GB VRAM:** Keeping stem separator + `muscriptor-large` (fp32) + `beat_this` loaded simultaneously on the user's 8 GB RTX 4070.
- **Stem / MIDI timing desynchronization:** Exporting stems and MIDI whose Bar 1 Beat 1 do not align when dropped onto FL Studio's Playlist.
- **Destructive defaults:** Automatically discarding ghost snares, quiet chords, or original unquantized timings without user control.
- **Regressions:** Breaking any existing unit tests (`tests/` must pass completely unmodified), `/transcribe/midi`, `/sheets`, `/auralize`, or existing CLI options.

---

## 3. Constraints & Environment

- **Hardware:** Windows 11/10, **NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)**, Python 3.13.5, Node v24.18.0 (`uv` and `pnpm` are not currently on PATH; setup must work via standard Python virtual environment + `pip` and Node `npm` or `corepack pnpm`, while keeping `pyproject.toml` and `web/package.json` compatible with `uv` / `pnpm`).
- **VRAM Budget (8 GB) & Server Model Lifecycle:**
  - `create_app(model)` keeps its exact existing signature and behavior for direct Python callers and existing `tests/test_server.py` (uses the passed pre-loaded `model` instance directly without lazy unloading).
  - For `muscriptor serve`, introduce a `ModelManager` (`muscriptor/model_manager.py`) passed via `create_app(model=None, model_manager=...)` with **lazy startup by default** (plus optional `--preload` flag): FastAPI boots immediately (`loaded_model: null`) without blocking on multi-GB HuggingFace downloads or crashing on missing HF auth.
  - If `muscriptor serve --model /local/path.safetensors` is launched with a custom local checkpoint, `/capabilities` reports `custom_model_locked: true` and the UI locks the model picker to that checkpoint; otherwise `small` / `medium` / `large` are switchable in the UI (defaulting to the CLI `--model` value, default `medium`).
  - Strict sequential GPU execution per transcription job:
    1. If stem separation is requested and MuScriptor is currently on CUDA, `ModelManager.release_gpu()` moves/unloads it off CUDA and calls `torch.cuda.empty_cache()`.
    2. Load stem separator on CUDA → separate track in chunks (e.g., 7–8s segments) → move resulting stem tensors to CPU RAM and save `44.1 kHz` WAV stems to the session temp directory → delete separator model & call `torch.cuda.empty_cache()`.
    3. Run `beat_this` beat/downbeat detection on the full mix → release GPU cache.
    4. Load/move the requested MuScriptor variant (`small`/`medium`/`large`) onto CUDA — when loading `large` (1.4B) on a CUDA device with `<= 12 GB` VRAM and `dtype` unspecified, automatically use `torch.bfloat16` (if `torch.cuda.is_bf16_supported()`, e.g., RTX 4070 Ada Lovelace) else `torch.float16` (~2.8 GB weights instead of ~5.6 GB fp32). Keep `model.condition_provider` in `float32` (already handled in `TranscriptionModel.load_model`).
    5. When switching `model_size` between requests, unload the previous model and empty CUDA cache before loading the new one.
  - Model downloads, lazy loading, and runtime failures (missing HF auth, CUDA OOM) happen **inside the SSE stream** (when `studio=true`), emitting `{type: "stage", stage: "loading_model", ...}` and `{type: "error", message: "..."}` so the Web UI displays live status and recovers cleanly to the Welcome screen with actionable instructions instead of timing out before HTTP 200 or showing a generic server-down error.
- **HuggingFace Gated Weights:** MuScriptor checkpoints require accepting CC BY-NC 4.0 on HuggingFace and local auth (`HF_TOKEN` or `hf auth login`). `/capabilities` reports `hf_auth_configured: bool` so the Welcome screen can proactively show setup instructions before the user even clicks Transcribe.
- **Optional Separator Dependency:** Stem separation uses `demucs-infer` (PyTorch 2.x inference-only Demucs package supporting `htdemucs_ft` and `htdemucs_6s`, with fallback to standard `demucs` if installed). If neither is installed, the app still starts and defaults cleanly to `Full Mix` mode with an install hint in `/capabilities` and UI.

---

## 4. Approaches Considered

| Approach | Summary | Trade-Offs | Verdict |
|---|---|---|---|
| **A. Stem-First Studio (Web/Python)** | Local GPU stem separation (`htdemucs_ft` / `htdemucs_6s`) + per-stem MuScriptor (`medium`/`large`) + audio-derived velocity + Piano Roll editor + FL Studio ZIP bundle (MIDIs + split drums + aligned WAV stems). | Longer per-song runtime (separation + multi-stem passes) and extra model disk space, in exchange for dramatically cleaner polyphonic transcription on dense mixes plus aligned WAV stems. | **Selected by user** |
| **B. Single-Pass Upgrade** | No stem separation; add velocity estimation, cleanup, `large` model selector, and FL export on top of single-pass MuScriptor. | Faster, fewer dependencies, but vocal/drum/guitar bleed remains on dense AI-generated mixes. | Rejected |
| **C. NeuralNote C++ VST3 Fork** | Fork `DamRsn/NeuralNote` (JUCE/C++) to run inside FL Studio directly. | In-DAW convenience, but heavy C++/Vulkan build on Windows, lacks `beat_this`/quantization, and stem separation + rich editing is far slower to iterate in C++. | Rejected |

---

## 5. Architecture & Technical Design

### 5.1 Core Data Model Extensions (Zero-Regression Wire Contract)

- **Existing unit tests remain 100% unmodified and passing.**
- **`muscriptor/events.py`:**
  - Keep `NoteStartEvent(pitch, start_time, index, instrument)` **completely unchanged** (no new dataclass fields), so `dataclasses.asdict(ev)` in `event_to_dict()` and `_event_to_dict()` preserves exact dict equality across `test_server.py`, `test_cli.py`, and `test_integration.py`.
  - Add `StageEvent(stage: str, message: str, stem: str | None = None, stem_index: int | None = None, stem_total: int | None = None)` used only on Studio SSE streams.
- **`muscriptor/tokenizer/notes.py`:**
  - Add `velocity: int = 100` to `Note` (clamped `1..127` on MIDI write).
  - Add `midi_velocity: int | None = None` to `NoteEvent` (keeping `NoteEvent.velocity` strictly `0` or `1` for tokenizer compatibility). `note2note_event(notes)` populates `midi_velocity=note.velocity`. In `note_event2midi(..., velocity=100)`, note-on messages use `ne.midi_velocity if ne.midi_velocity is not None else velocity`. Existing callers passing `NoteEvent(..., velocity=1)` continue writing MIDI velocity `100`.

### 5.2 Decoupled Logit Masking & Stem Separation (`muscriptor/transcription_model.py`, `muscriptor/utils/stems.py`)

- **Decoupling `ClassConditioner` from `forbidden_tokens` (`TranscriptionModel.transcribe`):**
  - Keep existing `instruments: list[str] | None = None` behavior **unchanged** (activates BOTH `ClassConditioner` prompt AND `forbidden_tokens` logit masking), preserving `tests/test_strict_instruments.py`.
  - Add an optional `allowed_instruments: list[str] | None = None` parameter to `TranscriptionModel.transcribe()`. When `instruments` is `None` and `allowed_instruments` is provided, `ClassConditioner` receives `instrument_group=None` (unconditional auto-detect, avoiding out-of-distribution 30-token prompts!), while `forbidden_tokens` masks out every program/drum outside `allowed_instruments`.
- **Separation Modes (`muscriptor/utils/stems.py`):**
  - `"none"`: Single-pass full mix (existing behavior).
  - `"stems_4"` (Studio UI default when separator available): `htdemucs_ft` → `["drums", "bass", "vocals", "other"]`. Chosen over `htdemucs_6s` as default because `htdemucs_ft` produces substantially cleaner primary stems with fewer phase artifacts, while MuScriptor natively separates piano/guitar/synths/strings inside the clean `"other"` stem.
  - `"stems_6"`: `htdemucs_6s` → `["drums", "bass", "vocals", "guitar", "piano", "other"]` for users who explicitly want isolated Guitar/Piano stems.
- **Per-Stem Conditioning vs. Masking Rules:**
  - Allowed instrument groups per stem:
    - `drums` → `["drums"]` (1 group)
    - `vocals` → `["voice"]` (1 group)
    - `bass` → `["electric_bass", "acoustic_bass", "contrabass"]`
    - `guitar` (`stems_6` only) → `["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"]`
    - `piano` (`stems_6` only) → `["acoustic_piano", "electric_piano"]`
    - `other` → all groups in `MT3_FULL_PLUS_GROUP_NAMES` **except** `drums`, `voice`, and the stems already split out in that mode (`bass` in 4-stem; `bass` + `guitar` + `piano` in 6-stem).
  - **When user leaves `instruments` empty (Auto):**
    - Single-group stems (`drums`, `vocals`) pass `instruments=allowed` (both `ClassConditioner` and mask).
    - Multi-group stems (`bass`, `guitar`, `piano`, `other`) pass `instruments=None, allowed_instruments=allowed` (unconditional auto-detect + strict logit mask — no out-of-distribution conditioning prompt, zero drum/vocal bleed!).
  - **When user explicitly picks `instruments`:**
    - Intersect `allowed ∩ user_instruments` per stem; skip any stem with empty intersection; pass that small explicit intersection as `instruments=intersection` (matching user's explicit conditioning request).
- **Silent Stem Skipping:**
  - Before transcribing a stem, compute peak/RMS energy; if peak `< -40 dBFS` (or active 50 ms frame ratio `< 0.5%` above `-45 dBFS`), skip transcribing that stem and emit a stage notice (e.g., skipping silent `vocals` on an instrumental track).
- **Resampling & Time Alignment:**
  - Separation runs at native stereo `44.1 kHz`; stereo stems are saved to the session temp directory for auditioning and FL Bundle export.
  - Each stem is downmixed to `16 kHz` mono via `muscriptor.utils.audio.resample` for MuScriptor inference.
  - All stems preserve the exact input sample duration `T`, guaranteeing zero inter-stem drift.

### 5.3 Audio-Derived Per-Note Velocity (`muscriptor/utils/velocity.py`)

- **Input:** List of `Note` objects (paired with their source stem name or `"mix"`) + dict of `{stem_name: (waveform, sr)}` (using native `44.1 kHz` when available for crisp hi-hat/cymbal transient bands, falling back to `16 kHz`).
- **Algorithm:**
  - **Melodic / Harmonic Notes:**
    - Compute one shared STFT magnitude spectrogram per stem (`hop_length ≈ 10 ms`, `n_fft = 2048` at 16 kHz or `4096` at 44.1 kHz, Hann window).
    - For each note `(pitch, onset, offset)`:
      - Inspect attack/early-sustain frames `[onset - 10ms, min(offset, onset + 80ms)]` (at least 1 frame).
      - Sum weighted spectral energy in ±0.6 semitone bins around `f0` (`1.0`), `2*f0` (`0.5`), and `3*f0` (`0.25`), where `f0 = 440 * 2^((pitch - 69)/12)` (clamped below Nyquist).
      - Take the 80th-percentile frame energy in that window and convert to dB (`20 * log10(E + 1e-8)`).
  - **Drum Notes:**
    - Compute a transient STFT (`hop_length ≈ 5 ms`, `n_fft = 512` at 16 kHz or `1024` at 44.1 kHz) over `[onset - 5ms, onset + 35ms]`.
    - Measure peak band energy in the drum lane's frequency band:
      - Kick (`35, 36`): `35–140 Hz`
      - Snare / Rim / Clap (`37, 38, 39, 40`): `150–2500 Hz`
      - Toms (`41, 43, 45, 47, 48, 50`): `80–450 Hz`
      - Hi-Hat (`42, 44, 46`): `4000–14000 Hz` (clamped to `0.95 * Nyquist`)
      - Cymbals (`49, 51, 52, 53, 55, 57, 59`): `3000–14000 Hz` (clamped to `0.95 * Nyquist`)
      - Other Percussion: `200–6000 Hz`
  - **Normalization to MIDI `1..127`:**
    - Normalize per instrument track (and per drum lane for drums) using robust percentiles (`p5` → low anchor, `p95` → high anchor, enforcing a minimum dynamic span of `24 dB` so uniform hits never amplify noise into artificial swings).
    - Map linearly into `[vel_min, vel_max]` (default `[32, 122]`, clamped `[1, 127]`).
    - Tracks with `< 3` notes anchor against overall stem RMS so isolated hits land naturally (`~75–110`).
- **Performance:** Vectorized PyTorch/NumPy; `< 0.3s` for a 4-minute song.

### 5.4 Post-Processing Cleanup & Drum Lane Splitting (`muscriptor/utils/cleanup.py` & `muscriptor/utils/midi.py`)

- **Cleanup Pass (`cleanup_notes`, applied non-destructively at export time):**
  1. **Instrument Exclusion & Remapping:** Drop instruments in `excluded_instruments`, apply `instrument_remap` (e.g., merging two guitar/piano tracks into one).
  2. **Velocity Floor:** Drop notes with `velocity < min_velocity` (default `1` = off for raw export; FL Studio UI preset defaults to `18`, user-adjustable).
  3. **Minimum Duration Filter:** Drop non-drum notes shorter than `min_note_duration_ms` (default `0 ms` for raw compatibility; FL Studio UI preset defaults to `35 ms`).
  4. **Same-Pitch Deduplication:** Within each `(instrument, pitch)`, if two notes start within `deduplicate_ms` (default `0 ms` for raw compatibility; FL Studio UI preset defaults to `25 ms`), merge/keep the higher-velocity note (extending `offset` to `max(offset)` for melodic notes).
  5. **Velocity Scaling / Flattening:** If `use_velocities=False`, write `100` for all notes; otherwise apply dynamic scaling around midpoint (`velocity_scale`, default `1.0`).
- **Drum Lane Grouping (`DRUM_LANES`):**
  - GM percussion pitches (channel 9) map deterministically into 6 FL Studio–friendly lanes:
    - `kick`: `{35, 36}`
    - `snare`: `{37, 38, 39, 40}`
    - `hihat`: `{42, 44, 46}`
    - `toms`: `{41, 43, 45, 47, 48, 50}`
    - `cymbals`: `{49, 51, 52, 53, 55, 57, 59}`
    - `perc`: all remaining drum pitches.
  - When `split_drums=True`, each non-empty drum lane becomes its own named track (`Drums - Kick`, `Drums - Snare`, etc.) in the combined MIDI (keeping GM pitches on channel 9), AND gets its own individual `.mid` file in the FL Bundle.
  - When `normalize_drum_lanes_to_c5=True` (default **on** for individual drum-lane `.mid` files in the FL Bundle; never applied to combined `00_full_mix.mid`), single-piece lanes (`kick`, `snare`, `hihat`) remap note pitches to **MIDI 60 (C5)** so dragging `midi/drums_kick.mid` onto an FL Studio Sampler channel plays the sample at its natural root pitch. Multi-pitch lanes (`toms`, `cymbals`, `perc`) retain GM pitches so tom fills and crash/ride stay distinct.

### 5.5 FL Studio Bundle Export & Bar-1 Stem Alignment (`muscriptor/utils/fl_export.py`)

- **Contents of `<song>_fl_studio.zip`:**
  - `00_full_mix.mid` — Combined Type 1 multi-track MIDI (with chosen timing/quantization/cleanup/split_drums).
  - `midi/01_<instrument>.mid`, `midi/02_<instrument>.mid`, … — Individual single-track `.mid` files sharing the **exact same conductor tempo, time signature, and bar-1 offset** as `00_full_mix.mid`, ready to drop straight onto individual FL Studio Channel Rack VSTs/Samplers.
  - `stems/<stem>.wav` (optional, included when stems exist and "Include WAV stems" is enabled) — 16-bit PCM stereo WAVs aligned to MIDI tick 0 via `offset_s = -timeline.seconds_at(0.0)`:
    - If `offset_s > 0`: prepend `round(offset_s * sr)` frames of silence so audio `t=0` plays at the exact same beat position as the MIDI.
    - If `offset_s < 0` (e.g., `trim_leading_bars=True`): trim the first `round(-offset_s * sr)` frames.
    - Note in `README_FL_STUDIO.txt` (and UI warning) when `steady=False` (variable tempo map): fixed tempo guarantees sample-exact alignment across the entire track; variable tempo aligns bar 1 beat 1 and requires matching tempo automation or stretching in FL Studio.
  - `README_FL_STUDIO.txt` — Detected BPM, time signature, bar-1 alignment shift, list of included tracks/stems, and concise FL Studio import steps.

### 5.6 Session Store & Backend API (`muscriptor/sessions.py`, `muscriptor/server.py`)

- **Session Store (`muscriptor/sessions.py`):**
  - Stores per-transcription separated `44.1 kHz` WAV stems under a temporary directory (`tempfile.gettempdir() / "muscriptor_sessions" / <session_id>`).
  - Bounded LRU eviction (keep at most `3` recent sessions; remove evicted/stale directories cleanly).
- **Endpoints:**
  - `GET /capabilities` → `{device, gpu_name, vram_mb, loaded_model, default_model, custom_model_locked, available_models, hf_auth_configured, separation_available, separation_modes}`.
  - `POST /transcribe` (strictly backward-compatible):
    - Existing fields: `file`, `instruments`, `detect_tempo`.
    - New optional fields: `studio: bool = False`, `separation: Literal["none", "stems_4", "stems_6"] = "none"`, `stems: list[str] = []`, `model_size: str | None = None`, `beam_size: int = 1`.
    - **Legacy mode (`studio=False`, default at API layer):** Exact existing SSE event sequence (`start`, `end`, `progress`, `transcription_complete: {type, data, beat_grid}`), keeping all existing `test_server.py` assertions intact without modification.
    - **Studio mode (`studio=True`, sent by Web UI):**
      1. Emits `stage` events (`loading_model`, `separating`, `transcribing` per stem with `stem`/`stem_index`/`stem_total`, `beats`, `velocity`).
      2. Catches model download / HF auth / separation / CUDA OOM exceptions inside the generator and emits `{type: "error", message: "..."}` cleanly instead of dropping the connection silently.
      3. Runs stem separation (if requested), transcribes active stems with globally unique note `index` values, detects global `beat_grid` on the full mix, computes global `onset_delay`, estimates per-note audio velocities, persists stems in `SessionStore`, and emits `transcription_complete: {type, data, beat_grid, session_id, stems, velocities}` where `velocities` maps `str(note_index) -> int (1..127)`.
  - `GET /sessions/{session_id}/stems/{stem_name}.wav` → streams stored separated stem WAV for browser auditioning (validated strictly against UUID session ID + stem allowlist — no path traversal).
  - `POST /export` → accepts `notes` (each with optional `velocity: int = 100`) + extended `ExportSettings` (`use_velocities`, `velocity_scale`, `min_note_duration_ms`, `min_velocity`, `deduplicate_ms`, `split_drums`, `excluded_instruments`, `instrument_remap`).
  - `POST /export/fl-bundle` → same payload + `session_id: str | None`, `include_stems: bool`, `normalize_drum_lanes_to_c5: bool`, returns `application/zip` (`<stem>_fl_studio.zip`).

### 5.7 Web UI Enhancements (`web/src/`)

1. **Welcome Screen (`WelcomeScreen.tsx` + `StudioSettingsPanel.tsx`):**
   - Queries `/capabilities` on mount (next to `/health`). Shows a clear non-blocking banner if `hf_auth_configured === false` with exact HuggingFace setup steps.
   - Compact **Studio Setup** card:
     - **Separation Mode:** `4 Stems (Recommended)` (default when available) / `6 Stems` / `Full Mix (Fast)`.
     - **Active Stems:** toggle pills (`Drums`, `Bass`, `Vocals`, `Other` [+ `Guitar`, `Piano` in 6-stem mode]); all enabled by default with 1-click mute for `Vocals`.
     - **Model Size:** `small (103M)` / `medium (307M)` / `large (1.4B — Best)` (disabled/locked if `custom_model_locked`).
     - **Decoding:** `Standard (Fast)` / `Beam Search (2)`.
2. **Stage Progress & Error Handling (`sse.ts`, `useTranscription.ts`, `App.tsx`, `OutputBar.tsx`):**
   - Handle `stage` and `error` SSE events cleanly.
   - Display current stage text alongside the progress bar (`"Loading muscriptor-large…"`, `"Separating 4 stems…"`, `"Transcribing drums (1/4)…"`, `"Estimating note velocities…"`).
3. **Interactive Piano Roll Editor (`pianoroll.ts`, `PianoRollCanvas.tsx`, `App.tsx`):**
   - Each note stores a stable `id: number` and `velocity: number` (default `100`, populated from `transcription_complete.velocities`).
   - Visual velocity: modulate note fill brightness/opacity based on velocity and show velocity value for hovered/selected notes.
   - **Mode toggle (`Scrub` vs `Edit`):**
     - In `Edit` mode: click / `Shift+Click` / marquee box select; drag selected note(s) in time/pitch; drag right edge to resize duration; `Delete`/`Backspace` to delete; `ArrowUp`/`ArrowDown` (±1 semitone, `Shift` ±12); double-click empty grid to add a note on the active instrument; selected-note velocity slider; `Ctrl+Z` / `Ctrl+Shift+Z` (or `Ctrl+Y`) Undo/Redo (50 snapshots).
   - Keep `AudioEngine` (`web/src/audio.ts`) synced with edits and velocities:
     - Send actual note `velocity` to `spessasynth_lib` (`noteOn(channel, pitch, velocity)`) instead of hardcoded `100`.
     - Add reference audio source selector in `Controls.tsx` (`Original Mix` / `Drums` / `Bass` / `Vocals` / `Other`…) when separated stems exist, loading `/sessions/{id}/stems/{stem}.wav` on demand.
4. **Instrument List (`InstrumentList.tsx`):**
   - Show note count per instrument.
   - Add a **"Merge into…"** action on each instrument row so stray detections can be consolidated before export.
5. **FL Studio Export Dialog (`FlExportDialog.tsx` + `OutputBar.tsx`):**
   - Primary export action for **FL Studio Bundle (.zip)** alongside **MIDI (.mid)** and existing WAV / Sheet Music downloads.
   - Controls: Timing & Quantization (with fixed-tempo recommendation notice when exporting WAV stems), Dynamics & Cleanup (real velocity toggle + scale, min duration ms, min velocity floor, deduplicate ms), FL Studio Layout (split drums into 6 lanes, normalize single-hit drum lanes to C5, exclude muted instruments, include bar-aligned WAV stems).

### 5.8 CLI Parity (`muscriptor/main.py`)

Extend `muscriptor transcribe` with backward-compatible optional flags:
- `--separation [none|stems_4|stems_6]` (default `none`).
- `--stems` (comma-separated subset, e.g. `drums,bass,other`).
- `--estimate-velocity/--no-estimate-velocity` (defaults to `False` for legacy `--format midi` with `--separation none` so existing CLI tests remain deterministic; automatically defaults to `True` whenever `--separation != none` or `--format fl-bundle`).
- `--min-note-ms`, `--min-velocity`, `--deduplicate-ms`, `--split-drums`, `--c5-drums`.
- `--format fl-bundle` writing `<audio_file>_fl_studio.zip` (or `--output path.zip`) with `--include-stems/--no-include-stems`.

---

## 6. Explicit Calls Made by the Architect (User Can Override)

1. **Default stem model is `htdemucs_ft` (4 stems: Drums, Bass, Vocals, Other)** rather than `htdemucs_6s` (6 stems), with `htdemucs_6s` available as a toggle — because `htdemucs_ft` yields measurably cleaner separation and MuScriptor already distinguishes piano/guitar/synths inside `"other"`.
2. **Decoupled logit masking (`allowed_instruments`) for multi-instrument stems (`other`, `bass`, `guitar`, `piano`)** so `ClassConditioner` isn't fed out-of-distribution ~30-instrument prompts, while single-group stems (`drums`, `vocals`) use full conditioning + masking.
3. **Lazy server startup (`ModelManager`) + automatic `bfloat16`/`float16` weight loading for `muscriptor-large` on `<= 12 GB` CUDA GPUs**, plus strict sequential GPU offloading (offload MuScriptor → run Demucs → unload Demucs → load MuScriptor) and structured SSE `stage`/`error` events, so the 8 GB RTX 4070 never hits OOM and missing HuggingFace auth displays cleanly in the UI.
4. **Zero-regression wire contract:** Existing `NoteStartEvent`, `create_app(model)`, legacy `POST /transcribe`, and CLI defaults remain backward-compatible so every existing test in `tests/` passes unmodified; Studio SSE features activate via `studio=true` sent by the Web UI.
5. **Individual single-hit drum-lane MIDI files (`Kick`, `Snare`, `Hi-Hat`) in the FL Bundle default to C5 (MIDI 60)** for instant FL Studio Sampler compatibility, while combined `00_full_mix.mid` preserves standard GM drum pitches on channel 9.
6. **Local setup without `uv`/`pnpm`:** Use a project `.venv` (Python 3.13 + PyTorch CUDA) and `npm`/`corepack pnpm` for the frontend build while keeping `pyproject.toml` and `package.json` cleanly maintained.

---

## 7. Left to the Builder

- Exact DSP window tuning constants, spectral bin weights, and unit test tolerances for velocity estimation (verified with synthetic louder/softer sine and drum transient fixtures).
- Exact Tailwind/CSS classes and icon SVGs matching the existing dark theme (`web/src/style.css`).
- Full GM percussion note map (35–81) assignment into the 6 drum lanes.
- Test strategy: fast deterministic unit tests with synthetic tensors and stubbed separator/model backends (zero network/weight downloads during `pytest`), plus TypeScript typecheck (`tsc --noEmit`) and Vite production build verification.
- Handling edge cases when a user switches `model_size` while another request holds the lock (returns 503 like existing busy handling).
