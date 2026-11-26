"""Tests for SessionStore, ModelManager, and Studio server endpoints."""

import base64
import io
import json
import math
import zipfile
from unittest.mock import MagicMock

import mido
import torch
from fastapi.testclient import TestClient

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.model_manager import ModelManager
from muscriptor.server import create_app
from muscriptor.sessions import SessionStore
from muscriptor.tokenizer.notes import DRUM_PROGRAM
from muscriptor.utils.beats import BeatGrid
from muscriptor.utils.download import ModelDownloadError
from muscriptor.utils.fl_export import write_pcm16_wav_bytes


def _parse_sse(body: str) -> list[dict]:
    out = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert chunk.startswith("data: "), f"unexpected SSE chunk: {chunk!r}"
        out.append(json.loads(chunk[len("data: ") :]))
    return out


def _make_tone_wav(sr: int = 16000, duration_s: float = 1.0, freq: float = 440.0, amp: float = 0.5) -> bytes:
    n = int(sr * duration_s)
    t = torch.arange(n, dtype=torch.float32) / sr
    wav = (amp * torch.sin(2.0 * math.pi * freq * t)).unsqueeze(0)
    return write_pcm16_wav_bytes(wav, sr)


def _make_stub_model():
    """Minimal real-MIDI-producing model stub for Studio server tests."""
    from muscriptor.transcription_model import TranscriptionModel
    from muscriptor.tokenizer.mt3 import MT3Tokenizer

    model = MagicMock()
    model._tokenizer = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
    # Remove MagicMock auto-attribute so _program_for_instrument builds the real lookup table.
    del model._inst_to_program
    model._program_for_instrument = lambda inst: TranscriptionModel._program_for_instrument(model, inst)
    model.events_to_midi_bytes = lambda events, beat_grid=None, export=None, note_velocities=None: (
        TranscriptionModel.events_to_midi_bytes(
            model, events, beat_grid=beat_grid, export=export, note_velocities=note_velocities
        )
    )
    model.detect_beat_grid_for.return_value = BeatGrid(
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat=0.0,
        onset_delay=0.0,
        beat_subdivision=4,
        steady=True,
    )
    return model


def test_capabilities_reports_models_separation_and_hf_auth(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("muscriptor.model_manager._hf_saved_token", lambda: None)
    manager = ModelManager(default_model="medium", device="cpu")
    client = TestClient(create_app(model=None, model_manager=manager))

    resp = client.get("/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "device",
        "gpu_name",
        "vram_mb",
        "loaded_model",
        "default_model",
        "custom_model_locked",
        "available_models",
        "hf_auth_configured",
        "separation_available",
        "separation_modes",
    ):
        assert key in body
    assert body["loaded_model"] is None
    assert body["default_model"] == "medium"
    assert body["custom_model_locked"] is False
    assert body["available_models"] == ["small", "medium", "large"]
    assert body["hf_auth_configured"] is False


def test_custom_local_model_locks_selector():
    manager = ModelManager(default_model="/tmp/my_custom.safetensors", device="cpu")
    caps = manager.capabilities()
    assert caps["custom_model_locked"] is True


def test_studio_transcribe_runs_sequential_gpu_stems_and_streams_stages(tmp_path):
    order: list[str] = []
    stub_model = _make_stub_model()

    def _fake_transcribe(audio, instruments=None, allowed_instruments=None, **kwargs):
        order.append(f"transcribe:{instruments or allowed_instruments}")
        yield ProgressEvent(completed=0, total=1)
        if instruments == ["drums"]:
            s0 = NoteStartEvent(pitch=36, start_time=0.1, index=0, instrument="drums")
            yield s0
            yield NoteEndEvent(end_time=0.12, start_event=s0)
        elif allowed_instruments and "electric_bass" in allowed_instruments:
            s0 = NoteStartEvent(pitch=40, start_time=0.2, index=0, instrument="electric_bass")
            yield s0
            yield NoteEndEvent(end_time=0.6, start_event=s0)
        else:
            s0 = NoteStartEvent(pitch=69, start_time=0.3, index=0, instrument="acoustic_piano")
            yield s0
            yield NoteEndEvent(end_time=0.8, start_event=s0)
        yield ProgressEvent(completed=1, total=1)

    stub_model.transcribe.side_effect = _fake_transcribe

    class _FakeManager(ModelManager):
        def release_gpu(self) -> None:
            order.append("release_gpu")

        def get_model(self, model_size: str | None = None):
            order.append(f"get_model:{model_size}")
            return stub_model

    def _fake_separator(wav, sr, mode="stems_4", device=None, segment=7.5):
        order.append(f"separate:{mode}")
        out_sr = 44100
        t = torch.arange(out_sr, dtype=torch.float32) / out_sr
        active_drums = (0.6 * torch.sin(2.0 * math.pi * 70.0 * t)).repeat(2, 1)
        active_bass = (0.4 * torch.sin(2.0 * math.pi * 82.4 * t)).repeat(2, 1)
        active_other = (0.5 * torch.sin(2.0 * math.pi * 440.0 * t)).repeat(2, 1)
        silent_vocals = torch.zeros(2, out_sr, dtype=torch.float32)
        return {
            "drums": (active_drums, out_sr),
            "bass": (active_bass, out_sr),
            "vocals": (silent_vocals, out_sr),
            "other": (active_other, out_sr),
        }

    store = SessionStore(root_dir=tmp_path / "sessions", max_sessions=3)
    app = create_app(
        model=None,
        model_manager=_FakeManager(default_model="medium", device="cpu"),
        session_store=store,
        separator_fn=_fake_separator,
    )
    client = TestClient(app)

    resp = client.post(
        "/transcribe",
        files={"file": ("song.wav", _make_tone_wav(), "audio/wav")},
        data={"studio": "true", "separation": "stems_4", "model_size": "large"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    # Sequential GPU safety: release_gpu() MUST happen before separator runs!
    assert order.index("release_gpu") < order.index("separate:stems_4")
    assert order.index("separate:stems_4") < order.index("get_model:large")

    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert "separating" in stages
    assert "transcribing" in stages
    assert "beats" in stages
    assert "velocity" in stages

    # Silent vocals stem was skipped; drums, bass, other all emitted with unique indices!
    starts = [e for e in events if e["type"] == "start"]
    assert len(starts) == 3
    assert [s["index"] for s in starts] == [0, 1, 2]
    assert {s["instrument"] for s in starts} == {"drums", "electric_bass", "acoustic_piano"}

    complete = events[-1]
    assert complete["type"] == "transcription_complete"
    assert complete["session_id"] is not None
    assert set(complete["stems"]) == {"drums", "bass", "vocals", "other"}
    assert set(complete["velocities"].keys()) == {"0", "1", "2"}
    assert all(1 <= v <= 127 for v in complete["velocities"].values())

    # Stored stem WAV is streamable from /sessions/{id}/stems/{stem}.wav
    stem_resp = client.get(f"/sessions/{complete['session_id']}/stems/drums.wav")
    assert stem_resp.status_code == 200
    assert stem_resp.headers["content-type"].startswith("audio/wav")


def test_studio_transcribe_all_silent_stems_returns_valid_complete(tmp_path):
    stub_model = _make_stub_model()
    stub_model.transcribe.return_value = iter([])

    class _FakeManager(ModelManager):
        def get_model(self, model_size: str | None = None):
            return stub_model

    def _silent_sep(wav, sr, mode="stems_4", device=None, segment=7.5):
        z = torch.zeros(2, 44100, dtype=torch.float32)
        return {"drums": (z, 44100), "bass": (z, 44100), "vocals": (z, 44100), "other": (z, 44100)}

    app = create_app(
        model=None,
        model_manager=_FakeManager(default_model="medium", device="cpu"),
        session_store=SessionStore(root_dir=tmp_path / "sessions"),
        separator_fn=_silent_sep,
    )
    client = TestClient(app)
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _make_tone_wav(amp=0.0), "audio/wav")},
        data={"studio": "true", "separation": "stems_4"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    complete = events[-1]
    assert complete["type"] == "transcription_complete"
    assert complete["velocities"] == {}
    midi = mido.MidiFile(file=io.BytesIO(base64.b64decode(complete["data"])))
    assert isinstance(midi, mido.MidiFile)


def test_studio_transcribe_model_download_error_emits_sse_error_and_releases_lock(tmp_path):
    attempts = {"n": 0}
    stub_model = _make_stub_model()
    stub_model.transcribe.return_value = iter([])

    class _FlakyManager(ModelManager):
        def get_model(self, model_size: str | None = None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ModelDownloadError("HF login required — run hf auth login")
            return stub_model

    app = create_app(
        model=None,
        model_manager=_FlakyManager(default_model="medium", device="cpu"),
        session_store=SessionStore(root_dir=tmp_path / "sessions"),
    )
    client = TestClient(app)

    r1 = client.post(
        "/transcribe",
        files={"file": ("a.wav", _make_tone_wav(), "audio/wav")},
        data={"studio": "true", "separation": "none"},
    )
    assert r1.status_code == 200
    ev1 = _parse_sse(r1.text)
    assert ev1[-1]["type"] == "error"
    assert "HF login required" in ev1[-1]["message"]

    # Lock MUST have been released cleanly: next request succeeds without 503.
    r2 = client.post(
        "/transcribe",
        files={"file": ("a.wav", _make_tone_wav(), "audio/wav")},
        data={"studio": "true", "separation": "none"},
    )
    assert r2.status_code == 200
    ev2 = _parse_sse(r2.text)
    assert ev2[-1]["type"] == "transcription_complete"


def test_session_stem_endpoint_blocks_path_traversal_and_unknown_stems(tmp_path):
    store = SessionStore(root_dir=tmp_path / "sessions", max_sessions=3)
    sid = store.create_session({"drums": (torch.zeros(2, 1000), 44100)})
    client = TestClient(create_app(_make_stub_model(), session_store=store))

    assert client.get(f"/sessions/{sid}/stems/evil.wav").status_code in (400, 404)
    assert client.get("/sessions/not-a-uuid/stems/drums.wav").status_code in (400, 404)
    assert client.get(f"/sessions/{sid}/stems/drums.wav").status_code == 200


def test_session_store_lru_evicts_oldest_beyond_limit(tmp_path):
    store = SessionStore(root_dir=tmp_path / "sessions", max_sessions=3)
    ids = [
        store.create_session({"drums": (torch.zeros(2, 100), 44100)})
        for _ in range(4)
    ]
    assert store.get_stem_path(ids[0], "drums") is None
    assert not (tmp_path / "sessions" / ids[0]).exists()
    for sid in ids[1:]:
        assert store.get_stem_path(sid, "drums") is not None


def test_export_with_velocities_and_cleanup_options():
    client = TestClient(create_app(_make_stub_model()))
    resp = client.post(
        "/export",
        json={
            "notes": [
                {"instrument": "drums", "pitch": 36, "start": 0.0, "end": 0.05, "velocity": 118},
                {"instrument": "drums", "pitch": 38, "start": 0.5, "end": 0.55, "velocity": 30},
                {"instrument": "acoustic_piano", "pitch": 60, "start": 0.0, "end": 0.5, "velocity": 82},
            ],
            "beat_grid": {
                "bpm": 120.0,
                "beats_per_bar": 4,
                "first_downbeat": 0.0,
                "onset_delay": 0.0,
            },
            "options": {
                "min_velocity": 50,
                "split_drums": True,
                "use_velocities": True,
            },
        },
    )
    assert resp.status_code == 200
    midi = mido.MidiFile(file=io.BytesIO(resp.content))
    note_ons = [
        (m.note, m.velocity)
        for t in midi.tracks
        for m in t
        if m.type == "note_on" and m.velocity > 0
    ]
    # Velocity-30 snare dropped by min_velocity=50; kick (118) & piano (82) preserved!
    assert (38, 30) not in note_ons
    assert (36, 118) in note_ons
    assert (60, 82) in note_ons


def test_export_fl_bundle_returns_zip_with_aligned_stems(tmp_path):
    store = SessionStore(root_dir=tmp_path / "sessions", max_sessions=3)
    sid = store.create_session(
        {
            "drums": (torch.ones(2, 44100) * 0.2, 44100),
            "other": (torch.ones(2, 44100) * 0.2, 44100),
        }
    )
    client = TestClient(create_app(_make_stub_model(), session_store=store))
    resp = client.post(
        "/export/fl-bundle",
        json={
            "notes": [
                {"instrument": "drums", "pitch": 36, "start": 0.5, "end": 0.52, "velocity": 120},
                {"instrument": "acoustic_piano", "pitch": 60, "start": 0.5, "end": 1.0, "velocity": 88},
            ],
            "beat_grid": {
                "bpm": 120.0,
                "beats_per_bar": 4,
                "first_downbeat": 0.5,
                "onset_delay": 0.0,
            },
            "options": {"split_drums": True, "quantize": True},
            "session_id": sid,
            "include_stems": True,
            "normalize_drum_lanes_to_c5": True,
            "song_name": "demo_track",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
        names = set(zf.namelist())
        assert "00_full_mix.mid" in names
        assert "midi/drums_kick.mid" in names
        assert "midi/01_acoustic_piano.mid" in names
        assert "stems/drums.wav" in names
        assert "stems/other.wav" in names
        assert "README_FL_STUDIO.txt" in names
