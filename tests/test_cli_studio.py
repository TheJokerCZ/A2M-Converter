"""Tests for CLI Studio parity (--format fl-bundle, --separation, lazy serve)."""

import io
import math
import zipfile
from unittest.mock import MagicMock

import mido
import torch
from typer.testing import CliRunner

import muscriptor.main as main_mod
from muscriptor.events import NoteEndEvent, NoteStartEvent
from muscriptor.model_manager import ModelManager
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.beats import BeatGrid
from muscriptor.utils.fl_export import write_pcm16_wav_bytes


def _write_tone_wav(path, sr: int = 16000, duration_s: float = 1.0):
    n = int(sr * duration_s)
    t = torch.arange(n, dtype=torch.float32) / sr
    wav = (0.5 * torch.sin(2.0 * math.pi * 440.0 * t)).unsqueeze(0)
    path.write_bytes(write_pcm16_wav_bytes(wav, sr))
    return path


def _make_stub_model():
    model = MagicMock()
    model._tokenizer = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
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


def test_cli_fl_bundle_format_writes_zip(monkeypatch, tmp_path):
    audio_path = _write_tone_wav(tmp_path / "track.wav")
    out_zip = tmp_path / "out.zip"
    model = _make_stub_model()

    def _fake_transcribe(audio, **kwargs):
        s0 = NoteStartEvent(pitch=69, start_time=0.1, index=0, instrument="acoustic_piano")
        s1 = NoteStartEvent(pitch=36, start_time=0.2, index=1, instrument="drums")
        yield s0
        yield NoteEndEvent(end_time=0.6, start_event=s0)
        yield s1
        yield NoteEndEvent(end_time=0.22, start_event=s1)

    model.transcribe.side_effect = _fake_transcribe
    monkeypatch.setattr(main_mod, "_load_model", lambda *a, **kw: model)

    runner = CliRunner()
    res = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(audio_path),
            "--format",
            "fl-bundle",
            "--split-drums",
            "-o",
            str(out_zip),
        ],
    )
    assert res.exit_code == 0, res.stderr + res.stdout
    assert out_zip.is_file()
    with zipfile.ZipFile(out_zip, "r") as zf:
        names = set(zf.namelist())
        assert "00_full_mix.mid" in names
        assert "midi/01_acoustic_piano.mid" in names
        assert "midi/drums_kick.mid" in names
        assert "README_FL_STUDIO.txt" in names


def test_cli_separation_stems_4_transcribes_active_stems_with_velocity(monkeypatch, tmp_path):
    audio_path = _write_tone_wav(tmp_path / "song.wav")
    out_zip = tmp_path / "song_fl.zip"
    model = _make_stub_model()
    seen_stems: list[str] = []

    def _fake_sep(wav, sr, mode="stems_4", device=None, segment=7.5):
        out_sr = 44100
        t = torch.arange(out_sr, dtype=torch.float32) / out_sr
        drums = (0.7 * torch.sin(2.0 * math.pi * 70.0 * t)).repeat(2, 1)
        bass = (0.4 * torch.sin(2.0 * math.pi * 82.4 * t)).repeat(2, 1)
        vocals = torch.zeros(2, out_sr)
        other = (0.5 * torch.sin(2.0 * math.pi * 440.0 * t)).repeat(2, 1)
        return {
            "drums": (drums, out_sr),
            "bass": (bass, out_sr),
            "vocals": (vocals, out_sr),
            "other": (other, out_sr),
        }

    def _fake_transcribe(audio, instruments=None, allowed_instruments=None, **kwargs):
        if instruments == ["drums"]:
            seen_stems.append("drums")
            s0 = NoteStartEvent(pitch=36, start_time=0.1, index=0, instrument="drums")
            yield s0
            yield NoteEndEvent(end_time=0.12, start_event=s0)
        elif allowed_instruments and "electric_bass" in allowed_instruments:
            seen_stems.append("bass")
            s0 = NoteStartEvent(pitch=40, start_time=0.2, index=0, instrument="electric_bass")
            yield s0
            yield NoteEndEvent(end_time=0.6, start_event=s0)

    model.transcribe.side_effect = _fake_transcribe
    monkeypatch.setattr(main_mod, "_load_model", lambda *a, **kw: model)
    monkeypatch.setattr(main_mod, "separate_stems", _fake_sep)

    runner = CliRunner()
    res = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(audio_path),
            "--separation",
            "stems_4",
            "--stems",
            "drums,bass",
            "--format",
            "fl-bundle",
            "--split-drums",
            "-o",
            str(out_zip),
        ],
    )
    assert res.exit_code == 0, res.stderr + res.stdout
    assert seen_stems == ["drums", "bass"]
    with zipfile.ZipFile(out_zip, "r") as zf:
        names = set(zf.namelist())
        assert "stems/drums.wav" in names
        assert "stems/bass.wav" in names
        mid = mido.MidiFile(file=io.BytesIO(zf.read("00_full_mix.mid")))
        ons = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
        assert len(ons) == 2


def test_cli_serve_starts_lazily_without_loading_model(monkeypatch):
    loaded = {"called": False}
    captured = {}

    def _fail_load(*a, **kw):
        loaded["called"] = True
        raise AssertionError("serve should not eagerly load model by default")

    def _fake_create_app(model=None, web_dir=None, **kwargs):
        captured["model"] = model
        captured["model_manager"] = kwargs.get("model_manager")
        return object()

    def _fake_uvicorn_run(app, host, port):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(TranscriptionModel, "load_model", classmethod(lambda cls, *a, **kw: _fail_load()))
    import muscriptor.server as server_mod
    import uvicorn

    monkeypatch.setattr(server_mod, "create_app", _fake_create_app)
    monkeypatch.setattr(uvicorn, "run", _fake_uvicorn_run)

    runner = CliRunner()
    res = runner.invoke(main_mod.app, ["serve", "--port", "9123"])
    assert res.exit_code == 0, res.stderr + res.stdout
    assert loaded["called"] is False
    assert captured["model"] is None
    assert isinstance(captured["model_manager"], ModelManager)
    assert captured["port"] == 9123


def test_cli_separation_with_json_does_not_double_transcribe(monkeypatch, tmp_path):
    import json

    audio_path = _write_tone_wav(tmp_path / "song.wav")
    out_json = tmp_path / "out.json"
    model = _make_stub_model()
    load_calls = {"n": 0}

    def _fake_sep(wav, sr, mode="stems_4", device=None, segment=7.5):
        out_sr = 44100
        t = torch.arange(out_sr, dtype=torch.float32) / out_sr
        drums = (0.7 * torch.sin(2.0 * math.pi * 70.0 * t)).repeat(2, 1)
        return {"drums": (drums, out_sr)}

    def _fake_transcribe(audio, **kwargs):
        s0 = NoteStartEvent(pitch=36, start_time=0.1, index=0, instrument="drums")
        yield s0
        yield NoteEndEvent(end_time=0.12, start_event=s0)

    def _fake_load(*a, **kw):
        load_calls["n"] += 1
        return model

    model.transcribe.side_effect = _fake_transcribe
    monkeypatch.setattr(main_mod, "_load_model", _fake_load)
    monkeypatch.setattr(main_mod, "separate_stems", _fake_sep)

    runner = CliRunner()
    res = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(audio_path),
            "--separation",
            "stems_4",
            "--stems",
            "drums",
            "-f",
            "json",
            "-o",
            str(out_json),
        ],
    )
    assert res.exit_code == 0, res.stderr + res.stdout
    assert load_calls["n"] == 1
    assert model.transcribe.call_count == 1
    data = json.loads(out_json.read_text())
    assert len(data) == 2
    assert data[0]["instrument"] == "drums"


def test_cli_estimate_velocity_midi_runs_auralize(monkeypatch, tmp_path):
    audio_path = _write_tone_wav(tmp_path / "song.wav")
    out_mid = tmp_path / "out.mid"
    out_aur = tmp_path / "mix.wav"
    model = _make_stub_model()
    auralized = {"called": False}

    def _fake_transcribe(audio, **kwargs):
        s0 = NoteStartEvent(pitch=69, start_time=0.1, index=0, instrument="acoustic_piano")
        yield s0
        yield NoteEndEvent(end_time=0.5, start_event=s0)

    model.transcribe.side_effect = _fake_transcribe
    monkeypatch.setattr(main_mod, "_load_model", lambda *a, **kw: model)

    import muscriptor.utils.auralization as aur_mod

    monkeypatch.setattr(
        aur_mod,
        "auralize",
        lambda **kw: auralized.__setitem__("called", True),
    )

    runner = CliRunner()
    res = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(audio_path),
            "--estimate-velocity",
            "-f",
            "midi",
            "-o",
            str(out_mid),
            "--auralize",
            str(out_aur),
        ],
    )
    assert res.exit_code == 0, res.stderr + res.stdout
    assert out_mid.is_file()
    assert auralized["called"] is True


def test_cli_missing_demucs_exits_cleanly(monkeypatch, tmp_path):
    from muscriptor.utils.stems import StemSeparationUnavailableError

    audio_path = _write_tone_wav(tmp_path / "song.wav")

    def _missing_sep(*a, **kw):
        raise StemSeparationUnavailableError("Install demucs-infer")

    monkeypatch.setattr(main_mod, "separate_stems", _missing_sep)

    runner = CliRunner()
    res = runner.invoke(
        main_mod.app,
        ["transcribe", str(audio_path), "--separation", "stems_4"],
    )
    assert res.exit_code == 1
    assert "Error:" in res.stderr
    assert "demucs-infer" in res.stderr
    assert res.exception is None or isinstance(res.exception, SystemExit)
