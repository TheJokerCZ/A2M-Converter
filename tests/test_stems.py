"""Tests for muscriptor/utils/stems.py and low-VRAM large auto-dtype."""

import math
from types import SimpleNamespace

import pytest
import torch

from muscriptor.transcription_model import _CONFIGS, _auto_dtype
from muscriptor.utils.stems import (
    STEM_NAMES_BY_MODE,
    StemPlan,
    is_stem_silent,
    plan_stem_conditioning,
    separate_stems,
)


def test_plan_stem_conditioning_auto_4_stems():
    plans = plan_stem_conditioning("stems_4")
    by_stem = {p.stem: p for p in plans}
    assert list(by_stem.keys()) == list(STEM_NAMES_BY_MODE["stems_4"])

    # Single-group stems use full ClassConditioner + mask (`instruments`).
    assert by_stem["drums"] == StemPlan(stem="drums", instruments=["drums"], allowed_instruments=None)
    assert by_stem["vocals"] == StemPlan(stem="vocals", instruments=["voice"], allowed_instruments=None)

    # Multi-group stems use unconditional ClassConditioner (`instruments=None`) + logit mask.
    assert by_stem["bass"].instruments is None
    assert by_stem["bass"].allowed_instruments == [
        "electric_bass",
        "acoustic_bass",
        "contrabass",
    ]
    assert by_stem["other"].instruments is None
    other_allowed = set(by_stem["other"].allowed_instruments or [])
    assert "acoustic_piano" in other_allowed
    assert "clean_electric_guitar" in other_allowed
    assert "synth_lead" in other_allowed
    for excluded in ("drums", "voice", "electric_bass", "acoustic_bass", "contrabass"):
        assert excluded not in other_allowed


def test_plan_stem_conditioning_auto_6_stems():
    plans = plan_stem_conditioning("stems_6")
    by_stem = {p.stem: p for p in plans}
    assert list(by_stem.keys()) == list(STEM_NAMES_BY_MODE["stems_6"])
    assert by_stem["guitar"].instruments is None
    assert by_stem["guitar"].allowed_instruments == [
        "acoustic_guitar",
        "clean_electric_guitar",
        "distorted_electric_guitar",
    ]
    assert by_stem["piano"].instruments is None
    assert by_stem["piano"].allowed_instruments == ["acoustic_piano", "electric_piano"]
    other_allowed = set(by_stem["other"].allowed_instruments or [])
    for excluded in (
        "drums",
        "voice",
        "electric_bass",
        "acoustic_bass",
        "contrabass",
        "acoustic_guitar",
        "clean_electric_guitar",
        "distorted_electric_guitar",
        "acoustic_piano",
        "electric_piano",
    ):
        assert excluded not in other_allowed
    assert "synth_lead" in other_allowed


def test_plan_stem_conditioning_with_explicit_user_instruments():
    plans = plan_stem_conditioning(
        "stems_4",
        user_instruments=["drums", "acoustic_piano"],
    )
    assert plans == [
        StemPlan(stem="drums", instruments=["drums"], allowed_instruments=None),
        StemPlan(stem="other", instruments=["acoustic_piano"], allowed_instruments=None),
    ]


def test_plan_stem_conditioning_respects_active_stems_subset():
    plans = plan_stem_conditioning("stems_4", active_stems=["drums", "other"])
    assert [p.stem for p in plans] == ["drums", "other"]


def test_is_stem_silent_detects_silence_and_activity():
    sr = 16000
    silent = torch.zeros(1, sr * 2, dtype=torch.float32)
    assert is_stem_silent(silent, sr=sr) is True

    quiet_noise = torch.randn(1, sr * 2) * 1e-4  # ~-80 dBFS
    assert is_stem_silent(quiet_noise, sr=sr) is True

    t = torch.arange(sr * 2, dtype=torch.float32) / sr
    active = torch.zeros(1, sr * 2, dtype=torch.float32)
    # 0.2s burst at -12 dBFS (~0.25 amplitude) -> 10% active frames
    i0, i1 = int(0.5 * sr), int(0.7 * sr)
    active[0, i0:i1] = 0.25 * torch.sin(2.0 * math.pi * 440.0 * t[i0:i1])
    assert is_stem_silent(active, sr=sr) is False


def test_separate_stems_handles_mono_48khz_and_clears_cuda(monkeypatch):
    calls: dict[str, object] = {"empty_cache": 0}

    class _FakeDemucsModel:
        samplerate = 44100
        audio_channels = 2
        sources = ["drums", "bass", "other", "vocals"]

        def to(self, _dev):
            return self

        def eval(self):
            return self

    def _fake_load_backend():
        def _get_model(name: str):
            calls["model_name"] = name
            return _FakeDemucsModel()

        def _apply_model(model, mix, device=None, split=True, segment=None, progress=False):
            # mix shape: [batch=1, channels=2, T]
            calls["input_shape"] = tuple(mix.shape)
            calls["segment"] = segment
            b, c, t = mix.shape
            # Return [batch, num_sources, channels, T]
            return torch.stack([mix * (0.1 * (i + 1)) for i in range(len(model.sources))], dim=1)

        return _get_model, _apply_model

    monkeypatch.setattr("muscriptor.utils.stems._load_demucs_backend", _fake_load_backend)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "torch.cuda.empty_cache",
        lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
    )

    mono_48k = torch.randn(1, 48000, dtype=torch.float32)
    stems = separate_stems(mono_48k, sr=48000, mode="stems_4", device="cpu", segment=7.5)

    assert calls["model_name"] == "htdemucs_ft"
    # Resampled from 48k -> 44.1k and expanded mono -> stereo
    assert calls["input_shape"][0] == 1
    assert calls["input_shape"][1] == 2
    assert abs(calls["input_shape"][2] - 44100) <= 4
    assert set(stems.keys()) == {"drums", "bass", "vocals", "other"}
    for wav, out_sr in stems.values():
        assert out_sr == 44100
        assert wav.device.type == "cpu"
        assert wav.shape[0] == 2
    assert calls["empty_cache"] >= 1


def test_auto_dtype_selects_half_for_large_on_8gb_cuda(monkeypatch):
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda _dev: SimpleNamespace(total_memory=8 * 1024**3),
    )
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: True)
    assert _auto_dtype(torch.device("cuda"), _CONFIGS["large"]) == torch.bfloat16
    assert _auto_dtype(torch.device("cuda"), _CONFIGS["medium"]) == torch.float32

    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: False)
    assert _auto_dtype(torch.device("cuda"), _CONFIGS["large"]) == torch.float16

    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda _dev: SimpleNamespace(total_memory=24 * 1024**3),
    )
    assert _auto_dtype(torch.device("cuda"), _CONFIGS["large"]) == torch.float32
    assert _auto_dtype(torch.device("mps"), _CONFIGS["medium"]) == torch.float16
    assert _auto_dtype(torch.device("cpu"), _CONFIGS["large"]) == torch.float32


def test_is_stem_silent_handles_sub_50ms_clips_and_3d_tensors():
    sr = 16000
    n = int(0.02 * sr)  # 20 ms < 50 ms frame_len
    t = torch.arange(n, dtype=torch.float32) / sr
    active_short = (0.4 * torch.sin(2.0 * math.pi * 440.0 * t)).unsqueeze(0)
    silent_short = torch.zeros(1, n, dtype=torch.float32)
    active_3d = active_short.repeat(2, 1).unsqueeze(0)  # [1, 2, T]

    assert is_stem_silent(active_short, sr=sr) is False
    assert is_stem_silent(silent_short, sr=sr) is True
    assert is_stem_silent(active_3d, sr=sr) is False


def test_load_model_builds_in_target_dtype_and_loads_weights_on_cpu(monkeypatch, tmp_path):
    from muscriptor.transcription_model import TranscriptionModel, _ModelConfig, _build_model

    tiny_cfg = _ModelConfig(dim=32, num_heads=2, num_layers=1, card=64)
    # 1. _build_model directly constructs transformer, out_norm, and linear in target dtype
    # while keeping condition_provider in float32 (avoiding fp32 peak allocation on 8 GB GPUs).
    built = _build_model(torch.device("cpu"), tiny_cfg, dtype=torch.bfloat16)
    assert built.emb.weight.dtype == torch.bfloat16
    assert built.out_norm.weight.dtype == torch.bfloat16
    assert built.linear.weight.dtype == torch.bfloat16
    assert next(built.condition_provider.parameters()).dtype == torch.float32

    # 2. load_model loads safetensors onto CPU first (not directly onto CUDA alongside model weights).
    fake_weights = tmp_path / "model.safetensors"
    fake_weights.write_bytes(b"")
    seen: dict[str, object] = {}

    def _fake_load_file(path, device="cpu"):
        seen["load_file_device"] = device
        return built.state_dict()

    monkeypatch.setattr("muscriptor.transcription_model.download_if_necessary", lambda src: fake_weights)
    monkeypatch.setattr("muscriptor.transcription_model._resolve_config", lambda src, p: tiny_cfg)
    monkeypatch.setattr("muscriptor.transcription_model.load_file", _fake_load_file)

    tm = TranscriptionModel.load_model(weights_path=fake_weights, device="cpu", dtype="bfloat16")
    assert seen["load_file_device"] == "cpu"
    assert tm._model.linear.weight.dtype == torch.bfloat16


def test_build_model_places_all_params_and_buffers_on_target_device():
    from muscriptor.modules.conditioners import ConditioningAttributes, WavCondition
    from muscriptor.transcription_model import _ModelConfig, _build_model

    tiny_cfg = _ModelConfig(dim=32, num_heads=2, num_layers=1, card=1395)
    meta_model = _build_model(torch.device("meta"), tiny_cfg, dtype=torch.bfloat16)
    bad_params = [n for n, p in meta_model.named_parameters() if p.device.type != "meta"]
    bad_buffers = [n for n, b in meta_model.named_buffers() if b.device.type != "meta"]
    assert bad_params == []
    assert bad_buffers == []

    if torch.cuda.is_available():
        cuda_model = _build_model(torch.device("cuda"), tiny_cfg, dtype=torch.bfloat16)
        cuda_model.eval()
        cond = ConditioningAttributes(
            text={"instrument_group": "0", "dataset_name": "0"},
            wav={
                "self_wav": WavCondition(
                    wav=torch.zeros(1, 1, 16000),
                    length=torch.tensor([16000]),
                    sample_rate=[16000],
                )
            },
        )
        tokens = list(
            cuda_model.generate(
                conditions=[cond],
                max_gen_len=2,
                use_sampling=False,
                cfg_coef=1.0,
                early_stop_on_token=1,
                beam_size=2,
            )
        )
        assert len(tokens) > 0
