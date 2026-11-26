"""Stem separation and per-stem instrument conditioning/masking planning."""

import dataclasses
import gc
import importlib.util
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch

import muscriptor.accelerator
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
from muscriptor.utils.audio import resample

SeparationMode = Literal["none", "stems_4", "stems_6"]

STEM_NAMES_BY_MODE: dict[str, tuple[str, ...]] = {
    "stems_4": ("drums", "bass", "vocals", "other"),
    "stems_6": ("drums", "bass", "vocals", "guitar", "piano", "other"),
}

DEMUCS_MODEL_BY_MODE: dict[str, str] = {
    "stems_4": "htdemucs_ft",
    "stems_6": "htdemucs_6s",
}

_DEDICATED_STEM_GROUPS: dict[str, list[str]] = {
    "drums": ["drums"],
    "vocals": ["voice"],
    "bass": ["electric_bass", "acoustic_bass", "contrabass"],
    "guitar": [
        "acoustic_guitar",
        "clean_electric_guitar",
        "distorted_electric_guitar",
    ],
    "piano": ["acoustic_piano", "electric_piano"],
}


class StemSeparationUnavailableError(RuntimeError):
    """Raised when stem separation is requested without Demucs installed."""


@dataclasses.dataclass(frozen=True)
class StemPlan:
    """How a single separated stem should be passed to `TranscriptionModel.transcribe`."""

    stem: str
    # Explicit ClassConditioner + logit mask (used for single-group stems or
    # explicit user instrument selections).
    instruments: list[str] | None
    # Logit mask only (ClassConditioner stays unconditional so multi-instrument
    # stems like `other` auto-detect without out-of-distribution prompts).
    allowed_instruments: list[str] | None


def is_separation_available() -> bool:
    """True when `demucs-infer` or `demucs` is importable in this environment."""
    return (
        importlib.util.find_spec("demucs_infer") is not None
        or importlib.util.find_spec("demucs") is not None
    )


def available_separation_modes() -> list[str]:
    if is_separation_available():
        return ["none", "stems_4", "stems_6"]
    return ["none"]


def _allowed_groups_for_stem(stem: str, mode: SeparationMode) -> list[str]:
    if stem in _DEDICATED_STEM_GROUPS:
        return list(_DEDICATED_STEM_GROUPS[stem])
    # "other": everything in MT3_FULL_PLUS except stems already split out in `mode`.
    excluded: set[str] = set()
    for split_stem in STEM_NAMES_BY_MODE[mode]:
        if split_stem != "other":
            excluded.update(_DEDICATED_STEM_GROUPS.get(split_stem, ()))
    return [name for name in MT3_FULL_PLUS_GROUP_NAMES if name not in excluded]


def plan_stem_conditioning(
    mode: SeparationMode,
    active_stems: Sequence[str] | None = None,
    user_instruments: Sequence[str] | None = None,
) -> list[StemPlan]:
    """Build ordered transcription plans for each active stem in `mode`."""
    if mode == "none":
        return []
    if mode not in STEM_NAMES_BY_MODE:
        raise ValueError(f"Unknown separation mode: {mode!r}")

    mode_stems = STEM_NAMES_BY_MODE[mode]
    if active_stems:
        selected_set = set(active_stems)
        stems_to_run = [s for s in mode_stems if s in selected_set]
    else:
        stems_to_run = list(mode_stems)

    user_list = list(user_instruments) if user_instruments else None
    user_set = set(user_list) if user_list else None

    plans: list[StemPlan] = []
    for stem in stems_to_run:
        allowed = _allowed_groups_for_stem(stem, mode)
        if user_set is not None:
            inter = [name for name in allowed if name in user_set]
            if not inter:
                continue
            plans.append(
                StemPlan(stem=stem, instruments=inter, allowed_instruments=None)
            )
        elif len(allowed) == 1:
            # Single-group stem (drums / vocals): activate both conditioning & mask.
            plans.append(
                StemPlan(stem=stem, instruments=allowed, allowed_instruments=None)
            )
        else:
            # Multi-group stem (bass / guitar / piano / other): unconditional
            # auto-detect with strict logit masking to prevent cross-stem bleed.
            plans.append(
                StemPlan(stem=stem, instruments=None, allowed_instruments=allowed)
            )
    return plans


def is_stem_silent(
    wav: torch.Tensor,
    sr: int = 16000,
    peak_db: float = -40.0,
    active_ratio: float = 0.005,
    frame_db: float = -45.0,
) -> bool:
    """Return True when a stem has negligible energy and can be skipped safely."""
    t = wav.detach().cpu().float()
    if t.numel() == 0:
        return True
    if t.dim() > 1:
        t = t.mean(dim=0)
    peak = float(t.abs().max().item())
    peak_thresh = 10.0 ** (peak_db / 20.0)
    if peak < peak_thresh:
        return True

    frame_len = max(1, int(round(sr * 0.05)))
    n_frames = max(1, t.numel() // frame_len)
    trimmed = t[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = torch.sqrt(torch.mean(trimmed * trimmed, dim=1))
    frame_thresh = 10.0 ** (frame_db / 20.0)
    active = float((frame_rms >= frame_thresh).float().mean().item())
    return active < active_ratio


def _load_demucs_backend() -> tuple[Callable[[str], Any], Callable[..., torch.Tensor]]:
    """Import and return `(get_model, apply_model)` from `demucs_infer` or `demucs`."""
    try:
        from demucs_infer.apply import apply_model
        from demucs_infer.pretrained import get_model

        return get_model, apply_model
    except ImportError:
        pass
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        return get_model, apply_model
    except ImportError as e:
        raise StemSeparationUnavailableError(
            "Stem separation requires `demucs-infer` (or `demucs`). "
            "Install with: `pip install demucs-infer`."
        ) from e


def separate_stems(
    wav: torch.Tensor,
    sr: int,
    mode: SeparationMode = "stems_4",
    device: str | torch.device | None = None,
    segment: float = 7.5,
) -> dict[str, tuple[torch.Tensor, int]]:
    """Separate `wav` ([C, T] or [T]) into stereo CPU stems at the separator's rate.

    Always frees the separator model and clears CUDA cache in `finally` so
    downstream MuScriptor inference gets the full GPU VRAM budget.
    """
    if mode == "none":
        return {}
    if mode not in DEMUCS_MODEL_BY_MODE:
        raise ValueError(f"Unknown separation mode: {mode!r}")

    get_model, apply_model = _load_demucs_backend()
    if device is None:
        target_device = (
            muscriptor.accelerator.current_accelerator()
            if muscriptor.accelerator.is_available()
            else torch.device("cpu")
        )
    else:
        target_device = torch.device(device)

    model = None
    try:
        model = get_model(DEMUCS_MODEL_BY_MODE[mode])
        model.eval()
        model.to(target_device)

        model_sr = int(getattr(model, "samplerate", 44100))
        model_channels = int(getattr(model, "audio_channels", 2))

        x = wav.detach().cpu().float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            x = x.squeeze(0)

        if sr != model_sr:
            x = resample(x, int(sr), model_sr)

        if x.shape[0] == 1 and model_channels == 2:
            x = x.repeat(2, 1)
        elif x.shape[0] > model_channels:
            x = x[:model_channels]
        elif x.shape[0] < model_channels:
            x = x.mean(dim=0, keepdim=True).repeat(model_channels, 1)

        # Normalize mix before Demucs and restore scale afterwards (Demucs convention).
        ref = x.mean(dim=0)
        ref_mean = ref.mean()
        ref_std = ref.std()
        norm_scale = float(ref_std.item()) if math.isfinite(float(ref_std.item())) and float(ref_std.item()) > 1e-6 else 1.0
        mix = ((x - ref_mean) / norm_scale).unsqueeze(0).to(target_device)

        with torch.inference_mode():
            separated = apply_model(
                model,
                mix,
                device=target_device,
                split=True,
                segment=segment,
                progress=False,
            )

        # separated: [batch=1, num_sources, channels, T]
        separated = (separated[0].detach().cpu().float() * norm_scale) + ref_mean
        source_names = list(getattr(model, "sources", STEM_NAMES_BY_MODE[mode]))

        out: dict[str, tuple[torch.Tensor, int]] = {}
        for idx, name in enumerate(source_names):
            if idx < separated.shape[0]:
                out[name] = (separated[idx].contiguous(), model_sr)
        return out
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
