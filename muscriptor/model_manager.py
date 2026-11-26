"""Lazy VRAM-aware lifecycle manager for MuScriptor checkpoints."""

import gc
import os
import threading

import torch

import muscriptor.accelerator
from muscriptor.transcription_model import (
    _DEFAULT_SIZE,
    _MODEL_SIZES,
    TranscriptionModel,
)
from muscriptor.utils.stems import available_separation_modes, is_separation_available


def _hf_saved_token() -> str | None:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


class ModelManager:
    """Manages lazy loading, model-size switching, and GPU unloading."""

    def __init__(
        self,
        default_model: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self._default_model = default_model or _DEFAULT_SIZE
        self._device = device
        self._dtype = dtype
        self._custom_model_locked = self._default_model not in _MODEL_SIZES
        self._lock = threading.Lock()
        self._model: TranscriptionModel | None = None
        self._loaded_name: str | None = None

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_name

    @property
    def custom_model_locked(self) -> bool:
        return self._custom_model_locked

    @staticmethod
    def hf_auth_configured() -> bool:
        return bool(os.environ.get("HF_TOKEN") or _hf_saved_token())

    def _resolved_device(self) -> torch.device:
        if self._device is not None:
            return torch.device(self._device)
        if muscriptor.accelerator.is_available():
            return muscriptor.accelerator.current_accelerator()
        return torch.device("cpu")

    def capabilities(self) -> dict:
        dev = self._resolved_device()
        gpu_name: str | None = None
        vram_mb: int | None = None
        if dev.type == "cuda" and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(dev)
                gpu_name = props.name
                vram_mb = int(props.total_memory // (1024 * 1024))
            except Exception:
                pass
        elif dev.type == "mps":
            gpu_name = "Apple Silicon (MPS)"

        return {
            "device": str(dev),
            "gpu_name": gpu_name,
            "vram_mb": vram_mb,
            "loaded_model": self._loaded_name,
            "default_model": self._default_model,
            "custom_model_locked": self._custom_model_locked,
            "available_models": list(_MODEL_SIZES),
            "hf_auth_configured": self.hf_auth_configured(),
            "separation_available": is_separation_available(),
            "separation_modes": available_separation_modes(),
        }

    def release_gpu(self) -> None:
        """Unload any currently resident model off CUDA to free VRAM for Demucs."""
        with self._lock:
            if self._model is None:
                return
            del self._model
            self._model = None
            self._loaded_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model(self, model_size: str | None = None) -> TranscriptionModel:
        """Return the requested TranscriptionModel, lazily loading or switching as needed."""
        if self._custom_model_locked:
            target = self._default_model
        else:
            target = model_size or self._default_model
            if target not in _MODEL_SIZES:
                raise ValueError(
                    f"Unknown model size {target!r}; expected one of {', '.join(_MODEL_SIZES)}"
                )

        with self._lock:
            if self._model is not None and self._loaded_name == target:
                return self._model
            if self._model is not None:
                del self._model
                self._model = None
                self._loaded_name = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._model = TranscriptionModel.load_model(
                weights_path=target,
                device=self._device,
                dtype=self._dtype,
            )
            self._loaded_name = target
            return self._model
