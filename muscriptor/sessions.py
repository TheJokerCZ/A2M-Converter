"""Bounded LRU session store for separated WAV stems."""

import re
import shutil
import tempfile
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch

from muscriptor.utils.audio import _read_wav_file
from muscriptor.utils.fl_export import write_pcm16_wav_bytes

SAFE_STEM_NAMES: frozenset[str] = frozenset(
    {"drums", "bass", "vocals", "guitar", "piano", "other"}
)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class SessionStore:
    """Persists separated stem WAVs in a bounded temporary directory."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        max_sessions: int = 3,
    ) -> None:
        self._root = (
            Path(root_dir)
            if root_dir is not None
            else Path(tempfile.gettempdir()) / "muscriptor_sessions"
        )
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_sessions = max(1, int(max_sessions))
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, Path] = OrderedDict()

    @staticmethod
    def is_valid_session_id(session_id: str) -> bool:
        return bool(_UUID_RE.match(session_id))

    @staticmethod
    def is_valid_stem_name(stem_name: str) -> bool:
        return stem_name in SAFE_STEM_NAMES

    def create_session(
        self, stems: Mapping[str, tuple[torch.Tensor, int]]
    ) -> str:
        session_id = str(uuid.uuid4())
        session_dir = self._root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        for name, (wav, sr) in stems.items():
            if not self.is_valid_stem_name(name):
                continue
            wav_bytes = write_pcm16_wav_bytes(wav, sr)
            (session_dir / f"{name}.wav").write_bytes(wav_bytes)

        with self._lock:
            self._sessions[session_id] = session_dir
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                _, evicted_dir = self._sessions.popitem(last=False)
                shutil.rmtree(evicted_dir, ignore_errors=True)

        return session_id

    def get_stem_path(self, session_id: str, stem_name: str) -> Path | None:
        if not self.is_valid_session_id(session_id):
            return None
        if not self.is_valid_stem_name(stem_name):
            return None
        with self._lock:
            session_dir = self._sessions.get(session_id)
            if session_dir is None:
                return None
            self._sessions.move_to_end(session_id)
        candidate = (session_dir / f"{stem_name}.wav").resolve()
        if candidate.parent != session_dir.resolve() or not candidate.is_file():
            return None
        return candidate

    def load_stems(
        self, session_id: str
    ) -> dict[str, tuple[torch.Tensor, int]]:
        if not self.is_valid_session_id(session_id):
            return {}
        with self._lock:
            session_dir = self._sessions.get(session_id)
            if session_dir is None:
                return {}
            self._sessions.move_to_end(session_id)

        out: dict[str, tuple[torch.Tensor, int]] = {}
        for name in ("drums", "bass", "vocals", "guitar", "piano", "other"):
            p = session_dir / f"{name}.wav"
            if p.is_file():
                out[name] = _read_wav_file(p)
        return out

    def cleanup(self) -> None:
        with self._lock:
            dirs = list(self._sessions.values())
            self._sessions.clear()
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
