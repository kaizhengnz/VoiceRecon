"""Speech-to-text via faster-whisper.

The model is loaded lazily on the first :meth:`transcribe` call: the
download and CTranslate2 init cost several seconds, and paying it during
``--help`` / ``--configure`` is user-hostile.

Auto language detection is on by default; each utterance is transcribed
independently, so a session that switches languages mid-way is handled per
segment (Whisper decides on each first frame). The detected language is
returned so the transcript writer can note it if desired.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

DEFAULT_MODEL_SIZE = "small"
"""Sweet spot for CJK + Latin scripts. ``base`` is faster but noticeably
weaker on non-English speech; ``medium`` is more accurate but 3× the size."""


class Transcriber:
    """Wraps a WhisperModel and offers one method: :meth:`transcribe`.

    The model is expensive to build (I/O + native init), so callers should
    keep one instance for the lifetime of a session.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        """``model_factory`` is a seam for tests: a callable returning any
        object with a ``.transcribe(audio, language=None, ...)`` method.
        Production leaves it ``None`` and faster-whisper is loaded lazily."""
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._factory = model_factory
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._factory is not None:
            self._model = self._factory(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
            return self._model
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        self._model = WhisperModel(
            self._model_size, device=self._device, compute_type=self._compute_type
        )
        return self._model

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe ``audio`` (float32 mono, 16 kHz). Never raises.

        Returns the empty string on backend failure so the caller can skip
        the segment without crashing the pipeline.
        """
        if audio.size == 0:
            return ""
        try:
            model = self._ensure_model()
            segments, _ = model.transcribe(audio, language=None, vad_filter=False)
            text_parts = [segment.text for segment in segments]
        except Exception:
            return ""
        return "".join(text_parts).strip()
