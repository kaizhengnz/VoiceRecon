"""Speech-to-text via faster-whisper.

The model is loaded lazily on the first :meth:`transcribe` call: the
download and CTranslate2 init cost several seconds, and paying it during
``--help`` / ``--configure`` is user-hostile.

The model runs on a CUDA GPU when one is visible and on the CPU otherwise;
:meth:`Transcriber._resolve_backend` owns that choice, including the compute
type, which cannot be left to faster-whisper's own ``auto`` (see there).

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


def _cuda_device_count() -> int:
    """Visible CUDA devices, or 0 when CTranslate2 cannot see any.

    Wrapped so a broken driver install degrades to CPU instead of killing
    transcription, and so tests can substitute it without a GPU.
    """
    try:
        import ctranslate2  # type: ignore[import-not-found]
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


class Transcriber:
    """Wraps a WhisperModel and offers one method: :meth:`transcribe`.

    The model is expensive to build (I/O + native init), so callers should
    keep one instance for the lifetime of a session.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        *,
        device: str = "auto",
        compute_type: str | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        """``device`` ``"auto"`` picks ``cuda`` when a CUDA device is visible
        and ``cpu`` otherwise; ``compute_type`` ``None`` then follows from the
        resolved device (see :meth:`_resolve_backend`). Both can be pinned by
        the caller.

        ``model_factory`` is a seam for tests: a callable returning any
        object with a ``.transcribe(audio, language=None, ...)`` method.
        Production leaves it ``None`` and faster-whisper is loaded lazily."""
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._factory = model_factory
        self._model: Any | None = None

    def _resolve_backend(self) -> tuple[str, str]:
        """Resolve ``("auto", None)`` into a concrete device + compute type.

        ``float16`` is the GPU default rather than letting faster-whisper
        choose: its own ``auto`` picks an int8 variant on Blackwell, and every
        int8 CUDA path there fails with ``CUBLAS_STATUS_NOT_SUPPORTED``.
        """
        device = self._device
        if device == "auto":
            device = "cuda" if _cuda_device_count() > 0 else "cpu"
        compute_type = self._compute_type
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        return device, compute_type

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        device, compute_type = self._resolve_backend()
        if self._factory is not None:
            self._model = self._factory(
                self._model_size, device=device, compute_type=compute_type
            )
            return self._model
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        self._model = WhisperModel(
            self._model_size, device=device, compute_type=compute_type
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
