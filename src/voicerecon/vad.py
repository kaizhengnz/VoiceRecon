"""Voice activity detection wrapper around Silero VAD.

We use silero-vad's ONNX build (no PyTorch required) via its ``VADIterator``,
which is designed for streaming: feed fixed-size frames, get back
``{"start": ...}`` / ``{"end": ...}`` markers when speech begins or ends.

Two independent :class:`StreamVAD` instances run in parallel — one for the
microphone, one for system loopback — and their events are consumed by
:mod:`voicerecon.segmenter`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

TARGET_SAMPLE_RATE = 16000
"""Silero's supported rates are 8000 and 16000; we standardise on 16 kHz."""

FRAME_SAMPLES = 512
"""Silero VAD expects 32 ms frames at 16 kHz (or 32 ms == 512 samples)."""


@dataclass
class SpeechEvent:
    """One VAD state transition on a single stream.

    ``kind`` is ``"start"`` or ``"end"``. ``timestamp`` is a monotonic
    seconds value (from ``time.monotonic()``) chosen by the caller — the
    segmenter uses it to enforce the silence threshold across streams.
    """

    kind: str
    timestamp: float


class StreamVAD:
    """Streaming VAD for one audio source.

    Feed :meth:`feed` with mono float32 samples in the range [-1, 1] at
    16 kHz. Any block size is accepted — the class buffers to the internal
    frame size Silero requires. Each call returns a list of events that
    fired during this block (usually zero, sometimes one).
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_silence_ms: int = 1500,
        min_speech_ms: int = 250,
        model_loader: Callable[[], Any] | None = None,
        iterator_factory: Callable[..., Any] | None = None,
    ) -> None:
        """``model_loader`` and ``iterator_factory`` are seams for tests.

        Production leaves them ``None`` and Silero's real model is loaded.
        Tests inject fakes that record calls and emit synthetic events.
        """
        self._threshold = threshold
        self._min_silence_ms = min_silence_ms
        self._min_speech_ms = min_speech_ms
        self._pending = np.empty(0, dtype=np.float32)
        self._model_loader = model_loader
        self._iterator_factory = iterator_factory
        self._iterator = self._build_iterator()

    def _build_iterator(self) -> Any:
        if self._iterator_factory is not None:
            return self._iterator_factory(
                threshold=self._threshold,
                min_silence_duration_ms=self._min_silence_ms,
                min_speech_duration_ms=self._min_speech_ms,
                sampling_rate=TARGET_SAMPLE_RATE,
            )
        loader = self._model_loader or _default_model_loader
        model = loader()
        from silero_vad import VADIterator  # type: ignore[import-not-found]
        return VADIterator(
            model,
            threshold=self._threshold,
            sampling_rate=TARGET_SAMPLE_RATE,
            min_silence_duration_ms=self._min_silence_ms,
            speech_pad_ms=100,
        )

    def feed(self, samples: np.ndarray, base_timestamp: float) -> list[SpeechEvent]:
        """Feed a block of samples. ``base_timestamp`` is the monotonic time
        at which the block *ends*; per-frame times are derived from it."""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)  # downmix to mono
        combined = np.concatenate([self._pending, samples])

        events: list[SpeechEvent] = []
        offset = 0
        block_samples = len(combined)
        block_duration = block_samples / TARGET_SAMPLE_RATE
        while offset + FRAME_SAMPLES <= block_samples:
            frame = combined[offset : offset + FRAME_SAMPLES]
            result = self._iterator(frame, return_seconds=False)
            if isinstance(result, dict):
                if "start" in result:
                    events.append(SpeechEvent("start", self._frame_time(
                        base_timestamp, offset + FRAME_SAMPLES, block_samples, block_duration
                    )))
                if "end" in result:
                    events.append(SpeechEvent("end", self._frame_time(
                        base_timestamp, offset + FRAME_SAMPLES, block_samples, block_duration
                    )))
            offset += FRAME_SAMPLES
        self._pending = combined[offset:].copy()
        return events

    @staticmethod
    def _frame_time(
        base_end: float, frame_end_sample: int, block_samples: int, block_duration: float
    ) -> float:
        # base_end corresponds to sample index == block_samples; earlier
        # frames are proportionally earlier.
        if block_samples <= 0:
            return base_end
        offset_from_end = (block_samples - frame_end_sample) / block_samples
        return base_end - offset_from_end * block_duration

def _default_model_loader() -> Any:
    from silero_vad import load_silero_vad  # type: ignore[import-not-found]
    return load_silero_vad(onnx=True)
