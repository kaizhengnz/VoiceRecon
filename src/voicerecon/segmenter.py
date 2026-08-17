"""Segmentation logic: turn per-stream audio + VAD events into cut segments.

Loopback owns the timeline. Rules:

1. **Silence threshold** on the currently active stream — surfaces as an
   ``end`` event from that stream's :class:`voicerecon.vad.StreamVAD`
   (which has ``min_silence_duration_ms`` set to the configured value).
2. **Loopback preempts mic** — when the loopback (``them``) VAD emits
   ``start`` while the mic (``me``) has an open segment, that mic
   segment is cut immediately and the loopback segment begins.
3. **Mic is suppressed while loopback is active** — mic audio frames
   and mic VAD events (``start`` / ``end``) are dropped for the whole
   duration between a loopback ``start`` and its matching ``end``. The
   reverse preemption does *not* exist: mic starting has no effect on
   an active loopback segment.

The trade-off is that user speech which lands on top of loopback speech
is not captured, which is the right call for the standard speaker + mic
setup where the mic inevitably picks up the leaked speaker audio; a
symmetric cross-cut chopped both sides into millisecond fragments and
lost almost everything.

Buffered audio between the ``start`` and the cut is collected in a list
of numpy blocks; the caller (runner) reassembles it into one array and
sends it to STT.

The segmenter is thread-safe: two audio threads call
:meth:`process_events` concurrently, and the main thread consumes
:attr:`ready_segments`. Access is serialized by an internal lock.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from . import vad


@dataclass
class PendingSegment:
    """State per stream for the currently-being-collected segment."""

    speaker: str
    started_at: float | None = None
    blocks: list[np.ndarray] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.started_at is not None

    def append_audio(self, samples: np.ndarray) -> None:
        if self.active and samples.size:
            self.blocks.append(samples)

    def start(self, timestamp: float) -> None:
        self.started_at = timestamp
        self.blocks = []

    def cut(self, ended_at: float) -> ReadySegment | None:
        if not self.active:
            return None
        audio = _concat_blocks(self.blocks)
        started = self.started_at or ended_at
        self.started_at = None
        self.blocks = []
        if audio.size == 0:
            return None
        return ReadySegment(
            speaker=self.speaker,
            audio=audio,
            started_at=started,
            ended_at=ended_at,
        )


@dataclass
class ReadySegment:
    """One utterance ready for STT. Audio is float32 mono at 16 kHz."""

    speaker: str
    audio: np.ndarray
    started_at: float
    ended_at: float


def _concat_blocks(blocks: Iterable[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(list(blocks)).astype(np.float32, copy=False)


class Segmenter:
    """Owns the per-stream state and the ready-segment queue."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingSegment] = {
            "me": PendingSegment("me"),
            "them": PendingSegment("them"),
        }
        self._ready: deque[ReadySegment] = deque()
        self._lock = threading.Lock()

    @property
    def _loopback_active(self) -> bool:
        return self._pending["them"].active

    def _emit(self, cut: ReadySegment | None) -> None:
        if cut is not None:
            self._ready.append(cut)

    def on_audio(self, speaker: str, samples: np.ndarray) -> None:
        """Push audio for the currently-active segment on ``speaker``.

        Silent audio (before a ``start`` event on this stream) is dropped,
        and mic audio is dropped entirely while loopback is active.
        """
        with self._lock:
            if speaker == "me" and self._loopback_active:
                return
            self._pending[speaker].append_audio(samples)

    def on_events(self, speaker: str, events: Iterable[vad.SpeechEvent]) -> None:
        """Feed VAD events for ``speaker``.

        Loopback (``them``) events always run; mic (``me``) events are
        dropped for the duration of an active loopback segment. A
        loopback ``start`` also cuts any pending mic segment. See the
        module docstring for the full state machine.
        """
        with self._lock:
            for event in events:
                if event.kind == "start":
                    self._handle_start(speaker, event.timestamp)
                elif event.kind == "end":
                    self._handle_end(speaker, event.timestamp)

    def _handle_start(self, speaker: str, timestamp: float) -> None:
        if speaker == "me":
            if self._loopback_active:
                return
            self._pending["me"].start(timestamp)
            return
        # speaker == "them": loopback preempts any pending mic segment.
        self._emit(self._pending["me"].cut(timestamp))
        self._pending["them"].start(timestamp)

    def _handle_end(self, speaker: str, timestamp: float) -> None:
        if speaker == "me" and self._loopback_active:
            return
        self._emit(self._pending[speaker].cut(timestamp))

    def drain(self) -> list[ReadySegment]:
        """Pop every segment that is ready right now."""
        with self._lock:
            items = list(self._ready)
            self._ready.clear()
        return items

    def flush(self, timestamp: float) -> list[ReadySegment]:
        """Cut whatever is still open (used on graceful shutdown)."""
        with self._lock:
            for pending in self._pending.values():
                self._emit(pending.cut(timestamp))
            items = list(self._ready)
            self._ready.clear()
        return items
