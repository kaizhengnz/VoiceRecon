"""Segmentation logic: turn per-stream VAD events into cut boundaries and
route the accepted audio to a streaming transcription sink.

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

Audio that survives the loopback-priority filter is forwarded to the
optional ``on_stream_audio`` callback (typically the runner's
per-speaker :class:`voicerecon.streaming.StreamingTranscriber`) as it
arrives; the segmenter itself no longer buffers audio blocks. Segment
boundaries still land on the ready queue as lightweight
:class:`ReadySegment` records — the runner uses each one to know when
to ``finalize`` the matching streamer and hand the accumulated text to
the transcript writer and AI.

The segmenter is thread-safe: two audio threads call
:meth:`on_audio` / :meth:`on_events` concurrently, and the main thread
consumes the ready queue. Access is serialized by an internal lock;
the ``on_stream_audio`` callback runs *outside* the lock so a slow
sink cannot stall the capture threads.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from . import vad

StreamAudioSink = Callable[[str, np.ndarray], None]


@dataclass
class PendingSegment:
    """Boundary state per stream; ``had_audio`` gates empty-utterance drop."""

    speaker: str
    started_at: float | None = None
    had_audio: bool = False

    @property
    def active(self) -> bool:
        return self.started_at is not None

    def note_audio(self) -> None:
        if self.active:
            self.had_audio = True

    def start(self, timestamp: float) -> None:
        self.started_at = timestamp
        self.had_audio = False

    def cut(self, ended_at: float) -> "ReadySegment | None":
        if not self.active:
            return None
        started = self.started_at or ended_at
        had_audio = self.had_audio
        self.started_at = None
        self.had_audio = False
        if not had_audio:
            return None
        return ReadySegment(
            speaker=self.speaker, started_at=started, ended_at=ended_at
        )


@dataclass
class ReadySegment:
    """One utterance boundary. Text lives in the runner's streaming buffer."""

    speaker: str
    started_at: float
    ended_at: float


class Segmenter:
    """Owns the per-stream state and the ready-boundary queue."""

    def __init__(self, *, on_stream_audio: StreamAudioSink | None = None) -> None:
        self._pending: dict[str, PendingSegment] = {
            "me": PendingSegment("me"),
            "them": PendingSegment("them"),
        }
        self._ready: deque[ReadySegment] = deque()
        self._lock = threading.Lock()
        self._on_stream_audio = on_stream_audio

    @property
    def _loopback_active(self) -> bool:
        return self._pending["them"].active

    def _emit(self, cut: "ReadySegment | None") -> None:
        if cut is not None:
            self._ready.append(cut)

    def on_audio(self, speaker: str, samples: np.ndarray) -> None:
        """Route audio for the currently-active segment on ``speaker``.

        Silent audio (before a ``start`` event on this stream) is dropped,
        and mic audio is dropped entirely while loopback is active.
        Accepted audio is forwarded to the streaming sink (if configured)
        after the segmenter's own lock is released.
        """
        if samples.size == 0:
            return
        forward = False
        with self._lock:
            if speaker == "me" and self._loopback_active:
                return
            pending = self._pending[speaker]
            if not pending.active:
                return
            pending.note_audio()
            forward = self._on_stream_audio is not None
        if forward:
            self._on_stream_audio(speaker, samples)  # type: ignore[misc]

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
        """Pop every boundary that is ready right now."""
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
