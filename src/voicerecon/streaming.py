"""Incremental transcription via LocalAgreement-2.

Standard Whisper is offline: you hand it an utterance and it returns the
whole text at once. For a long turn (e.g. an interviewer asking a 20 s
question) that means the transcript only appears after the speaker stops
— even though most of the words were locked in long before then.

LocalAgreement-2 (Machaček et al., IWSLT 2023) buys latency without
giving up Whisper's quality: transcribe a growing audio buffer on a
short cadence (``min_chunk_seconds``), and *commit* the longest prefix
of words that two consecutive hypotheses agree on. Two independent runs
that produce the same prefix are treated as a stable transcription for
that portion of audio; the committed audio is trimmed off the buffer so
Whisper's context stays bounded, and the next run works on the tail.

The class is thread-safe by design: ``feed`` runs on the audio capture
thread and ``commit_step`` / ``finalize`` on the main loop. The internal
lock only wraps buffer mutations — Whisper itself runs outside the lock,
so a slow transcription cannot stall the capture thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DEFAULT_MIN_CHUNK_SECONDS = 1.0
"""Paper's sweet spot on the latency/CPU curve; sub-second thrashes the model
without much perceptual gain, multi-second erodes the streaming feel."""

MAX_BUFFER_SECONDS = 5.0
"""Soft cap on the rolling buffer. LocalAgreement-2 needs the same audio
transcribed twice for the prefixes to be compared, so trimming the front
of the buffer would defeat agreement. Instead, when :meth:`commit_step`
finds the buffer at or past this cap it force-commits the whole current
Whisper hypothesis and resets — trading boundary accuracy for the
responsiveness that makes the streaming feel real-time. Whisper on 5 s
of audio finishes well under a second on CPU with the ``small`` model,
which sets the worst-case first-text latency during continuous speech."""

MAX_PROMPT_WORDS = 60
"""How many trailing committed words to feed back as Whisper's
``initial_prompt`` on the next pass. Enough to stabilise the decoder's
tokenisation of the fresh audio without approaching Whisper's ~200-token
prompt window."""


class StreamingTranscriber:
    """Feed audio, poll ``commit_step`` for stabilised text."""

    def __init__(
        self,
        model_factory: Callable[[], Any],
        *,
        min_chunk_seconds: float = DEFAULT_MIN_CHUNK_SECONDS,
        sample_rate: int = SAMPLE_RATE,
        label: str = "",
    ) -> None:
        self._model_factory = model_factory
        self._model: Any | None = None
        self._min_chunk_samples = int(min_chunk_seconds * sample_rate)
        self._max_buffer_samples = int(MAX_BUFFER_SECONDS * sample_rate)
        self._sample_rate = sample_rate
        self._label = label
        self._buffer = np.empty(0, dtype=np.float32)
        self._prev_words: list[str] = []
        self._committed_text: list[str] = []
        self._first_feed_logged = False
        self._lock = threading.Lock()

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self._model_factory()
        return self._model

    def feed(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, samples])
            # No front-trim here: dropping the front of the buffer would
            # defeat LocalAgreement-2, and clearing _prev_words on every
            # audio block starves commit_step of anything to compare
            # against. Buffer size is bounded by the force-flush branch
            # in commit_step (see MAX_BUFFER_SECONDS).
            if not self._first_feed_logged:
                self._first_feed_logged = True
                logger.debug(
                    "[%s] feed: first audio arrived (%d samples)",
                    self._label, samples.size,
                )

    def commit_step(self) -> str:
        """Return newly committed text, or ``""`` if nothing is stable yet."""
        with self._lock:
            if self._buffer.size < self._min_chunk_samples:
                return ""
            snapshot = self._buffer
            prompt = self._build_prompt()
        buf_s = snapshot.size / self._sample_rate
        at_cap = snapshot.size >= self._max_buffer_samples
        # Transcribe outside the lock. Trim below operates on self._buffer
        # (possibly extended by feed() in the meantime) from the front —
        # correct because feed() only appends, so the origin stays aligned
        # to the snapshot.
        started = time.monotonic()
        words = self._transcribe_words(snapshot, prompt=prompt)
        whisper_s = time.monotonic() - started
        logger.debug(
            "[%s] pump buf=%.2fs whisper=%.2fs words=%d%s",
            self._label, buf_s, whisper_s, len(words),
            " at-cap" if at_cap else "",
        )
        texts = [w.word for w in words]
        with self._lock:
            if at_cap:
                # Buffer hit the cap. Commit whatever Whisper returned
                # (empty list included — that means the tail was silent
                # or unintelligible and there's no point re-transcribing
                # the same audio next pass) and reset. Preserve any
                # samples feed() appended after the snapshot.
                added = self._buffer.size - snapshot.size
                self._buffer = (
                    self._buffer[-added:].copy()
                    if added > 0
                    else np.empty(0, dtype=np.float32)
                )
                self._prev_words = []
                if texts:
                    self._committed_text.extend(texts)
                    logger.debug(
                        "[%s]   force-flush emit %d words: %r",
                        self._label, len(texts), "".join(texts),
                    )
                else:
                    logger.debug("[%s]   force-flush empty (no words); buffer reset", self._label)
                return "".join(texts)
            if not words:
                self._prev_words = []
                return ""
            lcp = _common_prefix(texts, self._prev_words)
            if not lcp:
                self._prev_words = texts
                logger.debug(
                    "[%s]   prime prev=%d: %r",
                    self._label, len(texts), "".join(texts),
                )
                return ""
            commit_count = len(lcp)
            trim_samples = int(words[commit_count - 1].end * self._sample_rate)
            if trim_samples > self._buffer.size:
                trim_samples = self._buffer.size
            self._buffer = self._buffer[trim_samples:]
            self._prev_words = texts[commit_count:]
            self._committed_text.extend(lcp)
            logger.debug(
                "[%s]   commit lcp=%d: %r",
                self._label, commit_count, "".join(lcp),
            )
        return "".join(lcp)

    def finalize(self) -> str:
        """Return whatever is left in the buffer as final text, then reset."""
        with self._lock:
            if self._buffer.size == 0:
                self._prev_words = []
                return ""
            snapshot = self._buffer
            prompt = self._build_prompt()
        words = self._transcribe_words(snapshot, prompt=prompt)
        text = "".join(w.word for w in words)
        self.reset()
        return text

    def reset(self) -> None:
        with self._lock:
            self._buffer = np.empty(0, dtype=np.float32)
            self._prev_words = []
            self._committed_text = []

    def _build_prompt(self) -> str:
        """Return the trailing committed text passed to Whisper as its
        ``initial_prompt`` on the next pass. Biases the decoder to produce
        the same tokenisation for the just-committed context, which is
        what lets LocalAgreement-2 converge on real speech instead of
        thrashing on tokenisation drift."""
        return "".join(self._committed_text[-MAX_PROMPT_WORDS:]).strip()

    def _transcribe_words(self, audio: np.ndarray, *, prompt: str = "") -> list[Any]:
        try:
            model = self._ensure_model()
            segments, _ = model.transcribe(
                audio,
                language=None,
                vad_filter=False,
                word_timestamps=True,
                initial_prompt=prompt or None,
            )
            out: list[Any] = []
            for segment in segments:
                for word in getattr(segment, "words", None) or ():
                    out.append(word)
            return out
        except Exception:
            return []


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    """Longest agreeing prefix, tolerant of Whisper's leading-space and case
    drift across passes; the returned text keeps ``a``'s original formatting
    so downstream printing is not stripped of its natural word boundaries."""
    out: list[str] = []
    for x, y in zip(a, b):
        if x.strip().lower() != y.strip().lower():
            break
        out.append(x)
    return out
