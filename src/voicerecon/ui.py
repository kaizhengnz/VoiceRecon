"""Terminal output helpers: consistent prefixes and credential masking.

Every credential is masked before it reaches the terminal or a log.

:func:`info` / :func:`warn` / :func:`error` write to the terminal AND
emit a structured record on the ``voicerecon`` logger, so
:func:`setup_logging` can attach a rotating daily :class:`FileHandler`
without any callsite changes. Individual modules use their own
``logging.getLogger(__name__)`` for internal diagnostics (they do not
route through here, since those are debug-level and terminal output is
not desired).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_LOGGER = logging.getLogger("voicerecon")
# Library default: attach a NullHandler so logging is a no-op until
# setup_logging() runs (silences "No handlers found" and keeps tests
# quiet).
_LOGGER.addHandler(logging.NullHandler())

MASK_KEEP = 8
"""Number of leading characters kept when masking (show first 8 only)."""

MIN_SCRUB_LEN = 4
"""Shortest secret worth replacing."""


def mask(secret: str | None) -> str:
    """Render a credential as its first 8 characters plus a length hint."""
    if not secret:
        return "(not set)"
    text = str(secret)
    if len(text) <= MASK_KEEP:
        return text[0] + "*" * (len(text) - 1) if len(text) > 1 else "*"
    return f"{text[:MASK_KEEP]}... ({len(text)} chars)"


def scrub(text: str, secrets: Iterable[str | None]) -> str:
    """Strip credential plaintext out of arbitrary text before printing it."""
    result = str(text)
    for secret in secrets:
        if secret and len(str(secret)) >= MIN_SCRUB_LEN:
            result = result.replace(str(secret), mask(secret))
    return result


def info(message: str) -> None:
    print(message, flush=True)
    _LOGGER.info(message)


def warn(message: str) -> None:
    print(f"[warn] {message}", flush=True)
    _LOGGER.warning(message)


def error(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr, flush=True)
    _LOGGER.error(message)


def rule(title: str = "") -> None:
    """Print a separator line, optionally with a title."""
    if title:
        print(f"\n---- {title} " + "-" * max(0, 40 - len(title)), flush=True)
        _LOGGER.info("---- %s ----", title)
    else:
        print("-" * 48, flush=True)
        _LOGGER.info("-" * 48)


class SentenceStreamPrinter:
    """Buffer chunks from a streaming API and flush at sentence boundaries.

    Any of ``.!?。！？\\n`` counts as a sentence end. Chunks that don't
    contain one accumulate silently; :meth:`flush` prints whatever is
    left plus a trailing newline.
    """

    _TERMINATORS = ".!?。！？\n"

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def push(self, chunk: str) -> None:
        self._buffer.append(chunk)
        joined = "".join(self._buffer)
        idx = -1
        for term in self._TERMINATORS:
            found = joined.rfind(term)
            if found > idx:
                idx = found
        if idx < 0:
            return
        print(joined[: idx + 1], end="", flush=True)
        remainder = joined[idx + 1 :]
        self._buffer = [remainder] if remainder else []

    def flush(self) -> None:
        if self._buffer:
            print("".join(self._buffer), end="", flush=True)
            self._buffer = []
        print(flush=True)


def setup_logging(logs_dir: Path | str, *, level: int = logging.DEBUG) -> Path:
    """Configure the ``voicerecon`` logger to append to a daily file.

    ``<logs_dir>/YYYY-MM-DD.log`` is opened via
    :class:`TimedRotatingFileHandler` so each calendar day gets its own
    file with no manual rotation. Existing handlers on ``voicerecon``
    (the library-default :class:`NullHandler`, or a previous call's
    handler) are cleared first so tests and re-runs stay clean.

    ``level`` defaults to :data:`logging.DEBUG` so streaming diagnostics
    land in the file; the terminal output is unaffected because
    :func:`info` / :func:`warn` / :func:`error` write to the terminal
    directly via ``print`` and only *additionally* emit a log record.
    """
    resolved_dir = Path(str(logs_dir)).expanduser()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

    # Clear inherited handlers (NullHandler from module init, or a prior
    # setup_logging call from tests / re-entry).
    for handler in list(_LOGGER.handlers):
        _LOGGER.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    handler = TimedRotatingFileHandler(
        log_path, when="midnight", encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-7s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(level)

    _LOGGER.setLevel(level)
    _LOGGER.addHandler(handler)
    # A session-start marker gives grep-able boundaries when a single
    # day's log spans multiple runs.
    _LOGGER.info("=== session start ===")
    return log_path
