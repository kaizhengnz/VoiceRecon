"""Terminal output helpers: consistent prefixes and credential masking.

Every credential is masked before it reaches the terminal or a log.

:func:`tee_to_file` redirects ``sys.stdout`` and ``sys.stderr`` through
a tee wrapper so every print — from :func:`info`, from
:mod:`voicerecon.streaming`'s diagnostics, from raw ``print`` in the
runner, and from any third-party library — is also appended to a daily
log file. The tee runs for the process's lifetime; there is no matching
untee.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

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


def warn(message: str) -> None:
    print(f"[warn] {message}", flush=True)


def error(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr, flush=True)


def rule(title: str = "") -> None:
    """Print a separator line, optionally with a title."""
    if title:
        print(f"\n---- {title} " + "-" * max(0, 40 - len(title)), flush=True)
    else:
        print("-" * 48, flush=True)


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


class _Tee:
    """Duplicate every write to ``original`` and ``log_file`` both."""

    def __init__(self, original: Any, log_file: Any) -> None:
        self._orig = original
        self._log = log_file

    def write(self, s: str) -> int:
        n = self._orig.write(s)
        try:
            self._log.write(s)
        except Exception:
            pass
        return n

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._log.flush()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


def tee_to_file(path: Path | str) -> Path:
    """Tee stdout and stderr to ``path`` for the rest of the process.

    Creates the parent directory if needed. Appends across runs so a
    single daily file collects every session's output; a session header
    line marks the boundary.
    """
    resolved = Path(str(path)).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(resolved, "a", encoding="utf-8", buffering=1)
    log_file.write(
        f"\n=== session {datetime.now().isoformat(timespec='seconds')} ===\n"
    )
    log_file.flush()
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return resolved
