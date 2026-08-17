"""Terminal output helpers: consistent prefixes and credential masking.

Every credential is masked before it reaches the terminal or a log.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

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
