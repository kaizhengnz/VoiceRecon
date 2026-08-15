"""Context assembly for AI presets.

A preset's ``context`` field is either ``"current"`` (send just the segment
that fired the trigger) or ``"window:<seconds>"`` (send every retained
segment whose end timestamp is within the last N seconds, filtered by the
preset's speaker filter). This module owns the parsing and the assembly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """One completed utterance from a single stream.

    ``end`` is a monotonic seconds value; only relative differences are used
    for windowing, so any consistent clock is fine.
    """

    speaker: str  # "them" or "me"
    text: str
    end: float


def parse(context_spec: str) -> tuple[str, float]:
    """Return ``(kind, window_seconds)``.

    ``kind`` is ``"current"`` or ``"window"``. For ``"current"`` the second
    element is 0. For ``"window:N"`` it is the parsed positive number.
    """
    spec = (context_spec or "").strip()
    if spec == "current":
        return "current", 0.0
    if spec.startswith("window:"):
        _, _, rest = spec.partition(":")
        try:
            seconds = float(rest)
        except ValueError:
            raise ValueError(f"context {spec!r} needs a number after 'window:'") from None
        if seconds <= 0:
            raise ValueError(f"context {spec!r} window must be positive")
        return "window", seconds
    raise ValueError(
        f"context {spec!r} must be 'current' or 'window:<seconds>'"
    )


def assemble(
    context_spec: str,
    segments: Iterable[Segment],
    trigger: Segment,
    speaker_filter: str,
) -> list[Segment]:
    """Return the ordered list of segments to include in the AI prompt.

    ``trigger`` is the segment whose completion fired the send. For
    ``current`` we return exactly that one (it must already match the
    speaker filter; the runner is expected to enforce that upstream).

    For ``window`` we take every segment ending within ``trigger.end -
    window`` .. ``trigger.end``, keeping only speakers the filter admits,
    in original order. The trigger segment is always included.
    """
    kind, window = parse(context_spec)
    if kind == "current":
        return [trigger]

    keep: list[Segment] = []
    lower_bound = trigger.end - window
    for segment in segments:
        if segment.end < lower_bound:
            continue
        if segment.end > trigger.end:
            continue
        if not _speaker_matches(segment.speaker, speaker_filter):
            continue
        keep.append(segment)
    # The trigger is always the last segment we send; if the caller did not
    # include it in ``segments``, append it explicitly.
    if (not keep or keep[-1] is not trigger) and trigger not in keep:
        keep.append(trigger)
    return keep


def _speaker_matches(speaker: str, filter_value: str) -> bool:
    if filter_value == "both":
        return True
    return speaker == filter_value


def render(segments: Iterable[Segment]) -> str:
    """Format segments as ``[them|me]: text`` lines, one per segment.

    This is the payload the model actually sees under ``user`` role. The
    preset's own prompt is delivered as the system prompt.
    """
    return "\n".join(f"[{seg.speaker}]: {seg.text}" for seg in segments)
