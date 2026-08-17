"""Built-in scenario presets for the AI-per-segment pipeline.

Each preset declares:

- ``speaker_filter`` — which stream to consume. Segments from the ignored
  stream are dropped, not sent. Values: ``"both"``, ``"them"``, ``"me"``.
- ``context`` — what payload to include (per-segment triggers only).
  ``"current"`` sends only the segment that fired the trigger.
  ``"window:<seconds>"`` sends every retained segment (subject to the same
  speaker filter) whose end timestamp is within the last N seconds. Ignored
  when ``trigger`` is ``"on_shutdown"``.
- ``prompt`` — the system prompt shown to the model.
- ``trigger`` — when to call the AI. ``"per_segment"`` fires after each
  matching utterance (the default). ``"on_shutdown"`` fires once at Ctrl+C
  over the full session transcript, and its output is also saved to a file
  in ``save_dir`` alongside the transcript.

Selecting no preset yields transcript-only mode: segments are still written
to the local transcript file, but no AI call is made and Telegram is not
touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SPEAKER_FILTERS: tuple[str, ...] = ("both", "them", "me")
SpeakerFilter = Literal["both", "them", "me"]
Trigger = Literal["per_segment", "on_shutdown"]


@dataclass(frozen=True)
class Preset:
    """A scenario preset. See module docstring for the field semantics."""

    name: str
    speaker_filter: SpeakerFilter
    context: str
    prompt: str
    description: str
    trigger: Trigger = "per_segment"

    @property
    def is_batch(self) -> bool:
        """True when the AI fires once at Ctrl+C rather than per segment."""
        return self.trigger == "on_shutdown"

    @property
    def is_custom(self) -> bool:
        """True when this preset was synthesized from ``--prompt`` on the CLI."""
        return self.name == _CUSTOM_NAME


BUILT_IN: dict[str, Preset] = {
    "interview_candidate": Preset(
        name="interview_candidate",
        speaker_filter="them",
        context="current",
        prompt=(
            "You are helping the reader (a job candidate) answer an interview question. "
            "The following was just spoken by the interviewer. Identify what this "
            "question is testing and outline a concise answer approach."
        ),
        description="interview help for the candidate side",
    ),
    "interview_recruiter": Preset(
        name="interview_recruiter",
        speaker_filter="them",
        context="window:300",
        prompt=(
            "You are helping the reader (an interviewer) evaluate a candidate. "
            "Given the recent exchange with the candidate below, assess the latest "
            "response and suggest one concrete follow-up question."
        ),
        description="interview help for the recruiter side",
    ),
    "meeting_summary": Preset(
        name="meeting_summary",
        speaker_filter="both",
        context="current",
        prompt=(
            "The following is the full transcript of a conversation. "
            "[them] marks the other party, [me] marks the user. "
            "Produce a concise summary covering the main topics discussed, "
            "decisions reached, and any action items or open questions. "
            "Reply in the same language the conversation was primarily in, "
            "structured with short headers."
        ),
        description="summarize the whole session once at Ctrl+C",
        trigger="on_shutdown",
    ),
}


def get(name: str) -> Preset:
    """Return the preset with ``name`` or raise ``KeyError`` with a helpful list."""
    try:
        return BUILT_IN[name]
    except KeyError:
        known = ", ".join(sorted(BUILT_IN))
        raise KeyError(f"Unknown preset {name!r}. Known presets: {known}.") from None


def names() -> list[str]:
    return sorted(BUILT_IN)


def resolve(value: str) -> Preset | None:
    """Interpret a string as a preset selection.

    - Empty / whitespace → ``None`` (transcript-only)
    - Matches a :data:`BUILT_IN` name → that preset (with its baked
      speaker_filter / context / trigger)
    - Anything else → an ad-hoc :func:`make_custom` preset with default
      filter=them / context=current / trigger=per_segment

    A custom prompt that happens to equal a built-in name (unlikely in
    practice) resolves as the built-in — this is the one ambiguity of
    using a single string field for both selection modes.
    """
    stripped = (value or "").strip()
    if not stripped:
        return None
    if stripped in BUILT_IN:
        return BUILT_IN[stripped]
    return make_custom(stripped)


_CUSTOM_NAME = "custom"
_DESCRIPTION_MAX_LEN = 60


def make_custom(
    prompt: str, *, speaker_filter: SpeakerFilter = "them", context_spec: str = "current"
) -> Preset:
    """Build an ad-hoc streaming preset from user-supplied CLI values.

    Raises ``ValueError`` when the prompt is empty, the speaker filter is
    outside ``them/me/both``, or the context spec cannot be parsed. Batch
    (``on_shutdown``) is not offered via this path — the built-in
    ``meeting_summary`` preset is the sole way to get end-of-session
    summarization.
    """
    from . import context  # local import avoids a cycle at module load

    cleaned = (prompt or "").strip()
    if not cleaned:
        raise ValueError("--prompt requires non-empty text")
    if speaker_filter not in SPEAKER_FILTERS:
        raise ValueError(
            f"speaker filter must be one of {'/'.join(SPEAKER_FILTERS)}; "
            f"got {speaker_filter!r}"
        )
    context.parse(context_spec)

    return Preset(
        name=_CUSTOM_NAME,
        speaker_filter=speaker_filter,
        context=context_spec,
        prompt=cleaned,
        description=_summarize_prompt(cleaned),
    )


def _summarize_prompt(prompt: str) -> str:
    single_line = " ".join(prompt.split())
    if len(single_line) <= _DESCRIPTION_MAX_LEN:
        return single_line
    return single_line[: _DESCRIPTION_MAX_LEN - 3] + "..."
