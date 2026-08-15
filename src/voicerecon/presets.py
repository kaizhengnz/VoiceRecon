"""Built-in scenario presets for the AI-per-segment pipeline.

Each preset declares:

- ``speaker_filter`` — which stream to consume. Segments from the ignored
  stream are dropped, not sent. Values: ``"both"``, ``"them"``, ``"me"``.
- ``context`` — what payload to include. ``"current"`` sends only the current
  segment. ``"window:<seconds>"`` sends every segment (subject to the same
  speaker filter) whose end timestamp falls within the last N seconds.
- ``prompt`` — the system prompt shown to the model.

Selecting no preset yields transcript-only mode: segments are still written
to the local transcript file, but no AI call is made and Telegram is not
touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpeakerFilter = Literal["both", "them", "me"]


@dataclass(frozen=True)
class Preset:
    """A scenario preset. See module docstring for the field semantics."""

    name: str
    speaker_filter: SpeakerFilter
    context: str
    prompt: str
    description: str


BUILT_IN: dict[str, Preset] = {
    "translate": Preset(
        name="translate",
        speaker_filter="them",
        context="current",
        prompt=(
            "Translate the following speech into Chinese. Output only the translation; "
            "no explanation, no prefix."
        ),
        description="translate the other party's speech into Chinese",
    ),
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
    "lecture": Preset(
        name="lecture",
        speaker_filter="them",
        context="window:300",
        prompt=(
            "You are taking notes on a lecture. Given the recent excerpt below, "
            "extract the key concept being taught and briefly explain any technical "
            "term that a beginner might not know."
        ),
        description="lecture note-taking helper",
    ),
    "speaking": Preset(
        name="speaking",
        speaker_filter="me",
        context="current",
        prompt=(
            "The reader is practising spoken language. Below is what they just said. "
            "Give short, specific feedback on grammar, word choice, and naturalness. "
            "If the sentence is fine, say so."
        ),
        description="spoken-language practice feedback",
    ),
    "debate": Preset(
        name="debate",
        speaker_filter="them",
        context="window:180",
        prompt=(
            "You are helping the reader argue against the speaker below. Given the "
            "recent statements, suggest one or two strong counter-arguments the "
            "reader could use."
        ),
        description="debate / discussion counter-argument helper",
    ),
    "sales": Preset(
        name="sales",
        speaker_filter="them",
        context="current",
        prompt=(
            "You are assisting a salesperson on a live call. Given the customer's "
            "latest statement below, identify their underlying need and sentiment, "
            "and suggest one talking point the salesperson could use next."
        ),
        description="sales / customer service talking-point helper",
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
