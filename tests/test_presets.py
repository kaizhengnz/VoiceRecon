"""Built-in preset registry."""

from __future__ import annotations

import pytest

from voicerecon import presets


def test_all_expected_presets_exist():
    expected = {
        "interview_candidate",
        "interview_recruiter",
        "meeting_summary",
    }
    assert set(presets.names()) == expected


def test_get_returns_a_preset():
    p = presets.get("interview_candidate")
    assert p.name == "interview_candidate"
    assert p.speaker_filter == "them"
    assert p.context == "current"
    assert p.prompt


def test_get_unknown_raises_with_helpful_message():
    with pytest.raises(KeyError) as exc_info:
        presets.get("nope")
    assert "meeting_summary" in str(exc_info.value)


def test_speaker_filters_are_valid():
    allowed = {"both", "them", "me"}
    for name in presets.names():
        assert presets.get(name).speaker_filter in allowed


def test_context_specs_parse():
    from voicerecon import context

    for name in presets.names():
        # Should not raise
        context.parse(presets.get(name).context)


def test_interview_recruiter_uses_window_context():
    p = presets.get("interview_recruiter")
    assert p.context.startswith("window:")


def test_per_segment_presets_default_trigger():
    for name in ("interview_candidate", "interview_recruiter"):
        assert presets.get(name).trigger == "per_segment"


def test_meeting_summary_is_on_shutdown_with_both_speakers():
    p = presets.get("meeting_summary")
    assert p.trigger == "on_shutdown"
    assert p.speaker_filter == "both"
