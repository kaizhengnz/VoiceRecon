"""Context spec parsing and window assembly."""

from __future__ import annotations

import pytest

from voicerecon import context


def test_parse_current():
    kind, window = context.parse("current")
    assert kind == "current" and window == 0.0


def test_parse_window():
    kind, window = context.parse("window:300")
    assert kind == "window" and window == 300.0


def test_parse_window_accepts_float():
    kind, window = context.parse("window:1.5")
    assert kind == "window" and window == 1.5


def test_parse_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        context.parse("window:0")


def test_parse_rejects_unknown_spec():
    with pytest.raises(ValueError):
        context.parse("session")


def test_parse_rejects_non_numeric_window():
    with pytest.raises(ValueError):
        context.parse("window:abc")


def _seg(speaker: str, text: str, end: float) -> context.Segment:
    return context.Segment(speaker=speaker, text=text, end=end)


def test_assemble_current_returns_only_trigger():
    history = [_seg("them", "old", 10), _seg("them", "trigger", 20)]
    trigger = history[-1]
    result = context.assemble("current", history, trigger, "them")
    assert result == [trigger]


def test_assemble_window_keeps_segments_within_range():
    trigger = _seg("them", "trigger", 100)
    history = [
        _seg("them", "way old", 10),  # outside window
        _seg("them", "recent", 90),
        _seg("them", "very recent", 95),
        trigger,
    ]
    result = context.assemble("window:20", history, trigger, "them")
    assert [s.text for s in result] == ["recent", "very recent", "trigger"]


def test_assemble_window_filters_by_speaker():
    trigger = _seg("them", "trigger", 100)
    history = [
        _seg("me", "my part", 90),  # dropped by filter=them
        _seg("them", "their part", 95),
        trigger,
    ]
    result = context.assemble("window:60", history, trigger, "them")
    assert [s.text for s in result] == ["their part", "trigger"]


def test_assemble_window_speaker_both_keeps_everything_in_range():
    trigger = _seg("them", "trigger", 100)
    history = [
        _seg("me", "my part", 90),
        _seg("them", "their part", 95),
        trigger,
    ]
    result = context.assemble("window:60", history, trigger, "both")
    assert [s.text for s in result] == ["my part", "their part", "trigger"]


def test_assemble_appends_trigger_when_missing_from_history():
    trigger = _seg("them", "trigger", 100)
    history = [_seg("them", "prior", 95)]
    result = context.assemble("window:60", history, trigger, "them")
    assert result[-1] is trigger


def test_render_produces_speaker_labeled_lines():
    segments = [_seg("them", "hi", 10), _seg("me", "hey", 20)]
    assert context.render(segments) == "[them]: hi\n[me]: hey"
