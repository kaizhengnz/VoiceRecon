"""Segmenter cutting logic: silence threshold + speaker-change preemption."""

from __future__ import annotations

import numpy as np

from voicerecon import segmenter, vad


def _samples(n: int, value: float = 0.1) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def _event(kind: str, ts: float) -> vad.SpeechEvent:
    return vad.SpeechEvent(kind=kind, timestamp=ts)


def test_no_events_produces_no_segments():
    seg = segmenter.Segmenter()
    seg.on_audio("me", _samples(100))
    assert seg.drain() == []


def test_start_then_end_emits_one_segment():
    seg = segmenter.Segmenter()
    seg.on_events("me", [_event("start", 1.0)])
    seg.on_audio("me", _samples(200))
    seg.on_events("me", [_event("end", 3.0)])

    segments = seg.drain()
    assert len(segments) == 1
    assert segments[0].speaker == "me"
    assert segments[0].audio.size == 200
    assert segments[0].started_at == 1.0
    assert segments[0].ended_at == 3.0


def test_audio_before_start_is_dropped():
    seg = segmenter.Segmenter()
    seg.on_audio("me", _samples(100))  # dropped — no active segment yet
    seg.on_events("me", [_event("start", 1.0)])
    seg.on_audio("me", _samples(200))
    seg.on_events("me", [_event("end", 2.0)])

    segments = seg.drain()
    assert segments[0].audio.size == 200


def test_speaker_change_preempts_current_segment():
    seg = segmenter.Segmenter()
    seg.on_events("them", [_event("start", 1.0)])
    seg.on_audio("them", _samples(300))
    # 'me' starts talking mid-utterance — should cut them's segment now
    seg.on_events("me", [_event("start", 2.0)])
    seg.on_audio("me", _samples(150))
    seg.on_events("me", [_event("end", 4.0)])

    segments = seg.drain()
    assert [s.speaker for s in segments] == ["them", "me"]
    assert segments[0].audio.size == 300
    assert segments[0].ended_at == 2.0  # cut at the speaker-change moment
    assert segments[1].audio.size == 150
    assert segments[1].ended_at == 4.0


def test_speaker_start_with_no_prior_active_does_not_emit():
    seg = segmenter.Segmenter()
    seg.on_events("them", [_event("start", 1.0)])
    # No audio yet; me's start preempts an empty buffer, should not emit
    seg.on_events("me", [_event("start", 2.0)])
    assert seg.drain() == []


def test_flush_cuts_open_segments():
    seg = segmenter.Segmenter()
    seg.on_events("me", [_event("start", 1.0)])
    seg.on_audio("me", _samples(100))
    # No 'end' event — session was killed mid-utterance
    remaining = seg.flush(timestamp=5.0)
    assert len(remaining) == 1
    assert remaining[0].ended_at == 5.0


def test_end_without_active_does_nothing():
    seg = segmenter.Segmenter()
    seg.on_events("me", [_event("end", 1.0)])
    assert seg.drain() == []


def test_drain_clears_the_queue():
    seg = segmenter.Segmenter()
    seg.on_events("me", [_event("start", 1.0)])
    seg.on_audio("me", _samples(50))
    seg.on_events("me", [_event("end", 2.0)])
    assert len(seg.drain()) == 1
    assert seg.drain() == []


def test_multiple_events_in_one_call_are_ordered():
    seg = segmenter.Segmenter()
    seg.on_events("them", [_event("start", 1.0)])
    seg.on_audio("them", _samples(100))
    # end + start in the same batch: cut the current, open a new one
    seg.on_events("them", [_event("end", 2.0), _event("start", 3.0)])
    seg.on_audio("them", _samples(50))
    seg.on_events("them", [_event("end", 4.0)])

    segments = seg.drain()
    assert len(segments) == 2
    assert segments[0].started_at == 1.0 and segments[0].ended_at == 2.0
    assert segments[0].audio.size == 100
    assert segments[1].started_at == 3.0 and segments[1].ended_at == 4.0
    assert segments[1].audio.size == 50


def test_empty_utterance_is_dropped():
    """start immediately followed by end with no audio in between yields nothing.

    An "utterance" without any audio samples is not useful downstream; the
    segmenter drops it rather than emitting a zero-length ReadySegment.
    """
    seg = segmenter.Segmenter()
    seg.on_events("me", [_event("start", 1.0), _event("end", 2.0)])
    assert seg.drain() == []
