"""Segmenter cutting logic: silence threshold + loopback-priority preemption."""

from __future__ import annotations

import numpy as np

from voicerecon import segmenter, vad


def _samples(n: int, value: float = 0.1) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def _event(kind: str, ts: float) -> vad.SpeechEvent:
    return vad.SpeechEvent(kind=kind, timestamp=ts)


class RecordingSink:
    """Records ``(speaker, sample_count)`` for each accepted audio block."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, speaker: str, samples: np.ndarray) -> None:
        self.calls.append((speaker, int(samples.size)))

    def totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for speaker, size in self.calls:
            totals[speaker] = totals.get(speaker, 0) + size
        return totals


def test_no_events_produces_no_segments():
    seg = segmenter.Segmenter()
    seg.on_audio("me", _samples(100))
    assert seg.drain() == []


def test_start_then_end_emits_one_segment_and_routes_audio():
    sink = RecordingSink()
    seg = segmenter.Segmenter(on_stream_audio=sink)
    seg.on_events("me", [_event("start", 1.0)])
    seg.on_audio("me", _samples(200))
    seg.on_events("me", [_event("end", 3.0)])

    segments = seg.drain()
    assert len(segments) == 1
    assert segments[0].speaker == "me"
    assert segments[0].started_at == 1.0
    assert segments[0].ended_at == 3.0
    assert sink.totals() == {"me": 200}


def test_audio_before_start_is_dropped():
    sink = RecordingSink()
    seg = segmenter.Segmenter(on_stream_audio=sink)
    seg.on_audio("me", _samples(100))  # dropped — no active segment yet
    seg.on_events("me", [_event("start", 1.0)])
    seg.on_audio("me", _samples(200))
    seg.on_events("me", [_event("end", 2.0)])

    assert len(seg.drain()) == 1
    assert sink.totals() == {"me": 200}


def test_loopback_start_cuts_pending_mic_segment():
    sink = RecordingSink()
    seg = segmenter.Segmenter(on_stream_audio=sink)
    seg.on_events("me", [_event("start", 1.0)])
    seg.on_audio("me", _samples(150))
    # Loopback (them) starts mid-mic — should cut mic and start loopback.
    seg.on_events("them", [_event("start", 2.0)])
    seg.on_audio("them", _samples(300))
    seg.on_events("them", [_event("end", 4.0)])

    segments = seg.drain()
    assert [s.speaker for s in segments] == ["me", "them"]
    assert segments[0].ended_at == 2.0
    assert segments[1].ended_at == 4.0
    assert sink.totals() == {"me": 150, "them": 300}


def test_mic_start_does_not_cut_loopback():
    """The reverse of the loopback→mic preemption: mic starting during a
    loopback segment must not disturb loopback."""
    sink = RecordingSink()
    seg = segmenter.Segmenter(on_stream_audio=sink)
    seg.on_events("them", [_event("start", 1.0)])
    seg.on_audio("them", _samples(300))
    seg.on_events("me", [_event("start", 2.0)])  # dropped: loopback active
    seg.on_audio("me", _samples(150))            # dropped: loopback active
    seg.on_events("them", [_event("end", 4.0)])  # closes the loopback segment

    segments = seg.drain()
    assert [s.speaker for s in segments] == ["them"]
    assert segments[0].ended_at == 4.0
    assert sink.totals() == {"them": 300}  # mic audio dropped


def test_mic_events_dropped_while_loopback_active():
    sink = RecordingSink()
    seg = segmenter.Segmenter(on_stream_audio=sink)
    seg.on_events("them", [_event("start", 1.0)])
    seg.on_audio("them", _samples(100))
    # A full start/end pair on mic during loopback is silently ignored.
    seg.on_events("me", [_event("start", 1.5)])
    seg.on_audio("me", _samples(50))
    seg.on_events("me", [_event("end", 1.8)])
    seg.on_events("them", [_event("end", 2.0)])

    segments = seg.drain()
    assert [s.speaker for s in segments] == ["them"]
    assert sink.totals() == {"them": 100}


def test_mic_captured_after_loopback_ends():
    sink = RecordingSink()
    seg = segmenter.Segmenter(on_stream_audio=sink)
    seg.on_events("them", [_event("start", 1.0), _event("end", 2.0)])
    seg.on_audio("them", _samples(50))  # dropped — them segment already closed
    seg.on_events("me", [_event("start", 3.0)])
    seg.on_audio("me", _samples(200))
    seg.on_events("me", [_event("end", 4.0)])

    segments = seg.drain()
    speakers = [s.speaker for s in segments]
    assert "me" in speakers
    assert sink.totals() == {"me": 200}


def test_flush_cuts_open_segments():
    sink = RecordingSink()
    seg = segmenter.Segmenter(on_stream_audio=sink)
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
    sink = RecordingSink()
    seg = segmenter.Segmenter(on_stream_audio=sink)
    seg.on_events("them", [_event("start", 1.0)])
    seg.on_audio("them", _samples(100))
    # end + start in the same batch: cut the current, open a new one
    seg.on_events("them", [_event("end", 2.0), _event("start", 3.0)])
    seg.on_audio("them", _samples(50))
    seg.on_events("them", [_event("end", 4.0)])

    segments = seg.drain()
    assert len(segments) == 2
    assert segments[0].started_at == 1.0 and segments[0].ended_at == 2.0
    assert segments[1].started_at == 3.0 and segments[1].ended_at == 4.0
    assert sink.totals() == {"them": 150}


def test_empty_utterance_is_dropped():
    """start immediately followed by end with no audio in between yields nothing.

    An "utterance" without any audio samples is not useful downstream; the
    segmenter drops it rather than emitting a zero-length ReadySegment.
    """
    seg = segmenter.Segmenter()
    seg.on_events("me", [_event("start", 1.0), _event("end", 2.0)])
    assert seg.drain() == []


def test_sink_runs_outside_the_lock():
    """A slow sink must not block a second on_audio call on another thread.

    We prove it indirectly: from inside the sink we call on_audio again
    on the *same* thread. If the lock were still held, that call would
    deadlock; instead it re-enters the priority filter cleanly.
    """
    seen: list[str] = []

    def sink(speaker: str, samples: np.ndarray) -> None:
        seen.append(speaker)
        if speaker == "them" and len(seen) == 1:
            # Send a mic block while loopback is active — must be dropped
            # without deadlocking on our own lock.
            seg.on_audio("me", _samples(10))

    seg = segmenter.Segmenter(on_stream_audio=sink)
    seg.on_events("them", [_event("start", 1.0)])
    seg.on_audio("them", _samples(100))
    seg.on_events("them", [_event("end", 2.0)])
    assert seen == ["them"]  # mic re-entry was filtered out


def test_sink_is_optional():
    """Segmenter works without a sink — audio is silently dropped."""
    seg = segmenter.Segmenter()
    seg.on_events("me", [_event("start", 1.0)])
    seg.on_audio("me", _samples(200))
    seg.on_events("me", [_event("end", 2.0)])
    assert len(seg.drain()) == 1
