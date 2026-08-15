"""Transcript file writing."""

from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path

from voicerecon import transcript


def test_no_writes_no_file(tmp_path: Path):
    writer = transcript.TranscriptWriter(str(tmp_path))
    # Never called append — no file should exist
    assert writer.path is None
    assert list(tmp_path.iterdir()) == []


def test_append_creates_file_and_writes_line(tmp_path: Path):
    writer = transcript.TranscriptWriter(str(tmp_path))
    writer.append("me", "hello world", when=datetime(2026, 1, 2, 3, 4, 5))

    assert writer.path is not None
    content = writer.path.read_text(encoding="utf-8")
    assert content == "[03:04:05] [me]: hello world\n"


def test_multiple_appends_stack(tmp_path: Path):
    writer = transcript.TranscriptWriter(str(tmp_path))
    writer.append("me", "first", when=datetime(2026, 1, 2, 3, 4, 5))
    writer.append("them", "second", when=datetime(2026, 1, 2, 3, 4, 10))

    content = writer.path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert lines == ["[03:04:05] [me]: first", "[03:04:10] [them]: second"]


def test_empty_text_is_skipped(tmp_path: Path):
    writer = transcript.TranscriptWriter(str(tmp_path))
    writer.append("me", "   ")
    assert writer.path is None


def test_filename_matches_expected_pattern(tmp_path: Path):
    writer = transcript.TranscriptWriter(str(tmp_path))
    writer.append("me", "x")
    assert re.fullmatch(r"transcript-\d{8}-\d{6}\.txt", writer.path.name)


def test_writes_are_thread_safe(tmp_path: Path):
    writer = transcript.TranscriptWriter(str(tmp_path))

    def worker(speaker: str):
        for _ in range(50):
            writer.append(speaker, "x")

    threads = [
        threading.Thread(target=worker, args=(speaker,))
        for speaker in ("me", "them")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = writer.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    # Every line is well-formed
    for line in lines:
        assert re.fullmatch(r"\[\d{2}:\d{2}:\d{2}\] \[(me|them)\]: x", line)
