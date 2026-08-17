"""Local transcript file writer.

One file per session, named ``transcript-YYYYMMDD-HHMMSS.txt`` and living
under the configured ``save_dir``. Segments are appended immediately after
each completed utterance, so a crash mid-session still leaves a readable
partial transcript.

Line format: ``[HH:MM:SS] [them|me]: <text>``.

Thread-safe: writes go through an internal lock so any caller (whether the
main pipeline thread or a future direct producer) can append without
coordinating externally.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from . import storage, ui

FILENAME_STEM_FORMAT = "transcript-%Y%m%d-%H%M%S"
LINE_TIMESTAMP_FORMAT = "%H:%M:%S"


class TranscriptWriter:
    """Owns one transcript file for the duration of a listening session.

    Files are opened lazily on the first successful segment write, so a
    session that ends without any speech leaves no empty file behind.
    """

    def __init__(self, save_dir: str) -> None:
        self._save_dir = save_dir
        self._path: Path | None = None
        self._lock = threading.Lock()
        self._started_at = datetime.now()
        self._warned_about_error = False

    @property
    def path(self) -> Path | None:
        """The file being written, or None if no segment has been recorded yet."""
        return self._path

    def append(self, speaker: str, text: str, *, when: datetime | None = None) -> None:
        """Append one segment. Errors are printed once and never raised."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        stamp = (when or datetime.now()).strftime(LINE_TIMESTAMP_FORMAT)
        line = f"[{stamp}] [{speaker}]: {cleaned}\n"

        with self._lock:
            path = self._ensure_path()
            if path is None:
                return
            try:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as exc:
                if not self._warned_about_error:
                    ui.warn(f"Transcript write failed ({path}): {exc.strerror or exc}")
                    self._warned_about_error = True

    def _ensure_path(self) -> Path | None:
        if self._path is not None:
            return self._path
        stem = self._started_at.strftime(FILENAME_STEM_FORMAT)
        try:
            path = storage.new_private_file(self._save_dir, stem)
        except (OSError, RuntimeError) as exc:
            if not self._warned_about_error:
                ui.warn(f"Cannot create transcript file: {exc}")
                self._warned_about_error = True
            return None
        self._path = path
        return path
