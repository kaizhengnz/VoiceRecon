"""Save directory helpers.

The transcript itself is written by :mod:`voicerecon.transcript`, which
opens the file in append mode. This module just owns the resolution and
creation of the directory that contains it, plus the same private-file
posture as ScreenRecon: on POSIX platforms the directory is created
owner-only so a shared machine does not leak audio transcripts.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700


def restrict(path: Path, mode: int) -> None:
    """Apply owner-only permissions on POSIX. No-op on Windows."""
    if os.name == "nt":
        return
    with contextlib.suppress(OSError):
        path.chmod(mode)


def make_private_dir(path: Path) -> Path:
    """Create a directory (with parents), tighten only if we just created it.

    A pre-existing directory belongs to the user's wider environment
    (``~/VoiceRecon`` may live under a shared ``~/Documents``); silently
    chmod'ing something we did not create would break unrelated tools.
    """
    created = not path.exists()
    path.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    if created:
        restrict(path, PRIVATE_DIR_MODE)
    return path


def normalise_dir(save_dir: str | os.PathLike[str]) -> Path:
    """Expand ``%VARS%`` and ``~``, trim whitespace, and make the path absolute."""
    text = os.path.expandvars(str(save_dir)).strip()
    return Path(text).expanduser().resolve()


def resolve_dir(save_dir: str | os.PathLike[str]) -> Path:
    """Normalise the path, create the directory owner-only, and return it."""
    return make_private_dir(normalise_dir(save_dir))
