"""Platform notes for audio capture.

Windows and Linux expose system-audio loopback devices natively (WASAPI
loopback on Windows, PulseAudio / PipeWire monitor sources on Linux) so
system-audio capture works after ``pip install`` with no extra steps.

macOS does not: CoreAudio has no built-in loopback. The user must install
BlackHole (or a similar virtual audio driver) and select it as the input
device. This module owns the message that tells them so, in one place, so
every callsite that needs it says the same thing.
"""

from __future__ import annotations

import sys

MACOS_LOOPBACK_HINT = (
    "macOS does not expose system audio to third-party apps directly. "
    "Install BlackHole (https://existential.audio/blackhole/) and select it "
    "as the loopback input device. See README for the setup steps."
)


def is_macos() -> bool:
    return sys.platform == "darwin"


def loopback_hint() -> str | None:
    """Return a one-line hint to print at startup, or None if none is needed."""
    if is_macos():
        return MACOS_LOOPBACK_HINT
    return None
