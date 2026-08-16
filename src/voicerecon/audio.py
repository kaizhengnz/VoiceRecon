"""Cross-platform dual-source audio capture.

Windows and Linux expose system-audio loopback as an input device
automatically (WASAPI loopback / PulseAudio monitor). macOS does not; the
user must install BlackHole and select it as the loopback input.

The capture threads run inside :class:`AudioSource` context managers.
Each source resamples to 16 kHz mono float32 and pushes numpy arrays to a
user-supplied callback along with a monotonic timestamp taken at the end
of the block. The callback runs on the capture thread, so it must be
fast; downstream work (VAD, STT) is off-loaded via queues.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from . import vad

BLOCK_MS = 32
"""Callback block size in milliseconds. Matches Silero's 32 ms frames so
the VAD does not have to buffer partial frames."""

BLOCK_SAMPLES = int(vad.TARGET_SAMPLE_RATE * BLOCK_MS / 1000)


SampleCallback = Callable[[np.ndarray, float], None]


class AudioSource:
    """One capture stream running on its own soundcard recorder thread.

    Construction picks the device (default mic, or default speaker's
    loopback); the recorder starts on :meth:`open` and stops on
    :meth:`close` or context-manager exit.
    """

    def __init__(
        self,
        *,
        kind: str,
        device_name: str | None,
        callback: SampleCallback,
        recorder_factory: Callable[..., Any] | None = None,
    ) -> None:
        """``kind`` is ``"mic"`` or ``"loopback"``; ``device_name`` overrides
        the auto-picked default (used for macOS + BlackHole). ``recorder_factory``
        is a seam for tests: a callable returning a context manager that
        yields an object with ``.record(numframes=...)``."""
        if kind not in ("mic", "loopback"):
            raise ValueError(f"kind must be 'mic' or 'loopback', got {kind!r}")
        self.kind = kind
        self.device_name = device_name
        self._callback = callback
        self._recorder_factory = recorder_factory
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._reported_failure = False

    def open(self) -> None:
        """Start the capture thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"audio-{self.kind}", daemon=True
        )
        self._thread.start()

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self) -> AudioSource:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _run(self) -> None:
        recorder_cm = self._resolve_recorder()
        with recorder_cm as recorder:
            while not self._stop.is_set():
                try:
                    block = recorder.record(numframes=BLOCK_SAMPLES)
                    block = _to_mono_float32(block)
                    if block.size == 0:
                        continue
                    self._callback(block, time.monotonic())
                except Exception as exc:
                    # A single bad frame (Silero shape mismatch, transient
                    # audio glitch) must not kill this capture thread — the
                    # runner has no other way to notice, and it would silently
                    # stop transcribing this side of the conversation. Log
                    # once, then keep going.
                    from . import ui
                    if not self._reported_failure:
                        ui.warn(f"{self.kind} capture error ({type(exc).__name__}): {exc}")
                        self._reported_failure = True

    def _resolve_recorder(self) -> Any:
        if self._recorder_factory is not None:
            return self._recorder_factory(
                kind=self.kind,
                device_name=self.device_name,
                samplerate=vad.TARGET_SAMPLE_RATE,
                channels=1,
            )
        return _soundcard_recorder(
            kind=self.kind,
            device_name=self.device_name,
            samplerate=vad.TARGET_SAMPLE_RATE,
        )


def _to_mono_float32(block: np.ndarray) -> np.ndarray:
    if block.size == 0:
        return block.astype(np.float32, copy=False)
    if block.ndim > 1:
        block = block.mean(axis=1)
    return np.asarray(block, dtype=np.float32)


def _soundcard_recorder(
    *, kind: str, device_name: str | None, samplerate: int
) -> Any:
    """Return a context manager producing a soundcard recorder object.

    Loopback on Windows and Linux is exposed by ``all_microphones(include_loopback=True)``;
    the recorder for the default speaker's monitor is picked by
    :func:`_pick_loopback` when ``device_name`` is None.
    """
    import soundcard as sc  # type: ignore[import-not-found]

    if kind == "mic":
        mic = sc.get_microphone(device_name) if device_name else sc.default_microphone()
    else:
        mic = _pick_loopback(sc, device_name)
    return mic.recorder(samplerate=samplerate, channels=1)


def _pick_loopback(sc: Any, device_name: str | None) -> Any:
    """Return a soundcard microphone object representing a loopback source.

    With an explicit ``device_name`` the caller wins (macOS + BlackHole);
    otherwise pick the loopback that corresponds to the default speaker,
    which is what a fresh Windows / Linux user expects.
    """
    if device_name:
        return sc.get_microphone(device_name, include_loopback=True)

    default_speaker = sc.default_speaker()
    return sc.get_microphone(default_speaker.name, include_loopback=True)


def enumerate_devices() -> dict[str, Any]:
    """List currently visible input and loopback device names.

    Keys: ``input`` and ``loopback`` hold the name lists, ``default_input``
    and ``default_loopback`` name the entry an empty config field resolves
    to (the same choices :func:`_soundcard_recorder` and
    :func:`_pick_loopback` make), or ``""`` when that cannot be determined.

    Used by ``--configure`` and the ``--show-devices`` diagnostic to help
    the user pick the right one. Never raises: import / query failures
    yield empty values rather than crashing the wizard.
    """
    empty: dict[str, Any] = {
        "input": [],
        "loopback": [],
        "default_input": "",
        "default_loopback": "",
    }
    try:
        import soundcard as sc  # type: ignore[import-not-found]
    except ImportError:
        return empty

    try:
        mics = sc.all_microphones(include_loopback=False)
        loopbacks = sc.all_microphones(include_loopback=True)
    except Exception:
        return empty

    mic_names = [m.name for m in mics]
    loopback_names = [m.name for m in loopbacks if m.name not in mic_names]
    return {
        "input": mic_names,
        "loopback": loopback_names,
        "default_input": _default_device_name(sc.default_microphone),
        "default_loopback": _default_device_name(sc.default_speaker),
    }


def _default_device_name(getter: Callable[[], Any]) -> str:
    """Name reported by a soundcard default-device getter, ``""`` on failure.

    A machine with no default recording device raises rather than returning
    None, and that must not take the whole listing down.
    """
    try:
        return str(getter().name)
    except Exception:
        return ""


def format_device_lines(names: list[str], default_name: str) -> list[str]:
    """Numbered display lines, marking the entry blank input resolves to.

    Shared by the wizard and ``--show-devices`` so both number the same way.
    """
    return [
        f"{index}) {name}" + ("  — default" if name == default_name else "")
        for index, name in enumerate(names, start=1)
    ]
