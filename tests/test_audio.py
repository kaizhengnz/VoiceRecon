"""Device enumeration, its display formatting, and mic-recorder wiring.

Real capture hardware is never touched: a fake ``soundcard`` module is
installed in ``sys.modules`` so enumeration runs against known names, and
a fake ``sounddevice`` module drives the mic recorder tests.
"""

from __future__ import annotations

import builtins
import sys
import threading
import types

import numpy as np

from voicerecon import audio, ui


class _FakeMic:
    def __init__(self, name: str):
        self.name = name


def _fake_soundcard(
    *, mics: list[str], loopbacks: list[str], default_mic: str, default_spk: str
) -> types.ModuleType:
    module = types.ModuleType("soundcard")

    def all_microphones(include_loopback: bool = False):
        names = mics + loopbacks if include_loopback else mics
        return [_FakeMic(name) for name in names]

    module.all_microphones = all_microphones  # type: ignore[attr-defined]
    module.default_microphone = lambda: _FakeMic(default_mic)  # type: ignore[attr-defined]
    module.default_speaker = lambda: _FakeMic(default_spk)  # type: ignore[attr-defined]
    return module


def _install(monkeypatch, module: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, "soundcard", module)


def _hide_soundcard(monkeypatch, error: Exception) -> None:
    """Make ``import soundcard`` fail the way a broken install does."""
    monkeypatch.delitem(sys.modules, "soundcard", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "soundcard":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


EMPTY = {"input": [], "loopback": [], "default_input": "", "default_loopback": ""}


def test_enumerate_reports_defaults(monkeypatch):
    _install(
        monkeypatch,
        _fake_soundcard(
            mics=["Webcam Mic", "Headset Mic"],
            loopbacks=["Speakers (JBL)"],
            default_mic="Headset Mic",
            default_spk="Speakers (JBL)",
        ),
    )
    devices = audio.enumerate_devices()
    assert devices["input"] == ["Webcam Mic", "Headset Mic"]
    assert devices["loopback"] == ["Speakers (JBL)"]
    assert devices["default_input"] == "Headset Mic"
    assert devices["default_loopback"] == "Speakers (JBL)"


def test_enumerate_survives_missing_default_device(monkeypatch):
    module = _fake_soundcard(mics=["Webcam Mic"], loopbacks=[], default_mic="", default_spk="")

    def no_default():
        raise RuntimeError("no default recording device")

    module.default_microphone = no_default  # type: ignore[attr-defined]
    _install(monkeypatch, module)
    devices = audio.enumerate_devices()
    assert devices["input"] == ["Webcam Mic"]
    assert devices["default_input"] == ""


def test_enumerate_matches_a_pulseaudio_monitor_to_its_sink(monkeypatch):
    """PulseAudio names the monitor after the sink; the marker must still land."""
    _install(
        monkeypatch,
        _fake_soundcard(
            mics=["Webcam Mic"],
            loopbacks=["Monitor of Built-in Audio"],
            default_mic="Webcam Mic",
            default_spk="Built-in Audio",
        ),
    )
    assert audio.enumerate_devices()["default_loopback"] == "Monitor of Built-in Audio"


def test_enumerate_leaves_loopback_default_empty_when_nothing_matches(monkeypatch):
    _install(
        monkeypatch,
        _fake_soundcard(
            mics=["BlackHole 2ch"],
            loopbacks=[],
            default_mic="BlackHole 2ch",
            default_spk="MacBook Pro Speakers",
        ),
    )
    assert audio.enumerate_devices()["default_loopback"] == ""


def test_enumerate_survives_query_failure(monkeypatch):
    module = types.ModuleType("soundcard")

    def boom(include_loopback: bool = False):
        raise RuntimeError("COM failure")

    module.all_microphones = boom  # type: ignore[attr-defined]
    _install(monkeypatch, module)
    assert audio.enumerate_devices() == EMPTY


def test_enumerate_survives_soundcard_absent(monkeypatch):
    _hide_soundcard(monkeypatch, ImportError("no module named soundcard"))
    assert audio.enumerate_devices() == EMPTY


def test_enumerate_survives_soundcard_import_raising_oserror(monkeypatch):
    """No libpulse on Linux: the cffi dlopen raises OSError, not ImportError."""
    _hide_soundcard(monkeypatch, OSError("cannot load library 'libpulse.so.0'"))
    assert audio.enumerate_devices() == EMPTY


def test_format_device_lines_numbers_and_marks_default():
    lines = audio.format_device_lines(["Webcam Mic", "Headset Mic"], "Headset Mic")
    assert lines == ["1) Webcam Mic", "2) Headset Mic  — default"]


def test_format_device_lines_marks_nothing_when_default_unknown():
    assert audio.format_device_lines(["Webcam Mic"], "") == ["1) Webcam Mic"]


# --------------------------------------------------------------------------- #
# Mic recorder (sounddevice) tests
# --------------------------------------------------------------------------- #


class _FakeInputStream:
    def __init__(self, *, device, samplerate, channels, dtype):
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.entered = False
        self.exited = False
        self.reads: list[int] = []

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc_info):
        self.exited = True
        return False

    def read(self, frames):
        self.reads.append(frames)
        return np.zeros((frames, 1), dtype=np.float32), False


def _install_fake_sounddevice(monkeypatch, created: list[_FakeInputStream]) -> None:
    module = types.ModuleType("sounddevice")

    def InputStream(**kwargs):
        stream = _FakeInputStream(**kwargs)
        created.append(stream)
        return stream

    module.InputStream = InputStream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", module)


def test_mic_recorder_opens_default_input_when_no_device_name(monkeypatch):
    created: list[_FakeInputStream] = []
    _install_fake_sounddevice(monkeypatch, created)

    recorder_cm = audio._sounddevice_mic_recorder(device_name=None, samplerate=16000)
    with recorder_cm as recorder:
        block = recorder.record(numframes=512)

    assert len(created) == 1
    stream = created[0]
    assert stream.device is None
    assert stream.samplerate == 16000
    assert stream.channels == 1
    assert stream.dtype == "float32"
    assert stream.entered and stream.exited
    assert stream.reads == [512]
    assert block.shape == (512, 1)


def test_mic_recorder_passes_device_name_substring(monkeypatch):
    created: list[_FakeInputStream] = []
    _install_fake_sounddevice(monkeypatch, created)

    with audio._sounddevice_mic_recorder(
        device_name="Microphone Array (Realtek)", samplerate=16000
    ):
        pass

    assert created[0].device == "Microphone Array (Realtek)"


def test_mic_recorder_treats_empty_device_name_as_default(monkeypatch):
    """An empty ``input_device`` in config must not become an empty
    substring match (sounddevice would then match nothing)."""
    created: list[_FakeInputStream] = []
    _install_fake_sounddevice(monkeypatch, created)

    with audio._sounddevice_mic_recorder(device_name="", samplerate=16000):
        pass

    assert created[0].device is None


# --------------------------------------------------------------------------- #
# AudioSource error handling
# --------------------------------------------------------------------------- #


def test_audio_source_reports_when_recorder_creation_fails(monkeypatch):
    errors: list[str] = []
    monkeypatch.setattr(ui, "error", lambda msg: errors.append(msg))

    def boom(**kwargs):
        raise RuntimeError("no such device")

    src = audio.AudioSource(
        kind="mic",
        device_name=None,
        callback=lambda block, ts: None,
        recorder_factory=boom,
    )
    src.open()
    # The capture thread should exit almost immediately; close() joins it.
    src.close(timeout=2.0)

    assert any("mic capture failed to start" in msg for msg in errors), errors


def test_audio_source_delivers_blocks_from_fake_recorder():
    """Baseline: the injected factory drives the callback loop end-to-end."""
    received: list[tuple[int, float]] = []
    got_one = threading.Event()

    def callback(block, ts):
        received.append((block.shape[0], ts))
        got_one.set()

    class _FakeRecorder:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def record(self, numframes):
            return np.zeros((numframes, 1), dtype=np.float32)

    def factory(**kwargs):
        return _FakeRecorder()

    src = audio.AudioSource(
        kind="mic",
        device_name=None,
        callback=callback,
        recorder_factory=factory,
    )
    src.open()
    assert got_one.wait(timeout=2.0), "callback never fired"
    src.close(timeout=2.0)

    assert received[0][0] == audio.BLOCK_SAMPLES
