"""Device enumeration and its display formatting.

Real capture hardware is never touched: a fake ``soundcard`` module is
installed in ``sys.modules`` so enumeration runs against known names.
"""

from __future__ import annotations

import builtins
import sys
import types

from voicerecon import audio


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
