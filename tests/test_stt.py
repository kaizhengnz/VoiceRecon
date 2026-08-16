"""Device and compute-type resolution for the Whisper backend."""

from __future__ import annotations

import numpy as np

from voicerecon import stt


class FakeModel:
    """Records how it was constructed; transcribes to a fixed string."""

    def __init__(self, model_size, *, device, compute_type):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio, language=None, vad_filter=False):
        return [type("Seg", (), {"text": "hello"})()], None


def _loaded_model(monkeypatch, cuda_devices, **kwargs):
    """Return the FakeModel a Transcriber builds under ``cuda_devices`` GPUs."""
    monkeypatch.setattr(stt, "_cuda_device_count", lambda: cuda_devices)
    built = []

    def factory(model_size, **factory_kwargs):
        model = FakeModel(model_size, **factory_kwargs)
        built.append(model)
        return model

    transcriber = stt.Transcriber(model_factory=factory, **kwargs)
    transcriber.transcribe(np.zeros(160, dtype=np.float32))
    return built[0]


def test_auto_picks_cuda_and_float16_when_a_gpu_is_visible(monkeypatch):
    model = _loaded_model(monkeypatch, 1)
    assert model.device == "cuda"
    assert model.compute_type == "float16"


def test_auto_falls_back_to_cpu_and_int8_without_a_gpu(monkeypatch):
    model = _loaded_model(monkeypatch, 0)
    assert model.device == "cpu"
    assert model.compute_type == "int8"


def test_explicit_device_is_not_overridden(monkeypatch):
    model = _loaded_model(monkeypatch, 1, device="cpu")
    assert model.device == "cpu"
    assert model.compute_type == "int8"


def test_explicit_compute_type_wins_over_the_device_default(monkeypatch):
    model = _loaded_model(monkeypatch, 1, compute_type="int8_float16")
    assert model.device == "cuda"
    assert model.compute_type == "int8_float16"


def test_transcribe_returns_empty_string_on_backend_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("no model")

    transcriber = stt.Transcriber(model_factory=boom)
    assert transcriber.transcribe(np.zeros(160, dtype=np.float32)) == ""
