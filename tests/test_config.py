"""Config validation, load/save roundtrip, and credential requirement rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicerecon import config


def test_defaults_load_when_no_file(tmp_path: Path):
    cfg = config.load(tmp_path / "nope.json")
    assert cfg["save_dir"] == config.DEFAULT_SAVE_DIR
    assert cfg["speech_silence_seconds"] == config.DEFAULT_SILENCE_SECONDS
    assert cfg["whisper_model_size"] == config.DEFAULT_WHISPER_SIZE


def test_load_merges_partial_config(tmp_path: Path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"speech_silence_seconds": 3.0, "input_device": "MyMic"}))
    cfg = config.load(path)
    assert cfg["speech_silence_seconds"] == 3.0
    assert cfg["input_device"] == "MyMic"
    # Defaults for anything not specified
    assert cfg["model"] == config.DEFAULT_MODEL


def test_load_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "cfg.json"
    path.write_text("{not: json")
    with pytest.raises(config.ConfigError, match="not valid JSON"):
        config.load(path)


def test_load_rejects_non_object_top_level(tmp_path: Path):
    path = tmp_path / "cfg.json"
    path.write_text("[]")
    with pytest.raises(config.ConfigError, match="must be a JSON object"):
        config.load(path)


def test_env_key_overrides_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"anthropic_api_key": "from-file"}))
    monkeypatch.setenv(config.ENV_API_KEY, "from-env")
    cfg = config.load(path)
    assert cfg["anthropic_api_key"] == "from-env"


def test_validate_rejects_zero_silence():
    cfg = dict(config.DEFAULTS, speech_silence_seconds=0)
    with pytest.raises(config.ConfigError, match="speech_silence_seconds"):
        config.validate_config(cfg)


def test_validate_rejects_non_numeric_silence():
    cfg = dict(config.DEFAULTS, speech_silence_seconds="abc")
    with pytest.raises(config.ConfigError, match="speech_silence_seconds"):
        config.validate_config(cfg)


def test_validate_rejects_empty_model():
    cfg = dict(config.DEFAULTS, model="")
    with pytest.raises(config.ConfigError, match="'model'"):
        config.validate_config(cfg)


def test_validate_accepts_empty_credentials():
    """Empty credentials pass validation; require_credentials_for_ai is the gate."""
    cfg = dict(config.DEFAULTS)  # all credentials empty
    config.validate_config(cfg)


def test_require_credentials_for_ai_names_all_missing():
    cfg = dict(config.DEFAULTS)
    with pytest.raises(config.ConfigError) as exc_info:
        config.require_credentials_for_ai(cfg)
    message = str(exc_info.value)
    assert "Anthropic" in message
    assert "Telegram" in message


def test_require_credentials_for_ai_passes_when_all_set():
    cfg = dict(
        config.DEFAULTS,
        anthropic_api_key="k",
        telegram_bot_token="t",
        telegram_chat_id="c",
    )
    config.require_credentials_for_ai(cfg)


def test_save_load_roundtrip(tmp_path: Path):
    path = tmp_path / "cfg.json"
    cfg = dict(
        config.DEFAULTS,
        save_dir=str(tmp_path / "archive"),
        speech_silence_seconds=2.0,
        anthropic_api_key="secret",
    )
    config.save(cfg, path)
    reloaded = config.load(path)
    assert reloaded["speech_silence_seconds"] == 2.0
    assert reloaded["anthropic_api_key"] == "secret"


DEVICES = ["Webcam Mic", "Headset Mic"]


def _answer(monkeypatch, text: str) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: text)


def test_ask_device_picks_by_number(monkeypatch):
    _answer(monkeypatch, "2")
    chosen = config._ask_device("microphone", "Microphones", DEVICES, "Webcam Mic", "")
    assert chosen == "Headset Mic"


def test_ask_device_accepts_a_typed_name(monkeypatch):
    _answer(monkeypatch, "BlackHole 2ch")
    chosen = config._ask_device("loopback", "Loopbacks", DEVICES, "Webcam Mic", "")
    assert chosen == "BlackHole 2ch"


def test_ask_device_blank_keeps_the_system_default(monkeypatch):
    """Blank stores "", so capture follows the default at run time."""
    _answer(monkeypatch, "")
    chosen = config._ask_device("microphone", "Microphones", DEVICES, "Webcam Mic", "")
    assert chosen == ""


def test_ask_device_blank_keeps_the_current_value(monkeypatch):
    _answer(monkeypatch, "")
    chosen = config._ask_device("microphone", "Microphones", DEVICES, "Webcam Mic", "Headset Mic")
    assert chosen == "Headset Mic"


def test_ask_device_zero_clears_a_pinned_device(monkeypatch):
    _answer(monkeypatch, "0")
    chosen = config._ask_device("microphone", "Microphones", DEVICES, "Webcam Mic", "Headset Mic")
    assert chosen == ""


def test_ask_device_keeps_a_digit_valued_current_as_a_name(monkeypatch):
    """Enter on a stored "2" keeps "2"; it is not re-read as list index 2."""
    _answer(monkeypatch, "")
    assert config._ask_device("microphone", "Microphones", DEVICES, "Webcam Mic", "2") == "2"


def test_ask_device_rejects_out_of_range_number(monkeypatch):
    _answer(monkeypatch, "9")
    with pytest.raises(config.WizardAborted, match="microphone"):
        config._ask_device("microphone", "Microphones", DEVICES, "Webcam Mic", "")


def test_ask_device_takes_a_digit_as_a_name_when_nothing_was_detected(monkeypatch):
    _answer(monkeypatch, "2")
    assert config._ask_device("microphone", "Microphones", [], "", "") == "2"


def test_ask_device_hints_the_system_default_when_unset(monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    config._ask_device("your own voice", "Microphones", DEVICES, "Webcam Mic", "")
    assert "[system default]" in prompts[0]


def test_ask_device_lists_the_default_marker(monkeypatch, capsys):
    _answer(monkeypatch, "")
    config._ask_device("microphone", "Microphones", DEVICES, "Headset Mic", "")
    out = capsys.readouterr().out
    assert "1) Webcam Mic" in out
    assert "2) Headset Mic  — default" in out


def test_config_path_uses_xdg(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.config_path() == tmp_path / "voicerecon" / "config.json"


def test_config_path_override_wins(tmp_path: Path):
    override = tmp_path / "custom.json"
    assert config.config_path(override) == override
