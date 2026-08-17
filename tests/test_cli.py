"""CLI argument parsing and top-level dispatch."""

from __future__ import annotations

import pytest

from voicerecon import cli


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    assert "voicerecon" in capsys.readouterr().out


def test_version_prints_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    from voicerecon import __version__
    assert __version__ in capsys.readouterr().out


def test_presets_flag_lists_all_built_ins(capsys):
    rc = cli.main(["--presets"])
    assert rc == 0
    out = capsys.readouterr().out
    for name in (
        "interview_candidate",
        "interview_recruiter",
        "meeting_summary",
    ):
        assert name in out


def test_configure_and_show_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--configure", "--show"])


def test_listen_with_show_rejects(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--listen", "interview_candidate", "--show"])


def test_show_reports_missing_config(tmp_path, capsys):
    missing = tmp_path / "no.json"
    rc = cli.main(["--show", "--config", str(missing)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "No config" in (captured.err + captured.out)


def test_listen_and_prompt_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--listen", "interview_candidate", "--prompt", "translate"])


def test_prompt_with_show_rejects(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--prompt", "translate", "--show"])


def test_from_without_prompt_rejects(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--from", "them"])


def test_context_without_prompt_rejects(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--context", "current"])


def test_from_with_invalid_speaker_rejects(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--prompt", "translate", "--from", "everybody"])


def _write_valid_config(tmp_path, *, with_credentials: bool = True) -> str:
    import json

    payload = {
        "save_dir": str(tmp_path / "voicerecon"),
        "speech_silence_seconds": 1.5,
        "whisper_model_size": "small",
        "input_device": "",
        "loopback_device": "",
        "model": "claude-haiku-4-5",
        "anthropic_api_key": "key" if with_credentials else "",
        "telegram_bot_token": "token" if with_credentials else "",
        "telegram_chat_id": "chat" if with_credentials else "",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_prompt_builds_custom_preset_and_calls_runner(tmp_path, monkeypatch, capsys):
    config_path = _write_valid_config(tmp_path)
    captured: dict = {}

    def fake_run(cfg, preset):
        captured["cfg"] = cfg
        captured["preset"] = preset
        return 0

    monkeypatch.setattr("voicerecon.runner.run", fake_run)

    rc = cli.main(
        ["--prompt", "Translate to English", "--from", "me", "--context", "window:60", "--config", config_path]
    )
    assert rc == 0
    preset = captured["preset"]
    assert preset is not None
    assert preset.name == "custom"
    assert preset.speaker_filter == "me"
    assert preset.context == "window:60"
    assert preset.prompt == "Translate to English"
    assert preset.is_batch is False


def test_prompt_uses_default_filter_and_context(tmp_path, monkeypatch):
    config_path = _write_valid_config(tmp_path)
    captured: dict = {}

    def fake_run(cfg, preset):
        captured["preset"] = preset
        return 0

    monkeypatch.setattr("voicerecon.runner.run", fake_run)
    rc = cli.main(["--prompt", "Do X", "--config", config_path])
    assert rc == 0
    preset = captured["preset"]
    assert preset.speaker_filter == "them"
    assert preset.context == "current"


def test_prompt_rejects_empty_text(tmp_path, monkeypatch, capsys):
    config_path = _write_valid_config(tmp_path)
    monkeypatch.setattr("voicerecon.runner.run", lambda cfg, preset: 0)
    rc = cli.main(["--prompt", "   ", "--config", config_path])
    assert rc == 1
    captured = capsys.readouterr()
    assert "non-empty" in (captured.err + captured.out)


def test_prompt_rejects_invalid_context(tmp_path, monkeypatch, capsys):
    config_path = _write_valid_config(tmp_path)
    monkeypatch.setattr("voicerecon.runner.run", lambda cfg, preset: 0)
    rc = cli.main(["--prompt", "Do X", "--context", "nonsense", "--config", config_path])
    assert rc == 1


def test_prompt_requires_credentials(tmp_path, monkeypatch, capsys):
    config_path = _write_valid_config(tmp_path, with_credentials=False)
    monkeypatch.setattr("voicerecon.runner.run", lambda cfg, preset: 0)
    rc = cli.main(["--prompt", "Do X", "--config", config_path])
    assert rc == 1
    err_and_out = capsys.readouterr()
    assert "API key" in (err_and_out.err + err_and_out.out) or "credentials" in (err_and_out.err + err_and_out.out).lower()
