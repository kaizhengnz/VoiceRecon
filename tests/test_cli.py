"""CLI argument parsing and top-level dispatch."""

from __future__ import annotations

import json

import pytest

from voicerecon import cli, config


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


def test_prompt_with_show_rejects(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--prompt", "translate", "--show"])


def test_bare_prompt_with_show_rejects(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--prompt", "--show"])


def test_bare_prompt_runs_setter(tmp_path, monkeypatch):
    called: dict = {}

    def fake_setter(path):
        called["path"] = path
        return 0

    monkeypatch.setattr("voicerecon.config.run_set_prompt", fake_setter)
    rc = cli.main(["--config", "some/path", "--prompt"])
    assert rc == 0
    assert called["path"] == "some/path"


def test_bare_prompt_refuses_when_no_config_exists(tmp_path, capsys):
    missing = tmp_path / "no.json"
    rc = cli.main(["--config", str(missing), "--prompt"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "No config" in (captured.err + captured.out)


def test_show_reports_missing_config(tmp_path, capsys):
    missing = tmp_path / "no.json"
    rc = cli.main(["--show", "--config", str(missing)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "No config" in (captured.err + captured.out)


def _write_valid_config(tmp_path, *, with_credentials: bool = True, **overrides) -> str:
    payload = dict(config.DEFAULTS, save_dir=str(tmp_path / "voicerecon"))
    if with_credentials:
        payload.update(
            anthropic_api_key="key",
            telegram_bot_token="token",
            telegram_chat_id="chat",
        )
    payload.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _record_run(monkeypatch) -> dict:
    captured: dict = {}

    def fake_run(cfg, preset):
        captured["cfg"] = cfg
        captured["preset"] = preset
        return 0

    monkeypatch.setattr("voicerecon.runner.run", fake_run)
    return captured


def test_cli_prompt_with_preset_name_uses_that_preset(tmp_path, monkeypatch):
    config_path = _write_valid_config(tmp_path)
    captured = _record_run(monkeypatch)
    rc = cli.main(["--prompt", "meeting_summary", "--config", config_path])
    assert rc == 0
    assert captured["preset"] is not None
    assert captured["preset"].name == "meeting_summary"
    assert captured["preset"].is_batch is True


def test_cli_prompt_with_free_text_becomes_custom(tmp_path, monkeypatch):
    config_path = _write_valid_config(tmp_path)
    captured = _record_run(monkeypatch)
    rc = cli.main(["--prompt", "Translate to English", "--config", config_path])
    assert rc == 0
    preset = captured["preset"]
    assert preset is not None
    assert preset.is_custom is True
    assert preset.prompt == "Translate to English"
    assert preset.speaker_filter == "them"
    assert preset.context == "current"


def test_cli_prompt_whitespace_becomes_transcript_only(tmp_path, monkeypatch):
    config_path = _write_valid_config(tmp_path)
    captured = _record_run(monkeypatch)
    rc = cli.main(["--prompt", "   ", "--config", config_path])
    assert rc == 0
    assert captured["preset"] is None


def test_cli_prompt_requires_credentials(tmp_path, monkeypatch, capsys):
    config_path = _write_valid_config(tmp_path, with_credentials=False)
    monkeypatch.setattr("voicerecon.runner.run", lambda cfg, preset: 0)
    rc = cli.main(["--prompt", "Do X", "--config", config_path])
    assert rc == 1
    captured = capsys.readouterr()
    assert "AI mode needs" in (captured.err + captured.out)


def test_no_flag_and_empty_cfg_prompt_is_transcript_only(tmp_path, monkeypatch):
    config_path = _write_valid_config(tmp_path, prompt="")
    captured = _record_run(monkeypatch)
    rc = cli.main(["--config", config_path])
    assert rc == 0
    assert captured["preset"] is None


def test_no_flag_uses_cfg_prompt_preset_name(tmp_path, monkeypatch):
    config_path = _write_valid_config(tmp_path, prompt="meeting_summary")
    captured = _record_run(monkeypatch)
    rc = cli.main(["--config", config_path])
    assert rc == 0
    assert captured["preset"].name == "meeting_summary"


def test_no_flag_uses_cfg_prompt_custom_text(tmp_path, monkeypatch):
    config_path = _write_valid_config(tmp_path, prompt="Summarise the last 30 seconds.")
    captured = _record_run(monkeypatch)
    rc = cli.main(["--config", config_path])
    assert rc == 0
    preset = captured["preset"]
    assert preset.is_custom is True
    assert preset.prompt == "Summarise the last 30 seconds."


def test_cli_prompt_overrides_cfg_prompt(tmp_path, monkeypatch):
    config_path = _write_valid_config(tmp_path, prompt="meeting_summary")
    captured = _record_run(monkeypatch)
    rc = cli.main(["--prompt", "interview_candidate", "--config", config_path])
    assert rc == 0
    assert captured["preset"].name == "interview_candidate"


def test_cfg_prompt_trigger_overrides_built_in(tmp_path, monkeypatch):
    """cfg["prompt_trigger"] flips a built-in preset's trigger at runtime."""
    config_path = _write_valid_config(
        tmp_path, prompt="meeting_summary", prompt_trigger="per_segment"
    )
    captured = _record_run(monkeypatch)
    rc = cli.main(["--config", config_path])
    assert rc == 0
    assert captured["preset"].name == "meeting_summary"
    assert captured["preset"].trigger == "per_segment"


def test_cfg_prompt_trigger_applies_to_custom(tmp_path, monkeypatch):
    config_path = _write_valid_config(
        tmp_path, prompt="Recap the meeting.", prompt_trigger="on_shutdown"
    )
    captured = _record_run(monkeypatch)
    rc = cli.main(["--config", config_path])
    assert rc == 0
    preset = captured["preset"]
    assert preset.is_custom is True
    assert preset.trigger == "on_shutdown"
    assert preset.speaker_filter == "both"


def test_cfg_prompt_requires_credentials(tmp_path, monkeypatch, capsys):
    config_path = _write_valid_config(
        tmp_path, with_credentials=False, prompt="meeting_summary"
    )
    monkeypatch.setattr("voicerecon.runner.run", lambda cfg, preset: 0)
    rc = cli.main(["--config", config_path])
    assert rc == 1
    captured = capsys.readouterr()
    assert "AI mode needs" in (captured.err + captured.out)
