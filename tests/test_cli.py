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
