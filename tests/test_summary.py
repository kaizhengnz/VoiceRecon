"""End-of-session summary writer and shutdown orchestration."""

from __future__ import annotations

import pytest

from voicerecon import ai, context, presets, summary


@pytest.fixture
def meeting_preset():
    return presets.get("meeting_summary")


@pytest.fixture
def cfg(tmp_path):
    return {
        "save_dir": str(tmp_path),
        "model": "claude-haiku-4-5",
        "anthropic_api_key": "test-key",
        "telegram_bot_token": "test-token",
        "telegram_chat_id": "test-chat",
    }


def _reply(ok: bool, text: str) -> ai.Reply:
    return ai.Reply(ok=ok, text=text)


def test_write_creates_file(tmp_path):
    path = summary.write(str(tmp_path), "meeting_summary", "hello world")
    assert path is not None
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "hello world"
    assert path.name == "meeting_summary.txt"


def test_write_handles_collision(tmp_path):
    first = summary.write(str(tmp_path), "meeting_summary", "one")
    second = summary.write(str(tmp_path), "meeting_summary", "two")
    assert first is not None and second is not None
    assert first != second
    assert first.read_text(encoding="utf-8") == "one"
    assert second.read_text(encoding="utf-8") == "two"


def test_write_returns_none_when_directory_unusable(tmp_path, monkeypatch):
    def blow_up(save_dir):
        raise OSError("nope")

    monkeypatch.setattr(summary.storage, "resolve_dir", blow_up)
    assert summary.write(str(tmp_path), "meeting_summary", "x") is None


def test_render_and_deliver_skips_empty_history(cfg, meeting_preset, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        summary.ai, "ask_streaming", lambda *a, **k: calls.append(a) or _reply(True, "x")
    )
    monkeypatch.setattr(
        summary.notify, "send_text", lambda *a, **k: calls.append(a) or True
    )
    summary.render_and_deliver(cfg, meeting_preset, [], cfg["save_dir"])
    assert calls == []


def test_render_and_deliver_skips_when_filter_drops_all(
    cfg, monkeypatch
):
    them_only = presets.Preset(
        name="them_only",
        speaker_filter="them",
        context="current",
        prompt="p",
        description="d",
        trigger="on_shutdown",
    )
    calls: list[tuple] = []
    monkeypatch.setattr(
        summary.ai, "ask_streaming", lambda *a, **k: calls.append(a) or _reply(True, "x")
    )
    monkeypatch.setattr(
        summary.notify, "send_text", lambda *a, **k: calls.append(a) or True
    )
    history = [context.Segment(speaker="me", text="hi", end=1.0)]
    summary.render_and_deliver(cfg, them_only, history, cfg["save_dir"])
    assert calls == []


def test_render_and_deliver_calls_ai_saves_file_and_pushes_telegram(
    cfg, meeting_preset, monkeypatch, tmp_path
):
    ai_calls: list[dict] = []
    tg_calls: list[tuple] = []

    def fake_ai(cfg_arg, system_prompt, user_text, on_delta):
        ai_calls.append({"system": system_prompt, "user": user_text})
        return _reply(True, "summary text")

    def fake_tg(token, chat, text):
        tg_calls.append((token, chat, text))
        return True

    monkeypatch.setattr(summary.ai, "ask_streaming", fake_ai)
    monkeypatch.setattr(summary.notify, "send_text", fake_tg)

    history = [
        context.Segment(speaker="them", text="hello", end=1.0),
        context.Segment(speaker="me", text="hi there", end=2.0),
    ]
    summary.render_and_deliver(cfg, meeting_preset, history, str(tmp_path))

    assert len(ai_calls) == 1
    assert ai_calls[0]["system"] == meeting_preset.prompt
    assert "[them]: hello" in ai_calls[0]["user"]
    assert "[me]: hi there" in ai_calls[0]["user"]

    assert len(tg_calls) == 1
    assert tg_calls[0] == ("test-token", "test-chat", "summary text")

    summary_files = list(tmp_path.glob("meeting_summary*.txt"))
    assert len(summary_files) == 1
    assert summary_files[0].read_text(encoding="utf-8") == "summary text"


def test_render_and_deliver_skips_telegram_when_credentials_missing(
    meeting_preset, monkeypatch, tmp_path
):
    ai_calls: list[str] = []
    tg_calls: list[tuple] = []

    monkeypatch.setattr(
        summary.ai,
        "ask_streaming",
        lambda *a, **k: ai_calls.append("called") or _reply(True, "summary text"),
    )
    monkeypatch.setattr(
        summary.notify, "send_text", lambda *a, **k: tg_calls.append(a) or True
    )

    cfg = {
        "save_dir": str(tmp_path),
        "model": "m",
        "anthropic_api_key": "k",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
    }
    history = [context.Segment(speaker="them", text="hi", end=1.0)]
    summary.render_and_deliver(cfg, meeting_preset, history, str(tmp_path))

    assert ai_calls == ["called"]
    assert tg_calls == []
    assert list(tmp_path.glob("meeting_summary*.txt"))


def test_render_and_deliver_swallows_telegram_errors(
    cfg, meeting_preset, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        summary.ai, "ask_streaming", lambda *a, **k: _reply(True, "summary text")
    )

    def fake_tg(token, chat, text):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(summary.notify, "send_text", fake_tg)

    history = [context.Segment(speaker="them", text="hi", end=1.0)]
    # Must not raise.
    summary.render_and_deliver(cfg, meeting_preset, history, str(tmp_path))

    assert list(tmp_path.glob("meeting_summary*.txt"))


def test_render_and_deliver_reports_ai_failure_but_still_writes(
    cfg, meeting_preset, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        summary.ai, "ask_streaming", lambda *a, **k: _reply(False, "network died")
    )
    tg_texts: list[str] = []
    monkeypatch.setattr(
        summary.notify,
        "send_text",
        lambda token, chat, text: tg_texts.append(text) or True,
    )

    history = [context.Segment(speaker="them", text="hi", end=1.0)]
    summary.render_and_deliver(cfg, meeting_preset, history, str(tmp_path))

    files = list(tmp_path.glob("meeting_summary*.txt"))
    assert len(files) == 1
    assert "network died" in files[0].read_text(encoding="utf-8")
    assert tg_texts == [files[0].read_text(encoding="utf-8")]
