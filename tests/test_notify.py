"""Telegram text chunking, delivery, and token hygiene."""

from __future__ import annotations

import sys

import pytest

from voicerecon import notify

TOKEN = "123456:AAHfake-telegram-bot-token"
CHAT_ID = "987654321"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def test_chunking_respects_the_message_limit():
    text = "z" * (notify.MESSAGE_LIMIT * 2 + 17)
    chunks = notify.chunk_text(text)
    assert len(chunks) == 3
    assert all(len(chunk) <= notify.MESSAGE_LIMIT for chunk in chunks)
    assert "".join(chunks) == text


def test_chunking_empty_text_returns_nothing():
    assert notify.chunk_text("") == []


def test_chunking_rejects_non_positive_size():
    with pytest.raises(ValueError):
        notify.chunk_text("abc", 0)


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def install_fake_requests(monkeypatch, responses):
    calls: list[dict] = []
    queue = list(responses)

    class FakeRequests:
        @staticmethod
        def post(url, timeout=None, **kwargs):
            calls.append({"url": url, "timeout": timeout, **kwargs})
            if isinstance(queue[0], Exception):
                raise queue.pop(0)
            return queue.pop(0)

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)
    return calls


def test_short_text_sends_one_message(monkeypatch):
    calls = install_fake_requests(monkeypatch, [FakeResponse({"ok": True})])
    assert notify.send_text(TOKEN, CHAT_ID, "hello") is True
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/sendMessage")
    assert calls[0]["data"]["text"] == "hello"


def test_empty_text_sends_nothing(monkeypatch):
    calls = install_fake_requests(monkeypatch, [])
    assert notify.send_text(TOKEN, CHAT_ID, "") is True
    assert calls == []


def test_long_text_is_split_across_messages(monkeypatch):
    long_text = "w" * (notify.MESSAGE_LIMIT + 500)
    calls = install_fake_requests(
        monkeypatch, [FakeResponse({"ok": True}), FakeResponse({"ok": True})]
    )
    assert notify.send_text(TOKEN, CHAT_ID, long_text) is True
    assert len(calls) == 2
    assert "".join(call["data"]["text"] for call in calls) == long_text


def test_bot_token_never_appears_in_output(monkeypatch, capsys):
    install_fake_requests(
        monkeypatch,
        [RuntimeError(f"HTTPSConnectionPool https://api.telegram.org/bot{TOKEN}/sendMessage")],
    )
    notify.send_text(TOKEN, CHAT_ID, "text")
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err


def test_percent_encoded_token_is_scrubbed(monkeypatch, capsys):
    from urllib.parse import quote

    token = "1234567890:AAH fake token with spaces"
    install_fake_requests(
        monkeypatch,
        [RuntimeError(f"Max retries exceeded with url: /bot{quote(token, safe='')}/sendMessage")],
    )
    notify.send_text(token, CHAT_ID, "text")
    captured = capsys.readouterr().out + capsys.readouterr().err
    assert quote(token, safe="") not in captured
    assert token not in captured


def test_chat_id_is_masked_in_errors(monkeypatch, capsys):
    not_found = FakeResponse(
        {"ok": False, "error_code": 400, "description": f"chat {CHAT_ID} not found"}
    )
    install_fake_requests(monkeypatch, [not_found])
    notify.send_text(TOKEN, CHAT_ID, "text")
    assert CHAT_ID not in capsys.readouterr().out


def test_api_error_description_is_surfaced(monkeypatch, capsys):
    install_fake_requests(
        monkeypatch,
        [FakeResponse({"ok": False, "error_code": 403, "description": "bot was blocked"})],
    )
    notify.send_text(TOKEN, CHAT_ID, "text")
    assert "bot was blocked" in capsys.readouterr().out


def test_network_error_does_not_raise(monkeypatch):
    install_fake_requests(monkeypatch, [RuntimeError("connection reset")])
    assert notify.send_text(TOKEN, CHAT_ID, "text") is False


def test_verify_requires_both_fields():
    ok, message = notify.verify_credentials("", CHAT_ID)
    assert ok is False and "token" in message.lower()
    ok, message = notify.verify_credentials(TOKEN, "")
    assert ok is False and "chat id" in message.lower()


def test_verify_calls_get_me_then_send_message(monkeypatch):
    calls = install_fake_requests(
        monkeypatch, [FakeResponse({"ok": True}), FakeResponse({"ok": True})]
    )
    ok, _ = notify.verify_credentials(TOKEN, CHAT_ID)
    assert ok is True
    assert [call["url"].rsplit("/", 1)[-1] for call in calls] == ["getMe", "sendMessage"]
