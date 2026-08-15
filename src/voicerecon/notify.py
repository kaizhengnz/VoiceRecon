"""Telegram text delivery.

Only ``sendMessage`` is used — VoiceRecon has no images to attach, unlike
ScreenRecon. The scrubbing / URL-token hygiene rules are the same: any
third-party string that quotes the request URL (which contains the bot
token) passes through :func:`_sanitize` before it reaches stdout or a log.

Failures print a warning and never raise; the audio pipeline keeps running.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from . import ui

API_BASE = "https://api.telegram.org"
TIMEOUT_SECONDS = 30

MESSAGE_LIMIT = 4096
"""Telegram's sendMessage body limit, in UTF-16 code units."""

_BOT_PATH = re.compile(r"/bot[^/\s]+", re.IGNORECASE)


def utf16_len(text: str) -> int:
    """Length in UTF-16 code units — the unit Telegram counts limits in."""
    return len(text.encode("utf-16-le")) // 2


def _truncate_utf16(text: str, budget: int) -> str:
    if utf16_len(text) <= budget:
        return text
    end = min(len(text), budget)
    while end > 0 and utf16_len(text[:end]) > budget:
        end -= 1
    return text[:end]


def chunk_text(text: str, size: int = MESSAGE_LIMIT) -> list[str]:
    """Split long text into chunks of at most ``size`` UTF-16 code units."""
    if size <= 0:
        raise ValueError("size must be a positive integer")
    chunks: list[str] = []
    rest = text or ""
    while rest:
        piece = _truncate_utf16(rest, size) or rest[:1]
        chunks.append(piece)
        rest = rest[len(piece) :]
    return chunks


def _endpoint(token: str, method: str) -> str:
    return f"{API_BASE}/bot{token}/{method}"


def _sanitize(text: str, token: str, chat_id: str | None = None) -> str:
    """Remove the bot token and chat ID from third-party text.

    The literal scrub runs first (only pass that can match a token containing
    ``/``); the URL-path regex runs afterwards as a backstop.
    """
    scrubbed = ui.scrub(str(text), [token, quote(token, safe=""), chat_id])
    return _BOT_PATH.sub("/bot<redacted>", scrubbed)


def _call(
    token: str, method: str, chat_id: str | None = None, **kwargs: Any
) -> tuple[bool, str, Any]:
    """Call the Bot API. Returns (ok, message, result). Never raises."""
    try:
        import requests
    except ImportError:
        return False, "Missing dependency 'requests'. Install with: pip install voicerecon", None

    try:
        response = requests.post(_endpoint(token, method), timeout=TIMEOUT_SECONDS, **kwargs)
    except Exception as exc:
        detail = _sanitize(f"request failed: {type(exc).__name__}: {exc}", token, chat_id)
        return False, detail, None

    status = response.status_code
    try:
        payload = response.json()
    except ValueError:
        return False, f"Telegram returned an unparseable response (HTTP {status}).", None

    if not isinstance(payload, dict):
        return False, f"Telegram returned an unexpected response (HTTP {status}).", None

    if payload.get("ok"):
        return True, "ok", payload.get("result")

    description = str(payload.get("description", "unknown error"))
    code = payload.get("error_code", response.status_code)
    return False, _sanitize(f"Telegram error {code}: {description}", token, chat_id), None


def send_text(bot_token: str, chat_id: str, text: str) -> bool:
    """Send ``text``, splitting into multiple messages if it exceeds the limit.

    Returns whether every chunk was delivered. Failures print a warning
    (each part identified so partial delivery is visible) and never raise.
    """
    if not text:
        return True
    chunks = chunk_text(text)
    ok_all = True
    for index, chunk in enumerate(chunks, start=1):
        ok, message, _ = _call(
            bot_token, "sendMessage", chat_id=chat_id, data={"chat_id": chat_id, "text": chunk}
        )
        if not ok:
            ui.warn(f"Telegram delivery failed (part {index}/{len(chunks)}): {message}")
            ok_all = False
    if ok_all:
        summary = (
            f"Sent to Telegram ({len(chunks)} part(s))."
            if len(chunks) > 1
            else "Sent to Telegram."
        )
        ui.info(summary)
    return ok_all


def verify_credentials(bot_token: str, chat_id: str) -> tuple[bool, str]:
    """Setup wizard probe: ``getMe`` validates the token, a test message the chat ID."""
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    if not bot_token:
        return False, "No bot token entered."
    if not chat_id:
        return False, "No chat ID entered."

    ok, message, result = _call(bot_token, "getMe", chat_id=chat_id)
    if not ok:
        return False, f"bot token check failed: {message}"
    username = ""
    if isinstance(result, dict) and result.get("username"):
        username = f" (@{result['username']})"

    ok, message, _ = _call(
        bot_token,
        "sendMessage",
        chat_id=chat_id,
        data={
            "chat_id": chat_id,
            "text": "VoiceRecon is configured. This message verifies the delivery channel.",
        },
    )
    if not ok:
        return False, f"token{username} is valid, but sending to that chat ID failed: {message}"
    return True, f"bot{username} reached chat {ui.mask(chat_id)}; a test message was sent."
