"""End-of-session summary for ``on_shutdown`` presets.

The runner calls :func:`render_and_deliver` from its shutdown path when the
selected preset declares ``trigger="on_shutdown"``. The AI sees the full
session transcript (subject to the preset's speaker filter), the reply is
written to a file inside the session directory as ``<preset_name>.txt``,
and — if Telegram is configured — pushed as a chat message. If nothing was
transcribed during the session, the AI call is skipped entirely so an
empty run does not burn an API call or send a blank Telegram message.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import ai, config, context, notify, presets, storage, ui


def render_and_deliver(
    cfg: Mapping[str, Any],
    preset: presets.Preset,
    history: list[context.Segment],
    session_dir: str,
) -> None:
    """Run the shutdown AI call, save its reply to a file, and push to Telegram."""
    segments = [
        segment
        for segment in history
        if context.speaker_matches(segment.speaker, preset.speaker_filter)
    ]
    if not segments:
        ui.info("No transcript content to summarize; skipping.")
        return

    payload = context.render(segments)
    text = _call_ai(cfg, preset, payload)

    path = write(session_dir, preset.name, text)
    if path is not None:
        ui.info(f"Summary saved to {path}")

    token = str(cfg.get("telegram_bot_token") or "")
    chat = str(cfg.get("telegram_chat_id") or "")
    if token and chat:
        ui.info("Sending to Telegram…")
        try:
            notify.send_text(token, chat, text)
        except Exception as exc:
            secrets = [str(cfg.get(key) or "") for key in config.CREDENTIAL_FIELDS]
            ui.error(
                ui.scrub(
                    f"Telegram delivery raised: {type(exc).__name__}: {exc}", secrets
                )
            )


def write(session_dir: str, preset_name: str, text: str) -> Path | None:
    """Write ``text`` to ``<session_dir>/<preset_name>.txt``.

    Returns the path on success, or ``None`` if the file could not be
    created or written. Warnings go through :mod:`voicerecon.ui`; the caller
    keeps going either way.
    """
    try:
        path = storage.new_private_file(session_dir, preset_name)
    except (OSError, RuntimeError) as exc:
        ui.warn(f"Cannot create summary file: {exc}")
        return None

    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        ui.warn(f"Cannot write summary file ({path}): {exc.strerror or exc}")
        return None
    return path


def _call_ai(cfg: Mapping[str, Any], preset: presets.Preset, payload: str) -> str:
    ui.rule(f"AI [{preset.name}]")
    ui.info("Calling AI…")
    printer = ui.SentenceStreamPrinter()
    reply = ai.ask_streaming(cfg, preset.prompt, payload, printer.push)
    if reply.ok:
        printer.flush()
        return reply.text
    text = f"(AI failed) {reply.text}"
    ui.info(text)
    return text
