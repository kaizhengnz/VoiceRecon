"""End-of-session summary for ``on_shutdown`` presets.

The runner calls :func:`render_and_deliver` from its shutdown path when the
selected preset declares ``trigger="on_shutdown"``. The AI sees the full
session transcript (subject to the preset's speaker filter), the reply is
written to a file under ``save_dir``, and — if Telegram is configured —
pushed as a chat message. If nothing was transcribed during the session,
the AI call is skipped entirely so an empty run does not burn an API call
or send a blank Telegram message.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from . import ai, config, context, notify, presets, storage, ui

FILENAME_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def render_and_deliver(
    cfg: Mapping[str, Any],
    preset: presets.Preset,
    history: list[context.Segment],
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

    path = write(str(cfg["save_dir"]), preset.name, text)
    if path is not None:
        ui.info(f"Summary saved to {path}")

    token = str(cfg.get("telegram_bot_token") or "")
    chat = str(cfg.get("telegram_chat_id") or "")
    if token and chat:
        try:
            notify.send_text(token, chat, text)
        except Exception as exc:
            secrets = [str(cfg.get(key) or "") for key in config.CREDENTIAL_FIELDS]
            ui.error(
                ui.scrub(
                    f"Telegram delivery raised: {type(exc).__name__}: {exc}", secrets
                )
            )


def write(save_dir: str, preset_name: str, text: str) -> Path | None:
    """Write ``text`` to ``<save_dir>/<preset_name>-YYYYMMDD-HHMMSS.txt``.

    Returns the path on success, or ``None`` if the file could not be
    created or written. Warnings go through :mod:`voicerecon.ui`; the caller
    keeps going either way.
    """
    stem = f"{preset_name}-{datetime.now().strftime(FILENAME_TIMESTAMP_FORMAT)}"
    try:
        path = storage.new_private_file(save_dir, stem)
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
    reply = ai.ask_streaming(
        cfg, preset.prompt, payload, lambda chunk: print(chunk, end="", flush=True)
    )
    if reply.ok:
        print(flush=True)
        return reply.text
    text = f"(AI failed) {reply.text}"
    ui.info(text)
    return text
