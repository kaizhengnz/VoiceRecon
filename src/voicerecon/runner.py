"""Main event loop.

Wires audio capture, VAD, segmentation, STT, transcript writing and
(optionally) AI-per-segment delivery together. Runs until the user hits
Ctrl+C or an unrecoverable error occurs.

Threading:

- Two :class:`voicerecon.audio.AudioSource` capture threads (mic +
  loopback) each push blocks into their own callback, which runs VAD and
  updates the shared :class:`voicerecon.segmenter.Segmenter`.
- The main thread pumps a work loop: drain ready segments, run STT
  synchronously, write to the transcript, and (when a preset is selected)
  fire the AI call + Telegram delivery.

STT on the main thread is intentional — running one segment at a time
keeps memory bounded and avoids interleaved AI calls stomping each other's
prompts.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from . import ai, audio, config, context, notify, presets, segmenter, stt, summary, transcript, ui, vad

MAX_HISTORY = 200
"""Cap on retained ready segments used to build ``window:N`` context. At the
longest configured window (5 min ≈ 60 short utterances) 200 is comfortable.
Suspended when the active preset is batch — the shutdown path needs the
full session or a long meeting gets truncated."""

DRAIN_INTERVAL_SECONDS = 0.1


def _secrets(cfg: Mapping[str, Any]) -> list[str]:
    return [str(cfg.get(key) or "") for key in config.CREDENTIAL_FIELDS]


def run(cfg: Mapping[str, Any], preset: presets.Preset | None) -> int:
    """Run the listen loop. Returns the process exit code."""
    from . import platform_check

    hint = platform_check.loopback_hint()
    if hint:
        ui.warn(hint)

    writer = transcript.TranscriptWriter(str(cfg["save_dir"]))
    seg = segmenter.Segmenter()

    silence_ms = int(float(cfg["speech_silence_seconds"]) * 1000)
    mic_vad = vad.StreamVAD(min_silence_ms=silence_ms)
    loop_vad = vad.StreamVAD(min_silence_ms=silence_ms)

    def _mic_callback(samples, ts):  # runs on the mic capture thread
        events = mic_vad.feed(samples, ts)
        seg.on_audio("me", samples)
        seg.on_events("me", events)

    def _loop_callback(samples, ts):  # runs on the loopback capture thread
        events = loop_vad.feed(samples, ts)
        seg.on_audio("them", samples)
        seg.on_events("them", events)

    mic = audio.AudioSource(
        kind="mic",
        device_name=str(cfg.get("input_device") or "") or None,
        callback=_mic_callback,
    )
    loopback = audio.AudioSource(
        kind="loopback",
        device_name=str(cfg.get("loopback_device") or "") or None,
        callback=_loop_callback,
    )

    transcriber = stt.Transcriber(model_size=str(cfg["whisper_model_size"]))

    ui.rule("VoiceRecon listening")
    ui.info(f"Save directory: {cfg['save_dir']}")
    ui.info(f"Silence threshold: {cfg['speech_silence_seconds']}s")
    ui.info(f"Whisper model: {cfg['whisper_model_size']} (loads on first speech)")
    if preset is None:
        ui.info("Mode: transcript only (no AI, no Telegram).")
    else:
        if preset.name == presets.CUSTOM_NAME:
            ui.info(f"Mode: --prompt — {preset.description}")
        else:
            ui.info(f"Mode: --listen {preset.name} — {preset.description}")
        if preset.is_batch:
            ui.info(f"  speaker filter: {preset.speaker_filter}   trigger: on shutdown (Ctrl+C)")
        else:
            ui.info(f"  speaker filter: {preset.speaker_filter}   context: {preset.context}")
        ui.info(f"  AI model: {cfg['model']}")
        ui.info(f"  Telegram: {ui.mask(str(cfg.get('telegram_chat_id')))}")
    ui.info("Press Ctrl+C to quit.\n")

    history: list[context.Segment] = []

    try:
        with mic, loopback:
            while True:
                ready = seg.drain()
                for item in ready:
                    _process_segment(
                        cfg, item, preset, writer, transcriber, history
                    )
                time.sleep(DRAIN_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        ui.info("\nStopping…")
    finally:
        remaining = seg.flush(time.monotonic())
        for item in remaining:
            _process_segment(cfg, item, preset, writer, transcriber, history)
        if preset is not None and preset.is_batch:
            summary.render_and_deliver(cfg, preset, history)
        if writer.path is not None:
            ui.info(f"Transcript saved to {writer.path}")
        ui.info("VoiceRecon stopped.")
    return 0


def _process_segment(
    cfg: Mapping[str, Any],
    item: segmenter.ReadySegment,
    preset: presets.Preset | None,
    writer: transcript.TranscriptWriter,
    transcriber: stt.Transcriber,
    history: list[context.Segment],
) -> None:
    text = transcriber.transcribe(item.audio)
    if not text:
        return

    writer.append(item.speaker, text)
    ui.info(f"[{item.speaker}] {text}")

    ctx_segment = context.Segment(
        speaker=item.speaker, text=text, end=item.ended_at
    )
    history.append(ctx_segment)
    if (preset is None or not preset.is_batch) and len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]

    if preset is None or preset.is_batch:
        return
    if not context.speaker_matches(item.speaker, preset.speaker_filter):
        return

    payload_segments = context.assemble(
        preset.context, history, ctx_segment, preset.speaker_filter
    )
    payload = context.render(payload_segments)

    ui.rule(f"AI [{preset.name}]")
    reply = ai.ask_streaming(
        cfg,
        preset.prompt,
        payload,
        lambda chunk: print(chunk, end="", flush=True),
    )
    if reply.ok:
        print(flush=True)
        ai_text = reply.text
    else:
        ai_text = f"(AI failed) {reply.text}"
        ui.info(ai_text)

    token = str(cfg.get("telegram_bot_token") or "")
    chat = str(cfg.get("telegram_chat_id") or "")
    if token and chat:
        try:
            notify.send_text(token, chat, f"[{item.speaker}] {text}\n\n{ai_text}")
        except Exception as exc:
            ui.error(
                ui.scrub(f"Telegram delivery raised: {type(exc).__name__}: {exc}", _secrets(cfg))
            )


