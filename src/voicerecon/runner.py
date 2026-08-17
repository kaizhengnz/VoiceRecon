"""Main event loop.

Wires audio capture, VAD, segmentation, streaming STT, transcript
writing and (optionally) AI-per-segment delivery together. Runs until
the user hits Ctrl+C or an unrecoverable error occurs.

Threading:

- Two :class:`voicerecon.audio.AudioSource` capture threads (mic +
  loopback) push blocks into their own callback, which runs VAD and
  updates the shared :class:`voicerecon.segmenter.Segmenter`. Audio
  that survives the segmenter's loopback-priority filter is routed
  into the matching :class:`voicerecon.streaming.StreamingTranscriber`
  in the same thread (StreamingTranscriber's own lock guards the
  buffer).
- The main thread pumps a work loop: drain segment boundaries (each
  triggers ``finalize`` on the matching streamer plus writer + AI
  delivery), then call ``commit_step`` on each streamer to flush any
  locally-agreed prefix to the terminal.

Whisper runs on the main thread on purpose — the streamer holds its
lock only around buffer mutations, so a 200 ms Whisper call does not
block the audio capture threads. Serial transcription also keeps AI
calls from interleaving.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import (
    ai,
    audio,
    config,
    context,
    notify,
    presets,
    segmenter,
    streaming,
    stt,
    summary,
    transcript,
    ui,
    vad,
)

MAX_HISTORY = 200
"""Cap on retained ready segments used to build ``window:N`` context. At the
longest configured window (5 min ≈ 60 short utterances) 200 is comfortable.
Suspended when the active preset is batch — the shutdown path needs the
full session or a long meeting gets truncated."""

COMMIT_INTERVAL_SECONDS = 0.5
"""Cadence at which the main loop asks each streamer to commit any locally-
agreed prefix. Slower than the old drain interval because each pump can
trigger a full Whisper pass; ~500 ms keeps the streaming feel while leaving
CPU headroom for a synchronous AI call."""


def _secrets(cfg: Mapping[str, Any]) -> list[str]:
    return [str(cfg.get(key) or "") for key in config.CREDENTIAL_FIELDS]


def run(cfg: Mapping[str, Any], preset: presets.Preset | None) -> int:
    """Run the listen loop. Returns the process exit code."""
    from . import platform_check

    # soundcard prints one of these on every buffer glitch on Windows loopback;
    # they interleave with transcript / AI output and are almost always harmless
    # (the VAD absorbs the missed samples). Filter by warning class — a
    # message-based filter did not stick on Python 3.14.
    import warnings
    try:
        import soundcard
        warnings.simplefilter("ignore", soundcard.SoundcardRuntimeWarning)
    except (ImportError, AttributeError):
        warnings.filterwarnings("ignore", message="data discontinuity in recording")

    hint = platform_check.loopback_hint()
    if hint:
        ui.warn(hint)

    writer = transcript.TranscriptWriter(str(cfg["save_dir"]))

    # One WhisperModel shared by both streamers — the model is stateless
    # across ``transcribe`` calls and both streamers only run on the main
    # thread, so sharing halves memory without introducing a race.
    model_size = str(cfg["whisper_model_size"])
    shared_model: Any | None = None

    def _model_factory() -> Any:
        nonlocal shared_model
        if shared_model is None:
            shared_model = stt.build_model(model_size)
        return shared_model

    streamers: dict[str, streaming.StreamingTranscriber] = {
        speaker: streaming.StreamingTranscriber(_model_factory)
        for speaker in ("me", "them")
    }
    accumulated: dict[str, list[str]] = {"me": [], "them": []}

    def _stream_router(speaker: str, samples: np.ndarray) -> None:
        streamers[speaker].feed(samples)

    seg = segmenter.Segmenter(on_stream_audio=_stream_router)

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

    ui.rule("VoiceRecon listening")
    ui.info(f"Save directory: {cfg['save_dir']}")
    ui.info(f"Silence threshold: {cfg['speech_silence_seconds']}s")
    ui.info(f"Whisper model: {model_size} (loads on first speech)")
    if preset is None:
        ui.info("Mode: transcript only (no AI, no Telegram).")
    else:
        label = "custom prompt" if preset.is_custom else preset.name
        ui.info(f"Mode: {label} — {preset.description}")
        if preset.is_batch:
            ui.info(f"  speaker filter: {preset.speaker_filter}   trigger: on shutdown (Ctrl+C)")
        else:
            ui.info(f"  speaker filter: {preset.speaker_filter}   context: {preset.context}")
        ui.info(f"  AI model: {cfg['model']}")
        ui.info(f"  Telegram: {ui.mask(str(cfg.get('telegram_chat_id')))}")
    ui.info("Press Ctrl+C to quit.\n")

    history: list[context.Segment] = []

    def _emit_chunk(speaker: str, chunk: str) -> None:
        if not chunk:
            return
        if not accumulated[speaker]:
            print(f"[{speaker}]", end="", flush=True)
        print(chunk, end="", flush=True)
        accumulated[speaker].append(chunk)

    def _close_line(speaker: str) -> str:
        text = "".join(accumulated[speaker]).strip()
        if accumulated[speaker]:
            print(flush=True)  # terminate the streaming line
        accumulated[speaker] = []
        return text

    def _flush_boundary(item: segmenter.ReadySegment) -> None:
        _emit_chunk(item.speaker, streamers[item.speaker].finalize())
        text = _close_line(item.speaker)
        if text:
            _dispatch_utterance(cfg, item, text, preset, writer, history)

    try:
        with mic, loopback:
            while True:
                for item in seg.drain():
                    _flush_boundary(item)
                for speaker, streamer in streamers.items():
                    _emit_chunk(speaker, streamer.commit_step())
                time.sleep(COMMIT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        ui.info("\nStopping…")
    finally:
        for item in seg.flush(time.monotonic()):
            _flush_boundary(item)
        if preset is not None and preset.is_batch:
            summary.render_and_deliver(cfg, preset, history)
        if writer.path is not None:
            ui.info(f"Transcript saved to {writer.path}")
        ui.info("VoiceRecon stopped.")
    return 0


def _dispatch_utterance(
    cfg: Mapping[str, Any],
    item: segmenter.ReadySegment,
    text: str,
    preset: presets.Preset | None,
    writer: transcript.TranscriptWriter,
    history: list[context.Segment],
) -> None:
    writer.append(item.speaker, text)

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
    printer = ui.SentenceStreamPrinter()
    reply = ai.ask_streaming(cfg, preset.prompt, payload, printer.push)
    if reply.ok:
        printer.flush()
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
