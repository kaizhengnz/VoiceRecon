# VoiceRecon — Design

| Field | Value |
| --- | --- |
| Project | VoiceRecon |
| Document version | 0.1.2 |
| Date | 2026-08-15 |
| Status | Alpha (MVP) |
| Audience | Contributors and integrators |

## 1. Purpose

Capture what people are saying — through the microphone and through the computer's speakers — turn it into text with an on-device speech-to-text model, and either **save the transcript locally** or **also feed each utterance to an AI** with a scenario-specific prompt.

The design goals mirror ScreenRecon:

- **Long-running.** The tool is meant to sit in the corner during a meeting or a call; a single failed segment must not take it down.
- **Local-first.** The transcript is complete without any cloud call. Cloud (AI + Telegram) is opt-in per launch.
- **Composable presets, not modes.** The runtime pipeline is the same for every scenario; only the prompt and the speaker filter change.

## 2. Component overview

```
┌────────────────────────┐   ┌────────────────────────┐
│ AudioSource (mic)      │   │ AudioSource (loopback) │
│  soundcard recorder    │   │  soundcard recorder    │
│  → 16 kHz mono float32 │   │  → 16 kHz mono float32 │
└─────────┬──────────────┘   └───────────┬────────────┘
          │  block + monotonic ts        │
          ▼                              ▼
      ┌───────────┐                ┌───────────┐
      │ StreamVAD │                │ StreamVAD │
      │  silero   │                │  silero   │
      └─────┬─────┘                └─────┬─────┘
            │  SpeechEvent(start|end)    │
            ▼                            ▼
        ┌─────────────────────────────────┐
        │            Segmenter            │
        │  per-stream PendingSegment,     │
        │  emits ReadySegment on:         │
        │   - end (silence threshold)     │
        │   - other-stream start (cut)    │
        └───────────────┬─────────────────┘
                        │  ReadySegment (audio, speaker, timestamps)
                        ▼
                 ┌──────────────┐
                 │ Transcriber  │  faster-whisper (small model default)
                 │  auto lang   │
                 └──────┬───────┘
                        │  Transcription(text, language)
                        ▼
        ┌──────────────────────────────┐
        │      TranscriptWriter        │  always
        │  append line to daily file   │
        └──────────────────────────────┘
                        │
                        ▼ (only when --listen <preset>)
        ┌──────────────────────────────┐
        │  Preset filter + context     │  drop / include per preset
        │  assemble system + user text │
        └──────────────┬───────────────┘
                       ▼
                  ┌──────────┐
                  │    AI    │  Anthropic streaming
                  └────┬─────┘
                       ▼
                  ┌──────────┐
                  │ Telegram │  sendMessage (text-only)
                  └──────────┘
```

## 3. Threading

- Two capture threads (one per `AudioSource`), owned by `soundcard`'s recorder context manager.
- Both capture threads push audio + events into a single, lock-protected `Segmenter`. The lock is held only for the duration of the update — VAD runs *before* acquiring it.
- The main thread runs a work loop at 100 ms cadence: drain ready segments, transcribe synchronously, write to disk, optionally call AI + Telegram.

STT is synchronous on the main thread on purpose. Running Whisper concurrently on multiple segments blows up CPU / RAM and interleaves partial output; queueing them keeps the pipeline predictable.

## 4. Segmentation rules

An utterance ends when *either* of:

1. **Silence threshold** on the currently active stream — the corresponding stream's VAD emits `end` after `speech_silence_seconds` of quiet.
2. **Speaker change** — the other stream's VAD emits `start` while this stream is still mid-utterance. The current utterance is cut *at the moment of the other speaker's start*, and the new speaker's utterance begins immediately.

Rule 2 is why Segmenter needs cross-stream visibility (a per-stream VAD alone cannot know the other stream started).

## 5. Speaker filter and context

Every preset declares:

- `speaker_filter`: `both`, `them` (system audio only), `me` (mic only). Segments not matching the filter are still transcribed and written to the local file, but not sent to the AI.
- `context`: `current` sends only the segment that just fired; `window:<seconds>` sends every retained segment (subject to the same speaker filter) whose end timestamp is within N seconds behind the trigger.

`session` (unbounded history) is intentionally not offered. A user who really wants "everything so far" can set `window:999999`; capping it out of the type system avoids the silent-cost-blowup class of bug.

## 6. Config

Fields (`~/.config/voicerecon/config.json` or `%APPDATA%\voicerecon\config.json`):

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `save_dir` | string | `~/VoiceRecon` | Transcript files land here |
| `speech_silence_seconds` | number > 0 | `1.5` | Silence VAD threshold |
| `whisper_model_size` | string | `small` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `input_device` | string | `""` | Empty = default microphone |
| `loopback_device` | string | `""` | Empty = default speaker's loopback |
| `model` | string | `claude-haiku-4-5` | Only used with `--listen <preset>` |
| `anthropic_api_key` | string | `""` | Required for `--listen`; env `ANTHROPIC_API_KEY` overrides |
| `telegram_bot_token` | string | `""` | Required for `--listen` |
| `telegram_chat_id` | string | `""` | Required for `--listen` |

Credentials are optional at load time; `require_credentials_for_ai` is the single gate applied when `--listen` is given.

## 7. Platform notes

- **Windows** — `soundcard` exposes each speaker's WASAPI loopback as a normal input device automatically. Zero setup after `pip install`.
- **Linux** — Same mechanism via PulseAudio / PipeWire monitor sources. Zero setup on distros with a working audio stack.
- **macOS** — CoreAudio does not expose loopback. Users must install [BlackHole](https://existential.audio/blackhole/) and pick it as `loopback_device`. See README §"macOS extra step" for the full procedure.

Cross-platform equivalence via ScreenCaptureKit / CoreAudio Tap is tracked as a separate design effort; it is out of scope for MVP.

## 8. Revision history

| Date | Version | Change |
| --- | --- | --- |
| 2026-08-15 | 0.1.0 | Initial design and MVP implementation |
