# VoiceRecon

Capture microphone + system audio, transcribe each utterance locally with Whisper, and optionally hand the transcript to an AI — either through a built-in scenario preset (interview help, meeting summary) or with your own prompt on the command line.

Sibling of [ScreenRecon](https://github.com/kaizhengnz/ScreenRecon) — same design philosophy (long-running desktop tool, local-first, Telegram delivery), applied to speech instead of screen regions.

Status: **alpha**. MVP is Windows-first; macOS works if you install BlackHole; Linux works via PulseAudio / PipeWire monitor sources.

## What it does

Three modes, picked at launch:

1. **Transcript-only** (default — no flag). Writes a plain-text transcript file under your save directory. Nothing goes to the cloud. Useful for recording meetings you want to search later.
2. **AI-assisted with a built-in preset** (`--listen <preset>`). Same transcript, plus the AI is called with a preset-specific system prompt. Presets fall into two families:
   - **Live** (`interview_candidate`, `interview_recruiter`) — the AI runs after each completed utterance and pushes its reply to Telegram in real time.
   - **On shutdown** (`meeting_summary`) — the AI runs once when you press Ctrl+C, over the entire session's transcript. The reply is saved to a file in `save_dir` and also pushed to Telegram. If nothing was recorded, the call is skipped.
3. **AI-assisted with a custom prompt** (`--prompt "..."`). Same transcript, but you supply the system prompt inline. Streaming only (per-segment); for a session summary use `--listen meeting_summary`.

Both modes segment audio when either:
- silence exceeds the configured threshold (`speech_silence_seconds`, default 1.5 s), or
- the other speaker starts talking (immediate cut).

## Install

```bash
pip install voicerecon
```

Requires Python 3.10+. Depends on PyTorch (pulled in transitively by `silero-vad`), so expect a substantial install. First run also downloads the Whisper model into your Hugging Face cache; subsequent runs use it directly.

Transcription uses a CUDA GPU automatically when one is visible, and falls back to the CPU otherwise. No configuration either way.

### macOS extra step: BlackHole

macOS does not expose system audio to third-party apps. To capture the other party's voice in a meeting:

1. Install [BlackHole 2ch](https://existential.audio/blackhole/) (free, ~30 s install).
2. Open **Audio MIDI Setup**, create a **Multi-Output Device** that includes both your speakers and BlackHole, and set it as your system output. This lets you *hear* audio while also routing it to BlackHole.
3. Run `voicerecon --show-devices` to confirm `BlackHole 2ch` appears. macOS has no loopback devices at all, so it is listed under Microphones.
4. Run `voicerecon --configure` and type `BlackHole 2ch` when asked for the loopback device — the loopback list is empty on macOS, so there is no number to pick.

## Usage

```bash
# One-time setup
voicerecon --configure

# Transcript-only listening
voicerecon

# Live help while being interviewed — AI reacts to each question the interviewer asks
voicerecon --listen interview_candidate

# One summary of the whole session, delivered when you hit Ctrl+C
voicerecon --listen meeting_summary

# Custom per-segment prompt — no preset needed
voicerecon --prompt "Translate the speaker's words into English"

# See what presets exist
voicerecon --presets

# See what audio devices are visible (numbered, with the default marked)
voicerecon --show-devices

# Print current config (credentials masked)
voicerecon --show
```

## Built-in AI presets

| Preset | Filter | Trigger | Context | Purpose |
| --- | --- | --- | --- | --- |
| `interview_candidate` | them | per segment | current | analyze the interviewer's question, outline an answer |
| `interview_recruiter` | them | per segment | last 5 min | evaluate the candidate's answer, suggest a follow-up |
| `meeting_summary` | both | on shutdown | full session | summarize topics, decisions, and action items at Ctrl+C |

- **Filter** — `them` (system audio only), `me` (mic only), or `both`. Segments from ignored streams are dropped, not sent.
- **Trigger** — `per segment` fires after each matching utterance and pushes the reply to Telegram immediately. `on shutdown` fires once at Ctrl+C over the whole session, saves the reply to a file in `save_dir` as `<preset>-YYYYMMDD-HHMMSS.txt`, and also pushes it to Telegram. When nothing was recorded, the `on shutdown` call is skipped.
- **Context** — applies only to per-segment presets. `current` sends only the segment that just finished. `window:<seconds>` sends every segment within that many seconds behind it.

## Custom prompt

If none of the built-in presets fit, hand the AI your own prompt on the command line:

```bash
voicerecon --prompt "Translate the speaker's words into English"
```

By default the prompt sees only the current segment from the other party (equivalent to `--from them --context current`). Both knobs are overridable:

- `--from them|me|both` — which speaker's segments the AI sees
- `--context current|window:<seconds>` — send just the latest segment, or every segment ending in the last N seconds

Custom prompts are streaming (per-segment) only — for an end-of-session summary use `--listen meeting_summary`. Same credential requirement as `--listen`: Anthropic API key + Telegram bot must be configured.

## Config file

`~/.config/voicerecon/config.json` on Linux/macOS, `%APPDATA%\voicerecon\config.json` on Windows (technically `$XDG_CONFIG_HOME/voicerecon/` if set).

```json
{
  "save_dir": "~/VoiceRecon",
  "speech_silence_seconds": 1.5,
  "whisper_model_size": "small",
  "input_device": "",
  "loopback_device": "",
  "listen": "",
  "model": "claude-haiku-4-5",
  "anthropic_api_key": "",
  "telegram_bot_token": "",
  "telegram_chat_id": ""
}
```

`input_device` and `loopback_device` stay empty to follow whatever the system defaults are at capture time. `--configure` lists the detected devices by index, marks the one an empty field resolves to, and accepts an index or a device name — partial names match; `0` clears a pinned device back to the system default.

`listen` names the default preset used when you run `voicerecon` with no `--listen` and no `--prompt`. Set it to `interview_candidate`, `interview_recruiter`, or `meeting_summary`; leave it empty to default to transcript-only. `--configure` picks it via a numbered menu (including a "none" entry). CLI flags always win: `--listen X` and `--prompt "..."` override the configured default per invocation.

The three credential fields are only required when using `--listen <preset>` or `--prompt "..."` — or when `listen` is set to a preset in the config. Transcript-only mode works with them all empty.

## Transcript file format

One file per session, named `transcript-YYYYMMDD-HHMMSS.txt`, one line per completed utterance:

```
[12:03:15] [them]: 我们下周的发布还是按原计划吗？
[12:03:22] [me]: 应该没问题，但要看 CI 是不是绿的。
```

## Development

```bash
git clone https://github.com/kaizhengnz/VoiceRecon
cd VoiceRecon
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0.
