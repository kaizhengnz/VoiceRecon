# VoiceRecon

Capture microphone + system audio, transcribe each utterance locally with Whisper, and optionally send each completed segment to an AI with a scenario preset (translate, interview help, lecture notes, and so on).

Sibling of [ScreenRecon](https://github.com/kaizhengnz/ScreenRecon) — same design philosophy (long-running desktop tool, local-first, Telegram delivery), applied to speech instead of screen regions.

Status: **alpha**. MVP is Windows-first; macOS works if you install BlackHole; Linux works via PulseAudio / PipeWire monitor sources.

## What it does

Two modes, picked at launch:

1. **Transcript-only** (default — no flag). Writes a plain-text transcript file under your save directory. Nothing goes to the cloud. Useful for recording meetings you want to search later.
2. **AI-per-segment** (`--listen <preset>`). Same transcript, plus each completed utterance is sent to the AI with a preset-specific system prompt; the response goes to Telegram.

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

# Listen and translate the other party's speech into Chinese
voicerecon --listen translate

# See what presets exist
voicerecon --presets

# See what audio devices are visible (numbered, with the default marked)
voicerecon --show-devices

# Print current config (credentials masked)
voicerecon --show
```

## Built-in AI presets

| Preset | Filter | Context | Purpose |
| --- | --- | --- | --- |
| `translate` | them | current | translate the other party's speech into Chinese |
| `interview_candidate` | them | current | analyze the interviewer's question, outline an answer |
| `interview_recruiter` | them | last 5 min | evaluate the candidate's answer, suggest a follow-up |
| `lecture` | them | last 5 min | extract the key concept from a lecture excerpt |
| `speaking` | me | current | give feedback on your own spoken sentence |
| `debate` | them | last 3 min | suggest counter-arguments |
| `sales` | them | current | identify customer need and suggest a talking point |

- **Filter** — `them` (system audio only), `me` (mic only), or `both`. Segments from ignored streams are dropped, not sent.
- **Context** — `current` sends only the segment that just finished. `window:<seconds>` sends every segment within that many seconds behind it.

## Config file

`~/.config/voicerecon/config.json` on Linux/macOS, `%APPDATA%\voicerecon\config.json` on Windows (technically `$XDG_CONFIG_HOME/voicerecon/` if set).

```json
{
  "save_dir": "~/VoiceRecon",
  "speech_silence_seconds": 1.5,
  "whisper_model_size": "small",
  "input_device": "",
  "loopback_device": "",
  "model": "claude-haiku-4-5",
  "anthropic_api_key": "",
  "telegram_bot_token": "",
  "telegram_chat_id": ""
}
```

`input_device` and `loopback_device` stay empty to follow whatever the system defaults are at capture time. `--configure` lists the detected devices by index, marks the one an empty field resolves to, and accepts an index or a device name — partial names match; `0` clears a pinned device back to the system default.

The three credential fields are only required when using `--listen <preset>`. Transcript-only mode works with them all empty.

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
