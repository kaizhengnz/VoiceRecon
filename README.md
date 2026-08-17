# VoiceRecon

Capture microphone + system audio, transcribe each utterance locally with Whisper, and optionally hand the transcript to an AI — either through a built-in scenario preset (interview help, meeting summary) or with your own prompt. The chosen prompt lives in one field (`cfg["prompt"]`) that can hold either a built-in preset name or free-form text; `--prompt VALUE` on the CLI overrides it for one session.

Sibling of [ScreenRecon](https://github.com/kaizhengnz/ScreenRecon) — same design philosophy (long-running desktop tool, local-first, Telegram delivery), applied to speech instead of screen regions.

Status: **alpha**. MVP is Windows-first; macOS works if you install BlackHole; Linux works via PulseAudio / PipeWire monitor sources.

## What it does

Two modes, chosen by whatever `cfg["prompt"]` (or `--prompt VALUE`) resolves to:

1. **Transcript-only** — `cfg["prompt"]` is empty and no `--prompt` is given. Writes a plain-text transcript file under `save_dir`. Nothing goes to the cloud.
2. **AI-assisted** — `cfg["prompt"]` (or `--prompt VALUE`) is non-empty. The value is interpreted as:
   - **Built-in preset name** (`interview_candidate`, `interview_recruiter`, `meeting_summary`) — the preset owns its speaker filter, context window, and trigger; e.g. `meeting_summary` fires once at Ctrl+C over the full session, saves the reply to `save_dir`, and pushes it to Telegram; the interview presets fire after each completed utterance and push to Telegram in real time.
   - **Any other text** — treated as a custom streaming prompt. Defaults to `filter=them` (only the other party's speech reaches the AI) and `context=current` (just the segment that fired the trigger). No CLI override for these defaults — use a built-in preset if you need different behaviour.

Utterances end when either:
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

# Run once with a built-in preset (name overrides cfg['prompt'])
voicerecon --prompt interview_candidate

# One summary of the whole session, delivered when you hit Ctrl+C
voicerecon --prompt meeting_summary

# Free-form text is treated as a custom streaming prompt
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

The filter / context / trigger of a built-in preset are baked in and cannot be overridden per-invocation — for behaviours the built-ins don't cover, either write a custom prompt (defaults to filter=them + context=current + per-segment streaming) or add a new preset in `src/voicerecon/presets.py`.

## Config file

`~/.config/voicerecon/config.json` on Linux/macOS, `%APPDATA%\voicerecon\config.json` on Windows (technically `$XDG_CONFIG_HOME/voicerecon/` if set).

```json
{
  "save_dir": "~/VoiceRecon",
  "speech_silence_seconds": 1.5,
  "whisper_model_size": "small",
  "input_device": "",
  "loopback_device": "",
  "prompt": "",
  "prompt_trigger": "",
  "model": "claude-haiku-4-5",
  "anthropic_api_key": "",
  "telegram_bot_token": "",
  "telegram_chat_id": ""
}
```

`input_device` and `loopback_device` stay empty to follow whatever the system defaults are at capture time. `--configure` lists the detected devices by index, marks the one an empty field resolves to, and accepts an index or a device name — partial names match; `0` clears a pinned device back to the system default.

`prompt` is the single field that decides AI mode when you run `voicerecon` with no `--prompt`. Empty = transcript-only; a value that matches a built-in preset name (`interview_candidate`, `interview_recruiter`, `meeting_summary`) uses that preset with its baked filter/context/trigger; any other text is treated as a custom prompt. `--configure` sets it via a numbered menu that also offers "type your own prompt" for the free-text case. `--prompt VALUE` on the CLI overrides `cfg["prompt"]` for a single session and follows the same disambiguation rule.

`prompt_trigger` picks when the AI is called: `per_segment` fires after every completed utterance, `on_shutdown` fires once at Ctrl+C over the full session. Empty means "use the built-in preset's baked default, or `per_segment` for a custom prompt". `--configure` (and bare `voicerecon --prompt`) asks this after the prompt itself; a non-empty value in the config **overrides** the preset's baked trigger — e.g. setting `"prompt": "meeting_summary", "prompt_trigger": "per_segment"` runs the meeting-summary prompt after every utterance instead of at Ctrl+C. `filter` on custom prompts follows the chosen trigger (`per_segment` → `them`, `on_shutdown` → `both`).

The three credential fields are only required when `cfg["prompt"]` (or `--prompt VALUE`) resolves to a preset. Transcript-only mode works with them all empty.

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
