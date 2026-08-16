"""Config loading, validation, saving and the interactive wizard.

Audio-focused fields (silence threshold, input devices, Whisper model
size). Anthropic + Telegram credentials are *optional*: they are only
required when a preset is selected via ``--listen <preset>``. Pure
transcript mode needs no cloud credentials.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import audio, storage, ui

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_SAVE_DIR = "~/VoiceRecon"
DEFAULT_WHISPER_SIZE = "small"
DEFAULT_SILENCE_SECONDS = 1.5
"""Silence before an utterance is considered ended. Meetings often have
short natural pauses; ``1.5s`` is a compromise between not chopping
mid-thought and not letting one utterance run for a minute."""

ENV_API_KEY = "ANTHROPIC_API_KEY"

WHISPER_SIZES: list[tuple[str, str]] = [
    ("tiny", "~40 MB, weakest, fastest"),
    ("base", "~75 MB, weak on non-English"),
    ("small", "~250 MB, good for CJK + Latin — default"),
    ("medium", "~770 MB, more accurate, slower"),
    ("large-v3", "~1.5 GB, best accuracy, needs GPU for real-time"),
]

MODEL_CHOICES: list[tuple[str, str, str]] = [
    ("claude-opus-5", "claude-opus-5", "high accuracy, more expensive"),
    ("claude-haiku-4-5", "claude-haiku-4-5", "cheaper and faster — the default"),
]

DEFAULTS: dict[str, Any] = {
    "save_dir": DEFAULT_SAVE_DIR,
    "speech_silence_seconds": DEFAULT_SILENCE_SECONDS,
    "whisper_model_size": DEFAULT_WHISPER_SIZE,
    "input_device": "",  # empty = default microphone
    "loopback_device": "",  # empty = default speaker's loopback
    "model": DEFAULT_MODEL,
    "anthropic_api_key": "",
    "telegram_bot_token": "",
    "telegram_chat_id": "",
}

CREDENTIAL_FIELDS = ("anthropic_api_key", "telegram_bot_token", "telegram_chat_id")

_CREDENTIAL_LABELS = {
    "anthropic_api_key": "Anthropic API key",
    "telegram_bot_token": "Telegram bot token",
    "telegram_chat_id": "Telegram chat ID",
}


class ConfigError(Exception):
    """Config is missing or invalid; the message is safe to show the user."""


class WizardAborted(Exception):
    """The setup wizard cannot continue (no input available, or too many retries)."""


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #


def _default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "voicerecon" / "config.json"


def config_path(override: str | os.PathLike[str] | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    return _default_config_path()


def read_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc.strerror or exc}") from None
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Config file {path} is not valid JSON (line {exc.lineno}): {exc.msg}. "
            "Run 'voicerecon --configure' to regenerate it."
        ) from None
    if not isinstance(data, dict):
        raise ConfigError(f"The top level of config file {path} must be a JSON object.")
    return data


def merge_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(DEFAULTS)
    for key, value in raw.items():
        merged[key] = value
    return merged


def apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    env_key = os.environ.get(ENV_API_KEY, "").strip()
    if env_key:
        cfg["anthropic_api_key"] = env_key
    return cfg


def load(path: str | os.PathLike[str] | None = None, *, validate: bool = True) -> dict[str, Any]:
    resolved = config_path(path)
    cfg = apply_env_overrides(merge_defaults(read_raw(resolved)))
    cfg["_path"] = str(resolved)
    if validate:
        validate_config(cfg)
    return cfg


def save(cfg: dict[str, Any], path: str | os.PathLike[str] | None = None) -> Path:
    resolved = config_path(path or cfg.get("_path"))
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        storage.make_private_dir(resolved.parent)
        # Direct write is fine — the wizard writes once at the end, and
        # atomic-rename adds complexity that ScreenRecon needed for its
        # more frequent writes but VoiceRecon doesn't.
        resolved.write_text(text, encoding="utf-8")
        storage.restrict(resolved, storage.PRIVATE_FILE_MODE)
    except OSError as exc:
        raise ConfigError(f"Cannot write config file {resolved}: {exc.strerror or exc}") from None
    return resolved


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate non-credential fields; invalid values raise a field-specific error."""
    silence = cfg.get("speech_silence_seconds")
    if isinstance(silence, bool) or not isinstance(silence, int | float):
        raise ConfigError(
            f"Config field 'speech_silence_seconds' must be a number, got {silence!r}"
        )
    if silence <= 0:
        raise ConfigError(
            f"Config field 'speech_silence_seconds' must be greater than 0, got {silence!r}"
        )

    for field in ("model", "save_dir", "whisper_model_size"):
        value = cfg.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"Config field {field!r} must be a non-empty string, got {value!r}")

    for field in ("input_device", "loopback_device"):
        value = cfg.get(field)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"Config field {field!r} must be a string")

    for field in CREDENTIAL_FIELDS:
        value = cfg.get(field)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"Config field {field!r} must be a string")


def require_credentials_for_ai(cfg: dict[str, Any]) -> None:
    """Refuse to start AI-per-segment mode when any credential is empty.

    Only called when the user selects ``--listen <preset>``. Pure
    transcript mode does not need any of these fields.
    """
    missing = [
        _CREDENTIAL_LABELS[field]
        for field in CREDENTIAL_FIELDS
        if not str(cfg.get(field) or "").strip()
    ]
    if missing:
        raise ConfigError(
            "AI preset needs: " + ", ".join(missing) + ".\n"
            "Run 'voicerecon --configure' to set them, or run without --listen "
            "for transcript-only mode."
        )


# --------------------------------------------------------------------------- #
# Interactive wizard
# --------------------------------------------------------------------------- #


MAX_PROMPT_RETRIES = 5


def _ask(label: str, current: Any, *, secret: bool = False) -> str:
    """Prompt for one value. Enter keeps the current value.

    ``secret=True`` does *not* hide characters during typing — API keys,
    bot tokens and chat IDs are all printable ASCII, and hiding them broke
    paste on Windows and left users unsure whether their input landed
    (this is the same decision ScreenRecon made). What ``secret`` still
    does: shows the current value as a mask in the ``[hint]`` (so an
    existing key is not redisplayed in full) and echoes a masked
    confirmation after Enter (so scrollback keeps only ``sk-ant-a... (N
    chars)`` prefixes). Runtime paths continue to scrub secrets.
    """
    has_current = current is not None and str(current) != ""
    if not has_current:
        hint = "empty"
    elif secret:
        hint = ui.mask(str(current))
    else:
        hint = str(current)

    prompt = f"{label} [{hint}]: "
    try:
        answer = input(prompt).strip()
    except EOFError:
        raise WizardAborted(
            "No input available (stdin is closed). Run 'voicerecon --configure' "
            "from an interactive terminal."
        ) from None
    if answer:
        if secret:
            ui.info(f"    received: {ui.mask(answer)}")
        return answer
    return "" if current is None else str(current)


def _ask_float(label: str, current: Any, *, minimum: float | None = None) -> float:
    for _ in range(MAX_PROMPT_RETRIES):
        answer = _ask(label, current)
        try:
            value = float(str(answer).strip())
        except ValueError:
            ui.warn(f"{label} needs a number, please try again.")
            continue
        if minimum is not None and value <= minimum:
            ui.warn(f"{label} must be greater than {minimum}, please try again.")
            continue
        return value
    raise WizardAborted(f"{label}: too many invalid answers, giving up.")


def _ask_choice(
    label: str, presets: list[tuple[str, str, str]], current: str, *, default: str | None = None
) -> str:
    current_index = len(presets) + 1
    labels = [preset_label for preset_label, _, _ in presets]
    width = max(len(preset_label) for preset_label in labels) if labels else 0
    current_preview = current if len(current) <= 60 else current[:57] + "..."

    default_index = current_index
    if default is not None:
        for index, (_, preset_value, _) in enumerate(presets, start=1):
            if preset_value == default:
                default_index = index
                break

    for index, (preset_label, _, note) in enumerate(presets, start=1):
        ui.info(f"    {index}) {preset_label:<{width}}  ({note})")
    ui.info(f"    {current_index}) (keep current — {current_preview})")

    for _ in range(MAX_PROMPT_RETRIES):
        prompt_label = f"    Enter 1-{current_index} or type any {label}"
        answer = _ask(prompt_label, str(default_index)).strip()
        if answer.isdigit():
            number = int(answer)
            if 1 <= number <= len(presets):
                return presets[number - 1][1]
            if number == current_index:
                return current
            ui.warn(f"    Choice must be 1-{current_index}, please try again.")
            continue
        return answer
    raise WizardAborted(f"{label}: too many invalid answers, giving up.")


def _ask_whisper_size(current: str) -> str:
    ui.info("   Whisper model size (bigger = more accurate, more RAM, slower):")
    presets = [(size, size, note) for size, note in WHISPER_SIZES]
    return _ask_choice("model size", presets, current, default=DEFAULT_WHISPER_SIZE)


def _ask_device(
    label: str, heading: str, names: list[str], default_name: str, current: str
) -> str:
    """Prompt for one audio device, listing the detected ones by number.

    A number picks from the list, any other text is stored as-is (soundcard
    matches names loosely, and macOS users type ``BlackHole 2ch``), and
    blank keeps the current value — empty meaning "follow whatever the
    system default is at capture time" rather than pinning today's name.
    """
    if names:
        ui.info(f"   {heading}:")
        for line in audio.format_device_lines(names, default_name):
            ui.info(f"     {line}")

    prompt = f"  {label} (number, name, or Enter for default)"
    for _ in range(MAX_PROMPT_RETRIES):
        answer = _ask(prompt, current).strip()
        if not names or not answer.isdigit():
            return answer
        number = int(answer)
        if 1 <= number <= len(names):
            return names[number - 1]
        ui.warn(f"{label}: enter 1-{len(names)} or a device name, please try again.")
    raise WizardAborted(f"{label}: too many invalid answers, giving up.")


def run_wizard(path: str | os.PathLike[str] | None = None) -> int:
    try:
        return _run_wizard(path)
    except WizardAborted as exc:
        ui.error(str(exc))
        ui.error("Nothing was saved.")
        return 1
    except KeyboardInterrupt:
        print()
        ui.info("Setup cancelled. Nothing was saved.")
        return 130
    except Exception as exc:
        # Credentials the user just typed live in this call stack; report
        # only the exception type to avoid leaking them into a traceback.
        ui.error(f"Setup failed: {type(exc).__name__}")
        return 1


def _run_wizard(path: str | os.PathLike[str] | None) -> int:
    from . import ai, notify, platform_check  # lazy imports for --help speed

    resolved = config_path(path)
    raw = read_raw(resolved)
    cfg = merge_defaults(raw)

    ui.rule("VoiceRecon setup")
    ui.info(f"Config file: {resolved}")
    ui.info("Press Enter to keep the current value. Nothing is saved until every step is done.\n")

    if platform_check.is_macos():
        ui.info(f"note: {platform_check.MACOS_LOOPBACK_HINT}\n")

    ui.info("1) Save directory (transcript files land here)")
    cfg["save_dir"] = _ask("  save directory", cfg["save_dir"])

    ui.info("\n2) Silence threshold (seconds of quiet before an utterance is cut)")
    cfg["speech_silence_seconds"] = _ask_float(
        "  silence seconds", cfg["speech_silence_seconds"], minimum=0
    )

    ui.info("\n3) Whisper model size")
    cfg["whisper_model_size"] = _ask_whisper_size(str(cfg["whisper_model_size"]))

    ui.info("\n4) Audio devices (Enter = auto-detect defaults)")
    devices = audio.enumerate_devices()
    if not devices["input"] and not devices["loopback"]:
        ui.info("   (no devices could be enumerated — soundcard may need to be installed)")
    cfg["input_device"] = _ask_device(
        "microphone",
        "Microphones detected",
        devices["input"],
        devices["default_input"],
        str(cfg["input_device"]),
    )
    cfg["loopback_device"] = _ask_device(
        "loopback / system audio",
        "Loopback / system-audio devices detected",
        devices["loopback"],
        devices["default_loopback"],
        str(cfg["loopback_device"]),
    )

    ui.info(
        "\n5) AI model + credentials (only needed if you plan to use --listen <preset>;"
        " Enter to skip and stay in transcript-only mode)"
    )
    cfg["model"] = _ask_choice(
        "AI model", MODEL_CHOICES, str(cfg["model"]), default=DEFAULT_MODEL
    )
    env_key = os.environ.get(ENV_API_KEY, "").strip()
    if env_key:
        ui.info(f"  {ENV_API_KEY} is set ({ui.mask(env_key)}); it wins over this file at runtime.")
    cfg["anthropic_api_key"] = _ask("  Anthropic API key", cfg["anthropic_api_key"], secret=True)
    cfg["telegram_bot_token"] = _ask("  Telegram bot token", cfg["telegram_bot_token"], secret=True)
    cfg["telegram_chat_id"] = _ask("  Telegram chat ID", cfg["telegram_chat_id"], secret=True)

    has_ai = bool(str(cfg["anthropic_api_key"]).strip())
    has_tg = bool(str(cfg["telegram_bot_token"]).strip() and str(cfg["telegram_chat_id"]).strip())

    if has_ai or has_tg:
        ui.rule("Verifying credentials")
        if has_ai:
            ok_ai, msg_ai = ai.verify_key(cfg)
            ui.info(("  OK   " if ok_ai else "  FAIL ") + f"AI: {msg_ai}")
        if has_tg:
            ok_tg, msg_tg = notify.verify_credentials(
                str(cfg["telegram_bot_token"]), str(cfg["telegram_chat_id"])
            )
            ui.info(("  OK   " if ok_tg else "  FAIL ") + f"Telegram: {msg_tg}")

    try:
        validate_config(merge_defaults(cfg))
    except ConfigError as exc:
        ui.error(str(exc))
        ui.error("Nothing was saved. Run 'voicerecon --configure' again.")
        return 1

    saved_to = save(cfg, resolved)
    ui.rule()
    ui.info(f"Config saved to {saved_to}")
    return 0
