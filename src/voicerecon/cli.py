"""Argument parsing and subcommand routing."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__, config, presets, ui

PROG = "voicerecon"

EPILOG = """\
examples:
  voicerecon                       transcript-only listening (no AI)
  voicerecon --listen translate    listen + translate the other party's speech
  voicerecon --listen interview_candidate
                                   listen + interview-answer helper
  voicerecon --configure           first-time interactive setup
  voicerecon --show                print the current config (credentials masked)
  voicerecon --show-devices        list detected audio input / loopback devices
  voicerecon --presets             list built-in AI presets
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Capture microphone + system audio, transcribe locally, optionally "
            "send each completed segment to an AI with a scenario preset."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    parser.add_argument(
        "--listen",
        metavar="PRESET",
        help="enable AI-per-segment mode with the named preset (see --presets)",
    )
    parser.add_argument(
        "--configure", action="store_true", help="run the interactive setup wizard"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="print the current config (credentials masked) and exit",
    )
    parser.add_argument(
        "--show-devices",
        dest="show_devices",
        action="store_true",
        help="list detected audio input / loopback devices and exit",
    )
    parser.add_argument(
        "--presets",
        action="store_true",
        help="list built-in AI presets and exit",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        dest="config_path",
        help="use an alternative config file (default: ~/.config/voicerecon/config.json)",
    )
    return parser


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def _print_show(cfg: dict) -> None:
    ui.rule("VoiceRecon config")
    ui.info(f"Config file: {cfg.get('_path')}")
    ui.info("")
    ui.info(f"  Save directory:        {cfg['save_dir']}")
    ui.info(f"  Silence threshold:     {cfg['speech_silence_seconds']} s")
    ui.info(f"  Whisper model:         {cfg['whisper_model_size']}")
    ui.info(f"  Input device:          {cfg.get('input_device') or '(default)'}")
    ui.info(f"  Loopback device:       {cfg.get('loopback_device') or '(default)'}")
    ui.info(f"  AI model:              {cfg['model']}")
    ui.info(f"  Anthropic key:         {ui.mask(str(cfg['anthropic_api_key']))}")
    ui.info(f"  Telegram bot:          {ui.mask(str(cfg['telegram_bot_token']))}")
    ui.info(f"  Telegram chat:         {ui.mask(str(cfg['telegram_chat_id']))}")


def _print_devices() -> None:
    from . import audio

    devices = audio.enumerate_devices()
    ui.rule("Audio devices")
    _print_device_list("Microphones", devices["input"], devices["default_input"])
    ui.info("")
    _print_device_list(
        "Loopback / system-audio inputs", devices["loopback"], devices["default_loopback"]
    )


def _print_device_list(heading: str, names: list[str], default_name: str) -> None:
    from . import audio

    if not names:
        ui.info(f"{heading}: (none detected)")
        return
    ui.info(f"{heading}:")
    for line in audio.format_device_lines(names, default_name):
        ui.info(f"  {line}")


def _print_presets() -> None:
    ui.rule("Built-in AI presets")
    for name in presets.names():
        preset = presets.get(name)
        ui.info(f"  {name}")
        ui.info(f"    speaker filter: {preset.speaker_filter}")
        ui.info(f"    context:        {preset.context}")
        ui.info(f"    purpose:        {preset.description}")


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    exclusive = [
        name
        for name in ("configure", "show", "show_devices", "presets")
        if getattr(args, name)
    ]
    if len(exclusive) > 1:
        parser.error(
            f"--{exclusive[0].replace('_', '-')} and --{exclusive[1].replace('_', '-')} "
            "cannot be combined; run one at a time"
        )
    if exclusive and args.listen:
        parser.error(
            f"--{exclusive[0].replace('_', '-')} cannot be combined with --listen"
        )

    if args.presets:
        _print_presets()
        return 0
    if args.show_devices:
        _print_devices()
        return 0

    cfg: dict | None = None
    try:
        if args.configure:
            return config.run_wizard(args.config_path)

        if args.show:
            path = config.config_path(args.config_path)
            raw = config.read_raw(path)
            if not raw:
                ui.error(
                    f"No config at {path}. Run 'voicerecon --configure' to create one."
                )
                return 1
            cfg = config.merge_defaults(raw)
            cfg["_path"] = str(path)
            _print_show(cfg)
            return 0

        cfg = config.load(args.config_path)

        preset_name: str | None = None
        if args.listen:
            preset_name = args.listen
            try:
                presets.get(preset_name)
            except KeyError as exc:
                ui.error(str(exc))
                return 1
            config.require_credentials_for_ai(cfg)

        from . import runner
        return runner.run(cfg, preset_name)

    except config.ConfigError as exc:
        ui.error(str(exc))
        return 1
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        secrets = (
            [str(cfg.get(field) or "") for field in config.CREDENTIAL_FIELDS] if cfg else []
        )
        ui.error(ui.scrub(f"Unexpected error: {type(exc).__name__}: {exc}", secrets))
        ui.error("This is a bug. The message above is safe to report; it contains no credentials.")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
