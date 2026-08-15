"""AI dispatcher — text-only.

VoiceRecon sends transcribed segments to an AI with a preset-defined system
prompt. There is no image path; unlike ScreenRecon's vision.py this module
only builds text messages.

Currently only Anthropic is wired up. A future OpenAI / Google provider
would land here as another module-level function plus a routing branch in
``ask_streaming``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

MAX_TOKENS = 4096
EFFORT = "low"


@dataclass(frozen=True)
class Reply:
    """One AI call. When ``ok`` is False, ``text`` is a readable error message."""

    ok: bool
    text: str


def ask_streaming(
    cfg: Mapping[str, Any],
    system_prompt: str,
    user_text: str,
    on_delta: Callable[[str], None],
) -> Reply:
    """Stream a response from the AI. ``on_delta`` is called with each text chunk.

    Never raises. On failure returns ``Reply(ok=False, text=<message>)`` so
    the segment loop keeps running.
    """
    return _anthropic_ask(cfg, system_prompt, user_text, on_delta)


def verify_key(cfg: Mapping[str, Any]) -> tuple[bool, str]:
    """Cheap probe used by the setup wizard."""
    api_key = str(cfg.get("anthropic_api_key") or "").strip()
    model = str(cfg.get("model") or "").strip()
    if not api_key:
        return False, "No API key entered."
    try:
        import anthropic
    except ImportError:
        return False, "Missing dependency 'anthropic'; cannot verify."
    try:
        anthropic.Anthropic(api_key=api_key).models.retrieve(model)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status == 404:
            return False, f"Key works, but model {model!r} does not exist."
        from . import ui
        return False, ui.scrub(f"AI API call failed: {type(exc).__name__}: {exc}", [api_key])
    return True, f"Key is valid and model {model} is available."


# --------------------------------------------------------------------------- #
# Anthropic implementation
# --------------------------------------------------------------------------- #


_effort_unsupported: set[str] = set()


def _anthropic_ask(
    cfg: Mapping[str, Any],
    system_prompt: str,
    user_text: str,
    on_delta: Callable[[str], None],
) -> Reply:
    api_key = str(cfg.get("anthropic_api_key") or "")
    model = str(cfg.get("model") or "")

    try:
        import anthropic
    except ImportError:
        return Reply(
            False,
            "Missing dependency 'anthropic'. Install with: pip install voicerecon",
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as exc:
        return Reply(False, _translate(exc, [api_key]))

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
    }
    use_effort = model not in _effort_unsupported

    for attempt in range(2):
        request = dict(payload)
        if use_effort:
            request["output_config"] = {"effort": EFFORT}
        chunks: list[str] = []
        try:
            with client.messages.stream(**request) as stream:
                for chunk in stream.text_stream:
                    chunks.append(chunk)
                    on_delta(chunk)
                final = stream.get_final_message()
        except Exception as exc:
            if use_effort and attempt == 0 and _is_parameter_error(exc):
                _effort_unsupported.add(model)
                use_effort = False
                continue
            return Reply(False, _translate(exc, [api_key]))

        text = "".join(chunks) or _extract_text(final)
        if not text:
            reason = getattr(final, "stop_reason", None)
            if reason == "refusal":
                return Reply(False, "The AI declined to answer (safety policy).")
            if reason == "max_tokens":
                return Reply(False, "The answer hit the output limit.")
            return Reply(False, "The AI returned an empty answer.")
        return Reply(True, text)

    return Reply(False, "The AI API call failed. Please retry.")


def _extract_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    joined = "\n".join(part for part in parts if part.strip())
    return joined.strip()


def _is_parameter_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if isinstance(exc, TypeError):
        return "output_config" in text or "effort" in text
    if getattr(exc, "status_code", None) != 400:
        return False
    return "effort" in text or "output_config" in text


def _translate(exc: Exception, secrets: list[str]) -> str:
    from . import ui
    try:
        import anthropic
    except ImportError:
        return ui.scrub(f"AI API call failed: {exc}", secrets)

    if isinstance(exc, anthropic.AuthenticationError):
        return (
            "The API key is invalid or revoked. "
            "Run 'voicerecon --configure' to set it again."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "This API key is not allowed to access that resource."
    if isinstance(exc, anthropic.RateLimitError):
        return "Too many requests. Please try again shortly."
    if isinstance(exc, anthropic.NotFoundError):
        return "Model or endpoint not found. Check the 'model' field."
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", "unknown")
        return f"The AI API returned {status}. Check your account credit or retry later."
    if isinstance(exc, anthropic.APITimeoutError):
        return "The AI API request timed out."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Network connection failed."
    return ui.scrub(f"AI API call failed: {type(exc).__name__}: {exc}", secrets)


