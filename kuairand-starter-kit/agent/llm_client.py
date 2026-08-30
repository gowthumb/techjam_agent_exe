"""Provider-neutral client for OpenAI-compatible chat-completion APIs."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


class LLMError(RuntimeError):
    """Base error for a failed compatible-provider request."""


class LLMAuthenticationError(LLMError):
    """The provider rejected the configured API credentials."""


class LLMQuotaError(LLMError):
    """The provider reported a quota or rate-limit failure."""


class LLMNetworkError(LLMError):
    """The provider could not be reached before the request completed."""


class LLMModelError(LLMError):
    """The provider rejected the configured model name or request."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


_ROOT = Path(__file__).resolve().parents[2]
_MODEL_ANNOUNCED = False
_QUOTA_PAUSE_COUNT = 0
_ROLE_TEMPERATURES = {"CODER": 0.15, "DEBUGGER": 0.15, "PLANNER": 0.6}


def _load_configuration() -> tuple[str, str, str]:
    load_dotenv(_ROOT / ".env")
    load_dotenv(_ROOT / "kuairand-starter-kit" / ".env")
    os.environ.setdefault("LLM_API_KEY", os.environ.get("GLM_API_KEY", ""))
    os.environ.setdefault("LLM_BASE_URL", os.environ.get("GLM_BASE_URL", ""))
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not api_key or not base_url or not model:
        raise LLMError("LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL must be configured.")
    return api_key, base_url, model


def resolve_model(role: str) -> str:
    """Return a role-specific compatible model, falling back to LLM_MODEL."""
    _, _, shared_model = _load_configuration()
    return os.environ.get(role.upper() + "_MODEL") or shared_model


def resolve_temperature(role: str) -> float | None:
    """Return a role-specific temperature, preserving provider defaults for unknown roles."""
    role = role.upper()
    configured = os.environ.get(role + "_TEMPERATURE")
    if configured is None:
        return _ROLE_TEMPERATURES.get(role)
    try:
        return float(configured)
    except ValueError as error:
        raise LLMError(role + "_TEMPERATURE must be a number.") from error


def _raise_provider_error(error: Exception) -> None:
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        raise LLMAuthenticationError("LLM authentication failed. Verify the configured credentials.") from error
    if isinstance(error, RateLimitError):
        raise LLMQuotaError("LLM request was rate-limited or the provider quota was exhausted.") from error
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        raise LLMNetworkError("LLM network request failed or timed out.") from error
    if isinstance(error, (NotFoundError, BadRequestError)):
        raise LLMModelError("LLM provider rejected the configured model name or request.") from error
    raise LLMError("LLM provider request failed: %s" % type(error).__name__) from error


def _rate_limit_delay(error: RateLimitError, default_backoff_s: float) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after)) + 2.0
        except ValueError:
            pass
    reset_at = headers.get("x-ratelimit-reset") or headers.get("ratelimit-reset")
    if reset_at:
        try:
            return max(0.0, float(reset_at) - datetime.now(timezone.utc).timestamp()) + 2.0
        except ValueError:
            pass
    return default_backoff_s


def reset_quota_pause_budget() -> None:
    """Reset the process-local quota retry budget for a newly started run."""
    global _QUOTA_PAUSE_COUNT
    _QUOTA_PAUSE_COUNT = 0


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def call_llm(
    system_prompt: str, user_prompt: str, *, max_tokens: int = 4096, model: str | None = None,
    temperature: float | None = None, max_quota_pauses: int = 3, role: str = "UNSPECIFIED",
) -> LLMResponse:
    """Send one chat-completion request and return provider-reported token usage."""
    global _MODEL_ANNOUNCED, _QUOTA_PAUSE_COUNT
    api_key, base_url, default_model = _load_configuration()
    selected_model = model or default_model
    request_timeout_s = float(os.environ.get("LLM_REQUEST_TIMEOUT_S", "120"))
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout_s)
    default_backoff_s = float(os.environ.get("LLM_RATE_LIMIT_BACKOFF_S", "300"))
    while True:
        request_started_at = time.monotonic()
        print(
            "[%s] LLM request starting: role=%s model=%s prompt_chars=%d"
            % (_timestamp(), role, selected_model, len(system_prompt) + len(user_prompt))
        )
        try:
            request = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "extra_headers": {
                    "HTTP-Referer": "https://github.com/gowthumb/techjam_agent_exe",
                },
            }
            if temperature is not None:
                request["temperature"] = temperature
            completion = client.chat.completions.create(**request)
            print(
                "[%s] LLM response received: role=%s model=%s elapsed_s=%.2f"
                % (_timestamp(), role, selected_model, time.monotonic() - request_started_at)
            )
            break
        except RateLimitError as error:
            if _QUOTA_PAUSE_COUNT >= max_quota_pauses:
                raise LLMQuotaError("LLM quota remained unavailable after %d pause cycles." % max_quota_pauses) from error
            delay_s = _rate_limit_delay(error, default_backoff_s)
            _QUOTA_PAUSE_COUNT += 1
            print(
                "[%s] LLM pause: role=%s model=%s reason=%s sleep_s=%.1f cycle=%d/%d"
                % (_timestamp(), role, selected_model, type(error).__name__, delay_s, _QUOTA_PAUSE_COUNT, max_quota_pauses)
            )
            time.sleep(delay_s)
        except Exception as error:
            _raise_provider_error(error)

    if not _MODEL_ANNOUNCED:
        print("LLM model: " + selected_model)
        _MODEL_ANNOUNCED = True
    if not completion.choices or completion.choices[0].message.content is None:
        raise LLMError("LLM provider returned no message content.")
    if completion.usage is None:
        raise LLMError("LLM provider did not return token usage.")
    return LLMResponse(
        text=completion.choices[0].message.content,
        input_tokens=int(completion.usage.prompt_tokens),
        output_tokens=int(completion.usage.completion_tokens),
    )