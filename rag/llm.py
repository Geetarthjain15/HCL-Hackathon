"""LLM providers with automatic failover.

Order: Groq primary -> Groq smaller model -> Gemini. A 429 on the free tier is
the single most likely thing to break a demo, so it is handled rather than
raised.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import requests

from .config import settings


class ProviderError(RuntimeError):
    """Raised when every configured provider has failed."""


@dataclass
class Provider:
    name: str
    kind: str  # "groq" | "gemini"
    model: str


def available_providers() -> list[Provider]:
    out: list[Provider] = []
    if settings.groq_key:
        out.append(Provider(f"Groq / {settings.groq_model}", "groq", settings.groq_model))
        if settings.groq_fallback_model != settings.groq_model:
            out.append(
                Provider(
                    f"Groq / {settings.groq_fallback_model}",
                    "groq",
                    settings.groq_fallback_model,
                )
            )
    if settings.gemini_key:
        out.append(
            Provider(f"Gemini / {settings.gemini_model}", "gemini", settings.gemini_model)
        )
    return out


# --- groq --------------------------------------------------------------------


def _groq_client():
    from groq import Groq

    return Groq(api_key=settings.groq_key, timeout=settings.request_timeout)


def _groq_call(model: str, system: str, user: str, stream: bool):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _groq_client().chat.completions.create(
        model=model, messages=messages, temperature=settings.temperature, stream=stream
    )


# --- gemini ------------------------------------------------------------------


def _gemini_call(model: str, system: str, user: str) -> str:
    r = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent",
        headers={"x-goog-api-key": settings.gemini_key},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            # This model spends tokens on hidden reasoning before emitting any
            # text, so a small cap returns an empty response, not a short one.
            "generationConfig": {
                "maxOutputTokens": 2048,
                "temperature": settings.temperature,
            },
        },
        timeout=settings.request_timeout,
    )
    r.raise_for_status()
    candidates = r.json().get("candidates") or []
    if not candidates:
        raise ProviderError("Gemini returned no candidates (likely a safety block)")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise ProviderError(
            f"Gemini returned empty text (finishReason="
            f"{candidates[0].get('finishReason')})"
        )
    return text


# --- public ------------------------------------------------------------------


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate" in text and "limit" in text or "quota" in text


def complete(system: str, user: str, on_fallback=None) -> tuple[str, str]:
    """Return (text, provider_name), trying each provider in turn."""
    providers = available_providers()
    if not providers:
        raise ProviderError(
            "No API key configured. Put GROQ_API_KEY or GEMINI_API_KEY in .env"
        )

    errors = []
    for provider in providers:
        for attempt in range(2):
            try:
                if provider.kind == "groq":
                    resp = _groq_call(provider.model, system, user, stream=False)
                    return resp.choices[0].message.content, provider.name
                return _gemini_call(provider.model, system, user), provider.name
            except Exception as exc:
                if _is_rate_limit(exc) and attempt == 0:
                    time.sleep(2)  # one short backoff before giving up on it
                    continue
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                if on_fallback:
                    on_fallback(provider.name, exc)
                break

    raise ProviderError("All providers failed.\n" + "\n".join(errors))


def stream(system: str, user: str, on_fallback=None) -> Iterator[str]:
    """Yield text as it arrives; the last provider that works wins.

    Gemini is not streamed here, so it yields one block. The final element
    yielded is always a sentinel dict with the provider name.
    """
    providers = available_providers()
    if not providers:
        raise ProviderError(
            "No API key configured. Put GROQ_API_KEY or GEMINI_API_KEY in .env"
        )

    errors = []
    for provider in providers:
        try:
            if provider.kind == "groq":
                got_any = False
                for part in _groq_call(provider.model, system, user, stream=True):
                    delta = part.choices[0].delta.content
                    if delta:
                        got_any = True
                        yield delta
                if not got_any:
                    raise ProviderError("empty stream")
            else:
                yield _gemini_call(provider.model, system, user)
            yield {"provider": provider.name}  # sentinel
            return
        except Exception as exc:
            errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
            if on_fallback:
                on_fallback(provider.name, exc)

    raise ProviderError("All providers failed.\n" + "\n".join(errors))
