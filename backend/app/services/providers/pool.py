"""Ordered failover pool.

Providers are tried best-first. Each failure mode triggers a specific,
narrow reaction so a single flaky key never blocks the whole app.
"""

import asyncio
import logging
import random
import time

from pydantic import BaseModel

from app.config import settings
from app.services.providers.base import (
    AllProvidersExhausted,
    AuthFailed,
    ModelNotFound,
    PermanentError,
    PromptTooLarge,
    Provider,
    ProviderError,
    ProviderResult,
    QuotaExceeded,
    TransientError,
)
from app.services.providers.gemini_provider import GeminiProvider
from app.services.providers.openai_compat_provider import OpenAICompatProvider

logger = logging.getLogger(__name__)

# Blacklisted models are remembered for the container's lifetime.
_BLACKLISTED: set[str] = set()


def _transient_cooldown() -> int:
    """Seconds to cool a model down after it exhausts its transient retries.

    Jittered so several clients that tripped the same 503 do not all return
    at the same instant and reproduce the spike.
    """
    base = settings.TRANSIENT_COOLDOWN_SECONDS
    if base <= 0:
        return 0
    return base + random.randint(0, max(1, base // 4))


def build_providers() -> list[Provider]:
    """Construct the ordered pool from environment configuration."""
    providers: list[Provider] = []

    for credential in settings.effective_gemini_credentials:
        for model in settings.GEMINI_MODEL_CHAIN:
            providers.append(
                GeminiProvider(
                    alias=credential.alias,
                    model=model,
                    api_key=credential.api_key,
                    fingerprint=credential.fingerprint,
                )
            )

    # Groq is a genuinely independent quota pool, so it sits last as a
    # free safety net. Open-weight models follow JSON schemas less reliably,
    # which is exactly why they are fallback rather than primary.
    if settings.GROQ_API_KEY:
        for model in settings.GROQ_MODEL_CHAIN:
            providers.append(
                OpenAICompatProvider(
                    alias="groq",
                    model=model,
                    api_key=settings.GROQ_API_KEY,
                    base_url=settings.GROQ_BASE_URL,
                    max_input_chars=settings.GROQ_MAX_INPUT_CHARS,
                    fingerprint=settings.GROQ_API_KEY[-4:],
                    vendor_label="groq",
                )
            )

    return providers


def reset_blacklist() -> None:
    """Clear session-level model blacklist (used by tests and on config change)."""
    _BLACKLISTED.clear()


class ProviderPool:
    def __init__(self, providers: list[Provider] | None = None):
        self._providers: list[Provider] = (
            providers if providers is not None else build_providers()
        )
        # Per-provider lock so concurrent requests queue instead of stampeding
        # one rate-limited key.
        self._locks: dict[str, asyncio.Lock] = {
            p.alias + "/" + p.model: asyncio.Lock() for p in self._providers
        }

    @property
    def providers(self) -> list[Provider]:
        return self._providers

    def _lock_for(self, provider: Provider) -> asyncio.Lock:
        return self._locks[provider.alias + "/" + provider.model]

    async def generate_json(
        self,
        schema: type[BaseModel],
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> ProviderResult:
        max_tokens = max_output_tokens or settings.MAX_OUTPUT_TOKENS
        prompt_chars = len(prompt)
        attempts: list[dict] = []
        cooldown_times: list[int] = []

        for provider in self._providers:
            label = f"{provider.alias}/{provider.model}"

            if provider.model in _BLACKLISTED:
                attempts.append(
                    {"alias": provider.alias, "model": provider.model, "reason": "model_blacklisted"}
                )
                continue

            if provider.status.disabled:
                attempts.append(
                    {"alias": provider.alias, "model": provider.model, "reason": "credential_disabled"}
                )
                continue

            if not provider.can_handle(prompt_chars):
                # Not a failure: this provider simply cannot take this input.
                attempts.append(
                    {
                        "alias": provider.alias,
                        "model": provider.model,
                        "reason": "prompt_too_large",
                    }
                )
                continue

            remaining = provider.status.cooldown_remaining
            if remaining > 0:
                attempts.append(
                    {
                        "alias": provider.alias,
                        "model": provider.model,
                        "reason": "cooldown",
                        "cooldown": remaining,
                    }
                )
                cooldown_times.append(remaining)
                continue

            async with self._lock_for(provider):
                # Re-check after waiting on the lock; another request may have
                # tripped this provider's cooldown while we were queued.
                if provider.status.cooldown_remaining > 0:
                    remaining = provider.status.cooldown_remaining
                    attempts.append(
                        {
                            "alias": provider.alias,
                            "model": provider.model,
                            "reason": "cooldown",
                            "cooldown": remaining,
                        }
                    )
                    cooldown_times.append(remaining)
                    continue

                try:
                    result = await asyncio.to_thread(
                        provider.generate_json, schema, prompt, max_tokens
                    )
                    logger.info("LLM request served by %s", label)
                    return ProviderResult(result, label)
                except QuotaExceeded as exc:
                    wait = exc.retry_after or 60
                    provider.status.cooldown_until = time.time() + wait
                    cooldown_times.append(wait)
                    attempts.append(
                        {
                            "alias": provider.alias,
                            "model": provider.model,
                            "reason": "quota_exceeded",
                            "cooldown": wait,
                        }
                    )
                    logger.warning("Quota exceeded on %s, cooling down %ss", label, wait)
                except ModelNotFound as exc:
                    _BLACKLISTED.add(provider.model)
                    attempts.append(
                        {
                            "alias": provider.alias,
                            "model": provider.model,
                            "reason": "model_not_found",
                            "detail": str(exc)[:200],
                        }
                    )
                    logger.warning("Model %s not found, blacklisting", provider.model)
                except AuthFailed as exc:
                    provider.status.disabled = True
                    attempts.append(
                        {
                            "alias": provider.alias,
                            "model": provider.model,
                            "reason": "auth_failed",
                            "detail": str(exc)[:200],
                        }
                    )
                    logger.error("Auth failed on %s, disabling credential", label)
                except PromptTooLarge as exc:
                    attempts.append(
                        {
                            "alias": provider.alias,
                            "model": provider.model,
                            "reason": "prompt_too_large",
                            "detail": str(exc)[:200],
                        }
                    )
                except PermanentError as exc:
                    # Retrying a request the provider has already rejected on
                    # its merits cannot help. Fail over immediately.
                    attempts.append(
                        {
                            "alias": provider.alias,
                            "model": provider.model,
                            "reason": "permanent_error",
                            "detail": str(exc)[:200],
                        }
                    )
                    logger.warning("Permanent failure on %s, advancing: %s", label, exc)
                except TransientError as exc:
                    # Two quick retries on the same provider, then move on.
                    for attempt in range(2):
                        await asyncio.sleep(1.5 * (attempt + 1))
                        try:
                            result = await asyncio.to_thread(
                                provider.generate_json, schema, prompt, max_tokens
                            )
                            logger.info("LLM request served by %s after retry", label)
                            return ProviderResult(result, label)
                        except PermanentError:
                            # A retry that comes back permanently rejected
                            # means the original failure was misclassified.
                            # Stop retrying this provider and let the outer
                            # handler record it and fail over.
                            last = exc
                            break
                        except ProviderError as retry_exc:
                            last = retry_exc
                    attempts.append(
                        {
                            "alias": provider.alias,
                            "model": provider.model,
                            "reason": "transient_error",
                            "detail": str(locals().get("last", exc))[:200],
                        }
                    )
                    # A model that has just failed every retry is overloaded,
                    # not broken. Google's own 503 wording says the spike is
                    # usually temporary, so cool it briefly and let the next
                    # request start elsewhere instead of re-running the same
                    # doomed retry ladder.
                    cooldown = _transient_cooldown()
                    if cooldown > 0:
                        provider.status.cooldown_until = time.time() + cooldown
                        cooldown_times.append(cooldown)
                    logger.warning(
                        "Transient failures on %s, cooling down %ss", label, cooldown
                    )
                except ProviderError as exc:
                    attempts.append(
                        {
                            "alias": provider.alias,
                            "model": provider.model,
                            "reason": "provider_error",
                            "detail": str(exc)[:200],
                        }
                    )

        raise AllProvidersExhausted(
            attempts=attempts,
            retry_after=min(cooldown_times) if cooldown_times else None,
        )

    def status(self, detailed: bool = True) -> list[dict]:
        return [p.status_dict(detailed=detailed) for p in self._providers]

    reset_blacklist = staticmethod(reset_blacklist)


_pool: ProviderPool | None = None


def get_pool() -> ProviderPool:
    global _pool
    if _pool is None:
        _pool = ProviderPool()
    return _pool