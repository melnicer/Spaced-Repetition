"""Provider abstraction shared by every LLM backend.

The pool only ever talks to a Provider, so adding a new vendor means
implementing `generate_json` and nothing else.
"""

import time
from dataclasses import dataclass, field

from pydantic import BaseModel


class ProviderError(Exception):
    """Base for classified upstream failures the pool knows how to react to."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceeded(ProviderError):
    """429 / RESOURCE_EXHAUSTED - cool this provider down."""


class ModelNotFound(ProviderError):
    """404 - the model is gone or retired; blacklist it for this session."""


class AuthFailed(ProviderError):
    """401 / 403 - the credential is bad; disable it."""


class TransientError(ProviderError):
    """5xx, timeouts, connection resets - worth retrying on the same provider."""


class PermanentError(ProviderError):
    """The request itself is unacceptable, so retrying cannot help.

    400 INVALID_ARGUMENT, malformed schema, and safety blocks land here.
    These used to be classified TransientError, which made the pool burn
    three calls and ~4.5s of sleep on an error that was never going to
    succeed, then advance as if the provider were merely unhealthy.
    """


class PromptTooLarge(ProviderError):
    """Request exceeds this provider's token window; skip rather than fail."""


class AllProvidersExhausted(Exception):
    """Every configured provider failed. Carries diagnostics for the client."""

    def __init__(self, attempts: list[dict], retry_after: int | None = None):
        self.attempts = attempts
        self.retry_after = retry_after
        reasons = "; ".join(
            f"{item['alias']}/{item['model']}: {item['reason']}" for item in attempts
        )
        super().__init__(reasons or "No providers configured")

    @property
    def earliest_retry(self) -> int | None:
        if self.retry_after is not None:
            return self.retry_after
        cooldowns = [
            item["cooldown"]
            for item in self.attempts
            if item.get("cooldown") is not None
        ]
        return min(cooldowns) if cooldowns else None


class ProviderResult:
    """Payload plus the provider that actually served it.

    Returning attribution with the data avoids the caller having to guess which
    provider in a chain answered, which is wrong as soon as requests overlap.
    """

    def __init__(self, data: dict, label: str):
        self.data = data
        self.label = label

    def get(self, key, default=None):
        return self.data.get(key, default)


@dataclass
class ProviderStatus:
    alias: str
    model: str
    vendor: str
    fingerprint: str = ""
    cooldown_until: float = 0.0
    disabled: bool = False
    calls: int = 0
    failures: int = 0
    blacklisted_models: list[str] = field(default_factory=list)
    # Truncated upstream text. Without this, a provider failing 3/3 looks
    # identical whether the cause is a safety block, a bad schema, or a 500,
    # and there is no way to tell from outside which one it is.
    last_error: str = ""
    last_error_kind: str = ""
    last_error_at: float = 0.0

    def note_error(self, kind: str, message: str) -> None:
        self.failures += 1
        self.last_error = message[:300]
        self.last_error_kind = kind
        self.last_error_at = time.time()

    @property
    def cooldown_remaining(self) -> int:
        remaining = self.cooldown_until - time.time()
        return max(0, int(remaining)) if remaining > 0 else 0

    @property
    def available(self) -> bool:
        return not self.disabled and self.cooldown_remaining == 0

    def to_dict(self, detailed: bool = True) -> dict:
        """Serialise for the status endpoint.

        `detailed=False` drops the credential fingerprint and the raw upstream
        error text. Both were published on an unauthenticated URL: the
        fingerprint is 4 characters of a live API key, and the error text can
        echo back prompt content from the request that failed.
        """
        payload = {
            "alias": self.alias,
            "model": self.model,
            "vendor": self.vendor,
            "available": self.available,
            "disabled": self.disabled,
            "cooldown_remaining": self.cooldown_remaining,
            "calls": self.calls,
            "failures": self.failures,
            "blacklisted_models": self.blacklisted_models,
        }
        if detailed:
            payload["fingerprint"] = self.fingerprint
            payload["last_error"] = self.last_error
            payload["last_error_kind"] = self.last_error_kind
            payload["last_error_at"] = self.last_error_at
        return payload


class Provider:
    """Base class for a single (credential, model) pair."""

    vendor = "unknown"

    def __init__(self, alias: str, model: str, api_key: str, fingerprint: str = ""):
        self.alias = alias
        self.model = model
        self.api_key = api_key
        self.status = ProviderStatus(
            alias=alias,
            model=model,
            vendor=self.vendor,
            fingerprint=fingerprint or (api_key[-4:] if len(api_key) >= 4 else "****"),
        )

    @property
    def max_input_chars(self) -> int:
        return 100_000

    def can_handle(self, prompt_chars: int) -> bool:
        """False means 'skip me', which is distinct from 'I failed'."""
        return prompt_chars <= self.max_input_chars

    def generate_json(self, schema: type[BaseModel], prompt: str, max_output_tokens: int) -> dict:
        raise NotImplementedError

    def status_dict(self, detailed: bool = True) -> dict:
        return self.status.to_dict(detailed=detailed)