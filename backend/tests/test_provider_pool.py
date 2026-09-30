"""Failover pool behaviour, using stub providers so no network is required."""

import asyncio

import pytest
from pydantic import BaseModel

from app.services.providers import pool as pool_module
from app.services.providers.base import (
    AllProvidersExhausted,
    AuthFailed,
    ModelNotFound,
    Provider,
    QuotaExceeded,
    TransientError,
)
from app.services.providers.pool import ProviderPool, reset_blacklist


class Out(BaseModel):
    value: str


class StubProvider(Provider):
    """Raises whatever it was configured with, or returns a payload."""

    vendor = "stub"

    def __init__(self, alias, model, behaviour=None, payload=None, max_chars=100_000):
        super().__init__(alias=alias, model=model, api_key="fake-key-abcd")
        self.behaviour = behaviour
        self.payload = payload or {"value": f"from-{alias}"}
        self._max_chars = max_chars
        self.calls = 0

    @property
    def max_input_chars(self):
        return self._max_chars

    def generate_json(self, schema, prompt, max_output_tokens):
        self.calls += 1
        if self.behaviour is not None:
            raise self.behaviour
        return dict(self.payload)


def run(coro):
    return asyncio.run(coro)


def test_success_on_first_provider():
    first = StubProvider("projA", "gemini-3.6-flash")
    second = StubProvider("projB", "gemini-3.5-flash")
    pool = ProviderPool([first, second])

    outcome = run(pool.generate_json(Out, "prompt"))
    assert outcome.data == {"value": "from-projA"}
    assert outcome.label == "projA/gemini-3.6-flash"
    assert second.calls == 0, "should not have tried the second provider"


def test_quota_exceeded_fails_over_to_next():
    first = StubProvider(
        "projA", "gemini-3.6-flash", behaviour=QuotaExceeded("429", retry_after=42)
    )
    second = StubProvider("projB", "gemini-3.5-flash")
    pool = ProviderPool([first, second])

    outcome = run(pool.generate_json(Out, "prompt"))
    assert outcome.data == {"value": "from-projB"}
    assert outcome.label == "projB/gemini-3.5-flash"
    assert first.status.cooldown_remaining > 0, "quota failure must start cooldown"
    assert 40 <= first.status.cooldown_remaining <= 42


def test_cooldown_is_skipped_on_subsequent_calls():
    first = StubProvider(
        "projA", "gemini-3.6-flash", behaviour=QuotaExceeded("429", retry_after=300)
    )
    second = StubProvider("projB", "gemini-3.5-flash")
    pool = ProviderPool([first, second])

    run(pool.generate_json(Out, "prompt"))
    first.calls = 0
    run(pool.generate_json(Out, "prompt"))

    assert first.calls == 0, "provider in cooldown must be skipped, not re-called"
    assert second.calls == 2


def test_model_not_found_blacklists_model_for_session():
    reset_blacklist()
    dead = StubProvider("projA", "gemini-dead", behaviour=ModelNotFound("404"))
    live = StubProvider("projA", "gemini-live")
    pool = ProviderPool([dead, live])

    result = run(pool.generate_json(Out, "prompt"))
    assert result.data == {"value": "from-projA"}
    assert "gemini-dead" in pool_module._BLACKLISTED

    # A brand-new provider for the same dead model must not be attempted.
    dead_again = StubProvider("projA", "gemini-dead", behaviour=ModelNotFound("404"))
    fresh_pool = ProviderPool([dead_again, live])
    run(fresh_pool.generate_json(Out, "prompt"))

    assert dead_again.calls == 0, "blacklisted model must never be called again"
    reset_blacklist()


def test_auth_failure_disables_credential():
    first = StubProvider("bad", "gemini-3.6-flash", behaviour=AuthFailed("401"))
    second = StubProvider("good", "gemini-3.5-flash")
    pool = ProviderPool([first, second])

    result = run(pool.generate_json(Out, "prompt"))
    assert result.data == {"value": "from-good"}
    assert first.status.disabled is True


def test_prompt_too_large_skips_without_counting_as_failure():
    small = StubProvider("groq", "llama-70b", max_chars=100)
    large = StubProvider("gemini", "gemini-3.6-flash", max_chars=100_000)
    pool = ProviderPool([small, large])

    result = run(pool.generate_json(Out, "x" * 500))
    assert result.data == {"value": "from-gemini"}
    assert small.calls == 0, "oversized provider should be skipped, never called"
    assert small.status.failures == 0


def test_transient_error_recovers_on_retry():
    class FlakyProvider(StubProvider):
        def generate_json(self, schema, prompt, max_output_tokens):
            self.calls += 1
            if self.calls == 1:
                raise TransientError("503 boom")
            return {"value": "recovered"}

    flaky = FlakyProvider("flaky", "gemini-3.6-flash")
    pool = ProviderPool([flaky])

    result = run(pool.generate_json(Out, "prompt"))
    assert result.data == {"value": "recovered"}
    assert flaky.calls == 2


def test_all_providers_failing_raises_typed_error_with_retry_info():
    first = StubProvider("projA", "m1", behaviour=QuotaExceeded("429", retry_after=90))
    second = StubProvider("projB", "m2", behaviour=ModelNotFound("404"))
    pool = ProviderPool([first, second])

    with pytest.raises(AllProvidersExhausted) as info:
        run(pool.generate_json(Out, "prompt"))

    exc = info.value
    assert exc.earliest_retry == 90
    reasons = {item["reason"] for item in exc.attempts}
    assert "quota_exceeded" in reasons
    assert "model_not_found" in reasons
    # Credentials must never leak into error payloads.
    assert "fake-key-abcd" not in str(exc)


def test_no_providers_configured_raises_immediately():
    pool = ProviderPool([])
    with pytest.raises(AllProvidersExhausted):
        run(pool.generate_json(Out, "prompt"))


def test_status_never_exposes_full_key():
    provider = StubProvider("projA", "gemini-3.6-flash")
    payload = provider.status_dict()
    assert payload["fingerprint"] == "abcd"
    assert "fake-key" not in str(payload)