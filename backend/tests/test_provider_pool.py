"""Failover pool behaviour, using stub providers so no network is required."""

import asyncio

import pytest
from pydantic import BaseModel

from app.config import settings
from app.services.providers import pool as pool_module
from app.services.providers.base import (
    AllProvidersExhausted,
    AuthFailed,
    ModelNotFound,
    PermanentError,
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


def test_permanent_error_does_not_burn_retries():
    """A request the provider rejected on its merits must not be retried.

    This is the shape of the real gemini-3.6-flash failure: 3 calls, 3
    failures, then failover. Retrying a permanent rejection costs ~4.5s of
    sleep per provider and makes a healthy provider look broken.
    """
    first = StubProvider("projA", "gemini-3.6-flash", behaviour=PermanentError("400 invalid argument"))
    second = StubProvider("projB", "gemini-3.5-flash-lite")
    pool = ProviderPool([first, second])

    result = run(pool.generate_json(Out, "prompt"))

    assert result.data == {"value": "from-projB"}
    assert first.calls == 1, "a permanent error must be attempted exactly once"
    assert second.calls == 1


def test_permanent_error_during_retry_stops_immediately():
    """A retry that turns permanent must not consume remaining attempts."""

    class FlakyProvider(StubProvider):
        def generate_json(self, schema, prompt, max_output_tokens):
            self.calls += 1
            if self.calls == 1:
                raise TransientError("500 boom")
            raise PermanentError("400 invalid argument")

    flaky = FlakyProvider("flaky", "gemini-3.6-flash")
    fallback = StubProvider("projB", "gemini-3.5-flash-lite")
    pool = ProviderPool([flaky, fallback])

    result = run(pool.generate_json(Out, "prompt"))

    assert result.data == {"value": "from-projB"}
    assert flaky.calls == 2, "should stop after the first permanent reply"
    assert fallback.calls == 1


def test_all_permanent_failures_report_reason():
    first = StubProvider("projA", "m1", behaviour=PermanentError("400 invalid argument"))
    pool = ProviderPool([first])

    with pytest.raises(AllProvidersExhausted) as info:
        run(pool.generate_json(Out, "prompt"))

    assert {item["reason"] for item in info.value.attempts} == {"permanent_error"}
    assert first.calls == 1


def test_safety_block_is_permanent_not_transient():
    """Regression: a safety block was classified transient and retried."""
    from app.services.providers.gemini_provider import _translate

    class Blocked(Exception):
        pass

    assert isinstance(_translate(Blocked("Response was blocked by safety filters")), PermanentError)
    assert isinstance(_translate(Blocked("400 INVALID_ARGUMENT: bad schema")), PermanentError)
    assert isinstance(_translate(Blocked("503 service unavailable")), TransientError)
    assert isinstance(_translate(Blocked("429 resource exhausted")), QuotaExceeded)


def test_status_records_last_error_for_diagnosis():
    """A provider failing 3/3 must say why, not just how often.

    Uses a real GeminiProvider whose SDK call is patched to raise, so the
    recording path under test is the one production actually takes. A bare
    stub would skip it and prove nothing.
    """
    from app.services.providers import gemini_provider as gemini_module

    class UpstreamError(Exception):
        status = 400

    provider = gemini_module.GeminiProvider(
        alias="gemini-default", model="gemini-3.6-flash", api_key="fake-key-abcd"
    )

    class FakeModels:
        def generate_content(self, *args, **kwargs):
            raise UpstreamError("400 INVALID_ARGUMENT: bad response schema")

    provider._client = type("FakeClient", (), {"models": FakeModels()})()
    pool = ProviderPool([provider])

    with pytest.raises(AllProvidersExhausted):
        run(pool.generate_json(Out, "prompt"))

    payload = provider.status_dict()
    assert "INVALID_ARGUMENT" in payload["last_error"]
    assert payload["last_error_kind"] == "PermanentError"
    assert payload["last_error_at"] > 0
    # One attempt only: a permanent rejection must not be retried.
    assert provider.status.calls == 1


def test_503_message_containing_400_is_not_classified_permanent():
    """Regression: substring matching on '400' inside a 503 body."""
    from app.services.providers.gemini_provider import _translate

    class UpstreamError(Exception):
        status = 503

    assert isinstance(_translate(UpstreamError("503 upstream: 400 capacity")), TransientError)


def test_exhausted_transient_triggers_cooldown(monkeypatch):
    """A model that burned all its retries is overloaded, not broken.

    Without a cooldown every subsequent request re-runs the same doomed
    retry ladder, which is what a 503 demand spike looks like in practice.
    """
    monkeypatch.setattr(settings, "TRANSIENT_COOLDOWN_SECONDS", 30)

    class AlwaysTransient(StubProvider):
        def generate_json(self, schema, prompt, max_output_tokens):
            self.calls += 1
            raise TransientError("503 UNAVAILABLE high demand")

    first = AlwaysTransient("projA", "gemini-3.6-flash")
    second = StubProvider("projB", "gemini-3.5-flash-lite")
    pool = ProviderPool([first, second])

    run(pool.generate_json(Out, "prompt"))
    assert first.calls == 3, "initial attempt plus two retries"
    assert first.status.cooldown_remaining > 0
    assert first.status.available is False

    # The next request must not re-run the retry ladder on the same model.
    run(pool.generate_json(Out, "prompt"))
    assert first.calls == 3, "cooled-down provider must be skipped entirely"
    assert second.calls == 2


def test_transient_cooldown_is_jittered(monkeypatch):
    """Identical cooldowns would resynchronise the clients that caused them."""
    monkeypatch.setattr(settings, "TRANSIENT_COOLDOWN_SECONDS", 30)

    values = {pool_module._transient_cooldown() for _ in range(40)}
    assert len(values) > 1, "cooldown must not be a constant"
    assert all(30 <= value <= 37 for value in values)


def test_transient_cooldown_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "TRANSIENT_COOLDOWN_SECONDS", 0)

    class AlwaysTransient(StubProvider):
        def generate_json(self, schema, prompt, max_output_tokens):
            self.calls += 1
            raise TransientError("503 boom")

    first = AlwaysTransient("projA", "m1")
    pool = ProviderPool([first, StubProvider("projB", "m2")])

    run(pool.generate_json(Out, "prompt"))
    assert first.status.cooldown_remaining == 0


def test_permanent_error_does_not_trigger_cooldown(monkeypatch):
    """A permanent rejection says nothing about current load.

    Cooling down here would wrongly penalise a model that is perfectly
    healthy and simply disliked one particular request.
    """
    monkeypatch.setattr(settings, "TRANSIENT_COOLDOWN_SECONDS", 30)

    first = StubProvider("projA", "m1", behaviour=PermanentError("400 bad schema"))
    pool = ProviderPool([first, StubProvider("projB", "m2")])

    run(pool.generate_json(Out, "prompt"))

    assert first.status.cooldown_remaining == 0
    assert first.status.available is True


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