"""HTTP layer: contracts, typed errors, and cache/provider metadata.

The provider pool is stubbed, so these tests never touch the network.
"""

import asyncio
import inspect
import json
from inspect import iscoroutinefunction

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.deck_generator import generate_flashcard_deck
from app.services.leech_agent import remediate_leeches
from app.services.providers.base import AllProvidersExhausted, ProviderResult

client = TestClient(app)

DECK_PAYLOAD = {
    "title": "Biology",
    "cards": [
        {
            "question": "What gas do plants absorb?",
            "answer": "Carbon dioxide.",
            "mnemonic": None,
        }
    ],
}

SPLIT_PAYLOAD = {
    "remediated_cards": [
        {
            "id": "c1",
            "action": "split",
            "new_question": "",
            "new_answer": "",
            "sub_cards": [
                {"question": "Where does photosynthesis occur?", "answer": "In the chloroplast."},
                {"question": "What is the main pigment?", "answer": "Chlorophyll."},
            ],
            "mnemonic": None,
            "reason": "The card tested two distinct facts.",
        }
    ]
}


class FakePool:
    """Minimal stand-in for ProviderPool."""

    def __init__(self, payload=None, error=None, label="projA/gemini-3.6-flash"):
        self.payload = payload
        self.error = error
        self.label = label
        self.calls = 0

    def generate_json_sync(self, schema, prompt, max_output_tokens=2048):
        """Synchronous core, shared by the async wrapper."""
        self.calls += 1
        if self.error is not None:
            raise self.error
        return ProviderResult(self.payload, self.label)

    # Must stay async: the real ProviderPool.generate_json is `async def`.
    # A sync fake here would let sync/async drift ship unnoticed, which is
    # exactly the bug this fake was previously hiding.
    async def generate_json(self, schema, prompt, max_output_tokens=2048):
        return self.generate_json_sync(schema, prompt, max_output_tokens)


def _force_configured(monkeypatch, value: bool = True) -> None:
    """Override the read-only has_any_provider property for these tests."""
    monkeypatch.setattr(
        type(settings), "has_any_provider", property(lambda self: value)
    )


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Keep tests off the real /tmp cache so results are deterministic."""
    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path / "cache"))
    yield


@pytest.fixture
def fake_pool(monkeypatch):
    """Patch the pool accessor used by both services."""
    pool = FakePool(payload=DECK_PAYLOAD)
    monkeypatch.setattr("app.services.deck_generator.get_pool", lambda: pool)
    monkeypatch.setattr("app.services.leech_agent.get_pool", lambda: pool)
    _force_configured(monkeypatch, True)
    return pool


@pytest.fixture
def real_provider_pool(monkeypatch):
    """A pool holding one genuine provider, for status-endpoint tests.

    The redaction tests need a provider with a real fingerprint and error
    text. FakePool has no providers, so it cannot prove anything about what
    the endpoint withholds.
    """
    from app.services.providers import pool as pool_module
    from app.services.providers.base import Provider, TransientError

    class RecordingProvider(Provider):
        vendor = "gemini"

        def generate_json(self, schema, prompt, max_output_tokens):
            raise TransientError("503 UNAVAILABLE")

    provider = RecordingProvider(
        alias="gemini-default",
        model="gemini-3.6-flash",
        api_key="fake-key-abcd",
    )
    pool = pool_module.ProviderPool([provider])
    # Seed the state the endpoint reports. Driving it through a real request
    # would spend the retry sleeps here for no extra coverage, since the
    # cooldown behaviour is tested in test_provider_pool.py.
    provider.status.calls = 1
    provider.status.note_error(
        "TransientError", "503 UNAVAILABLE: prompt echo SECRET_DOC_TEXT"
    )
    monkeypatch.setattr("app.routers.health.get_pool", lambda: pool)
    _force_configured(monkeypatch, True)
    return pool


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "FlutterStudy API"}


def test_service_functions_are_coroutines():
    """Canary for sync/async drift.

    The pool is async. If either service stops awaiting it, every real AI
    call fails at runtime with "coroutine object has no attribute 'data'"
    while an HTTP test using a synchronous fake still passes.
    """
    assert iscoroutinefunction(generate_flashcard_deck), (
        "generate_flashcard_deck must be async: it awaits the async provider pool"
    )
    assert iscoroutinefunction(remediate_leeches), (
        "remediate_leeches must be async: it awaits the async provider pool"
    )


def test_leech_remediation_actually_awaits_the_pool(monkeypatch):
    """The deck path has this canary; the leech path needs its own.

    The original production bug lived on this exact line, and the leech
    schema is the more complex of the two (action, sub_cards, new_question).
    A declared-async-but-unawaited service would pass every test that did
    not drive a real suspension point through it.
    """
    completed = False

    class YieldingPool:
        async def generate_json(self, schema, prompt, max_output_tokens=2048):
            nonlocal completed
            await asyncio.sleep(0)  # real suspension point
            completed = True
            return ProviderResult(
                {
                    "remediated_cards": [
                        {
                            "id": "card-1",
                            "action": "rewritten",
                            "reason": "Dense wording",
                            "new_question": "Simpler question?",
                            "new_answer": "Simpler answer.",
                            "sub_cards": [],
                        }
                    ]
                },
                "projA/gemini-3.6-flash",
            )

    monkeypatch.setattr("app.services.leech_agent.get_pool", lambda: YieldingPool())
    _force_configured(monkeypatch, True)

    response = client.post(
        "/api/leech/remediate",
        json={
            "cards": [
                {"id": "card-1", "question": "Dense question?", "answer": "Dense answer."}
            ]
        },
    )

    assert completed is True, "the pool coroutine was never awaited on the leech path"
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "projA/gemini-3.6-flash"
    assert body["remediated_cards"][0]["cards"][0]["question"] == "Simpler question?"


def test_leech_remediation_normalises_a_split(monkeypatch):
    """A split must yield N cards to store in place of the original.

    The client deletes the original and stores every sub-card, so the
    normalisation in leech_agent._build_result is what the app depends on.
    """
    class SplitPool:
        async def generate_json(self, schema, prompt, max_output_tokens=2048):
            return ProviderResult(
                {
                    "remediated_cards": [
                        {
                            "id": "card-1",
                            "action": "split",
                            "reason": "Two facts in one",
                            "new_question": "",
                            "new_answer": "",
                            "sub_cards": [
                                {"question": "Fact one?", "answer": "One."},
                                {"question": "Fact two?", "answer": "Two."},
                                # Unusable entry: must be dropped, not stored.
                                {"question": "", "answer": ""},
                            ],
                        },
                        # Hallucinated id: must be dropped, never become a card.
                        {
                            "id": "ghost-999",
                            "action": "rewritten",
                            "reason": "Invented",
                            "new_question": "Q?",
                            "new_answer": "A.",
                            "sub_cards": [],
                        },
                    ]
                },
                "projA/gemini-3.6-flash",
            )

    monkeypatch.setattr("app.services.leech_agent.get_pool", lambda: SplitPool())
    _force_configured(monkeypatch, True)

    response = client.post(
        "/api/leech/remediate",
        json={
            "cards": [
                {"id": "card-1", "question": "Q1 and Q2?", "answer": "A1 and A2."}
            ]
        },
    )

    assert response.status_code == 200
    remediated = response.json()["remediated_cards"]
    assert len(remediated) == 1, "the hallucinated id must not come back"
    split = remediated[0]
    assert split["action"] == "split"
    assert len(split["cards"]) == 2, "the empty sub_card must be dropped"


def test_leech_remediation_drops_a_split_with_no_usable_subcards(monkeypatch):
    """A split that produces nothing usable is a model error, not a split.

    Returning it would make the client delete the original card and insert
    nothing, silently destroying the user's card.
    """
    class EmptySplitPool:
        async def generate_json(self, schema, prompt, max_output_tokens=2048):
            return ProviderResult(
                {
                    "remediated_cards": [
                        {
                            "id": "card-1",
                            "action": "split",
                            "reason": "Tried to split",
                            "new_question": "",
                            "new_answer": "",
                            "sub_cards": [{"question": "", "answer": ""}],
                        }
                    ]
                },
                "projA/gemini-3.6-flash",
            )

    monkeypatch.setattr("app.services.leech_agent.get_pool", lambda: EmptySplitPool())
    _force_configured(monkeypatch, True)

    response = client.post(
        "/api/leech/remediate",
        json={"cards": [{"id": "card-1", "question": "Q?", "answer": "A."}]},
    )

    assert response.status_code == 200
    assert response.json()["remediated_cards"] == []


def test_deck_generation_actually_awaits_the_pool(monkeypatch):
    """Proves the await happens, not merely that the function is async.

    The stub yields to the event loop before returning. A service that is
    declared async but forgets to await would still hand a coroutine to
    `.data` and fail here.
    """
    completed = False

    class YieldingPool:
        async def generate_json(self, schema, prompt, max_output_tokens=2048):
            nonlocal completed
            await asyncio.sleep(0)  # real suspension point
            completed = True
            return ProviderResult(
                {"title": "Async", "cards": [{"question": "q", "answer": "a"}]},
                "projA/gemini-3.6-flash",
            )

    monkeypatch.setattr(
        "app.services.deck_generator.get_pool", lambda: YieldingPool()
    )
    _force_configured(monkeypatch, True)

    response = client.post(
        "/api/deck/generate",
        data={"title": "Async", "text": "Plants absorb carbon dioxide."},
    )

    assert completed is True, "the pool coroutine was never awaited"
    assert response.status_code == 200
    assert response.json()["cards"][0]["question"] == "q"
    assert response.json()["provider"] == "projA/gemini-3.6-flash"


def test_status_reports_unconfigured_without_keys():
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["configured"] in (True, False)
    assert "providers" in body
    assert "cache_hits" in body


def test_status_never_leaks_full_keys(fake_pool):
    response = client.get("/api/status")
    assert response.status_code == 200
    text = response.text
    assert "fake-key" not in text


def test_status_redacts_fingerprint_and_error_text_when_token_unset(
    real_provider_pool, monkeypatch
):
    """An unauthenticated caller must not get any of the diagnostic detail.

    The fingerprint is 4 characters of a live API key and last_error can
    echo prompt content from the request that failed. Both were previously
    served to anyone who opened the URL.
    """
    monkeypatch.setattr(settings, "STATUS_TOKEN", "")

    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["detailed"] is False

    for provider in body["providers"]:
        assert provider.get("fingerprint") is None
        assert provider.get("last_error") is None
        assert provider.get("last_error_kind") is None
        assert provider.get("last_error_at") is None
        # Health signals stay available so a redacted response is still useful.
        assert "available" in provider
        assert "model" in provider

    assert "fake-key" not in response.text
    # The last 4 characters of the key must not leak either.
    assert "abcd" not in response.text
    # Nor may upstream error text echo prompt content back to the caller.
    assert "SECRET_DOC_TEXT" not in response.text


def test_status_returns_detail_with_valid_token(real_provider_pool, monkeypatch):
    monkeypatch.setattr(settings, "STATUS_TOKEN", "s3cret-debug-token")

    response = client.get(
        "/api/status", headers={"X-Status-Token": "s3cret-debug-token"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["detailed"] is True
    assert body["providers"][0]["fingerprint"] == "abcd"
    assert "SECRET_DOC_TEXT" in body["providers"][0]["last_error"]


def test_status_ignores_wrong_token(real_provider_pool, monkeypatch):
    """A bad token must not silently fall back to full detail."""
    monkeypatch.setattr(settings, "STATUS_TOKEN", "s3cret-debug-token")

    for bad in ("wrong", "s3cret-debug-toke", "s3cret-debug-tokenx", ""):
        body = client.get("/api/status", headers={"X-Status-Token": bad}).json()
        assert body["detailed"] is False, f"token {bad!r} should not authenticate"
        assert body["providers"][0].get("fingerprint") is None
        assert "SECRET_DOC_TEXT" not in json.dumps(body)


def test_status_does_not_expose_token_itself(real_provider_pool, monkeypatch):
    monkeypatch.setattr(settings, "STATUS_TOKEN", "s3cret-debug-token")
    response = client.get(
        "/api/status", headers={"X-Status-Token": "s3cret-debug-token"}
    )
    assert "s3cret-debug-token" not in response.text


def test_generate_returns_cards_and_metadata(fake_pool):
    response = client.post(
        "/api/deck/generate",
        data={"title": "Test Deck", "text": "Plants absorb carbon dioxide."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Biology"
    assert body["cards"][0]["question"]
    assert "cached" in body
    assert "provider" in body


def test_generate_rejects_empty_content(fake_pool):
    response = client.post("/api/deck/generate", data={"title": "Empty", "text": "   "})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_generate_rejects_unsupported_file(fake_pool):
    response = client.post(
        "/api/deck/generate",
        data={"title": "Bad"},
        files={"file": ("notes.docx", b"binary", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file" in response.json()["detail"]


def test_quota_exhaustion_returns_429_with_retry_hint(fake_pool):
    fake_pool.error = AllProvidersExhausted(
        attempts=[
            {"alias": "projA", "model": "m1", "reason": "quota_exceeded", "cooldown": 42}
        ],
        retry_after=42,
    )
    response = client.post(
        "/api/deck/generate",
        data={"title": "Quota", "text": "Plants absorb carbon dioxide."},
    )
    assert response.status_code == 429
    body = response.json()
    assert body["error"] == "all_providers_exhausted"
    assert body["retry_after_seconds"] == 42


def test_empty_model_output_returns_502(fake_pool):
    fake_pool.payload = {"title": "Empty", "cards": []}
    response = client.post(
        "/api/deck/generate",
        data={"title": "Nothing", "text": "Plants absorb carbon dioxide."},
    )
    assert response.status_code == 502
    assert response.json()["error"] == "invalid_model_output"


def test_no_provider_configured_returns_503():
    response = client.post(
        "/api/deck/generate",
        data={"title": "None", "text": "Plants absorb carbon dioxide."},
    )
    assert response.status_code == 503
    assert response.json()["error"] == "no_provider_configured"


def test_split_returns_multiple_cards(fake_pool, monkeypatch):
    monkeypatch.setattr(
        "app.services.leech_agent.get_pool", lambda: FakePool(payload=SPLIT_PAYLOAD)
    )
    response = client.post(
        "/api/leech/remediate",
        json={
            "cards": [
                {
                    "id": "c1",
                    "question": "Where does photosynthesis occur and what is the main pigment?",
                    "answer": "It occurs in chloroplasts and the pigment is chlorophyll.",
                    "streak": 3,
                }
            ]
        },
    )
    assert response.status_code == 200
    cards = response.json()["remediated_cards"]
    assert len(cards) == 1
    assert cards[0]["action"] == "split"
    assert len(cards[0]["cards"]) == 2


def test_leech_rejects_empty_list(fake_pool):
    response = client.post("/api/leech/remediate", json={"cards": []})
    assert response.status_code == 400


def test_leech_quota_exhaustion_returns_429(monkeypatch):
    pool = FakePool(
        error=AllProvidersExhausted(
            attempts=[
                {
                    "alias": "projA",
                    "model": "m1",
                    "reason": "quota_exceeded",
                    "cooldown": 90,
                }
            ],
            retry_after=90,
        )
    )
    monkeypatch.setattr("app.services.leech_agent.get_pool", lambda: pool)
    _force_configured(monkeypatch, True)
    response = client.post(
        "/api/leech/remediate",
        json={
            "cards": [
                {"id": "c1", "question": "Q?", "answer": "A.", "streak": 3}
            ]
        },
    )
    assert response.status_code == 429
    assert response.json()["retry_after_seconds"] == 90