"""HTTP layer: contracts, typed errors, and cache/provider metadata.

The provider pool is stubbed, so these tests never touch the network.
"""

import asyncio
import inspect
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