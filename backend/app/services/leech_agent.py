"""Leech remediation via the provider pool, with disk caching."""

import json
import logging

from app.config import settings
from app.models.schemas import (
    FlashcardOut,
    LeechCardIn,
    RemediatedCardOut,
    RemediationAction,
    RemediationSchema,
)
from app.services.cache import ResponseCache
from app.services.providers.pool import get_pool

logger = logging.getLogger(__name__)

PROMPT = """You repair flashcards a student keeps failing.

For each card, diagnose the problem and apply exactly one remedy:

- rewritten: the wording was dense, ambiguous, or confusing.
- split: the card tested two or more distinct facts at once. Emit one entry in
  sub_cards per fact, each testing exactly one idea. The client deletes the
  original card and stores every sub-card in its place.
- mnemonic: the content was sound but abstract or unmemorable. Keep the
  question and answer, and add a short memory hook.

Constraints:
- Preserve the original factual meaning. Never introduce new claims.
- Echo each id back exactly as given.
- Return one result for every input id, even when you leave a card unchanged.
- sub_cards must be empty unless action is split; new_question/new_answer must
  be empty when action is split.
- Answers under 25 words.
- reason: one short sentence.

Cards:
{cards}
"""


def _clean_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _to_flashcard(item: dict) -> FlashcardOut | None:
    question = _clean_text(item.get("question"))
    answer = _clean_text(item.get("answer"))
    if not question or not answer:
        return None
    mnemonic = _clean_text(item.get("mnemonic")) or None
    return FlashcardOut(question=question, answer=answer, mnemonic=mnemonic)


def _build_result(item: dict) -> RemediatedCardOut | None:
    """Normalise one model result into the card list the client should store."""
    try:
        action = RemediationAction(item.get("action", "rewritten"))
    except ValueError:
        action = RemediationAction.REWRITTEN

    if action is RemediationAction.SPLIT:
        cards = [
            card
            for card in (_to_flashcard(sub) for sub in item.get("sub_cards") or [])
            if card is not None
        ]
        # A split with nothing usable in it is a model error, not a real split.
        if not cards:
            return None
        mnemonic = _clean_text(item.get("mnemonic")) or None
    else:
        card = _to_flashcard(
            {"question": item.get("new_question"), "answer": item.get("new_answer")}
        )
        if card is None:
            return None
        mnemonic = _clean_text(item.get("mnemonic")) or None
        if action is RemediationAction.MNEMONIC and mnemonic:
            card.mnemonic = mnemonic
        cards = [card]

    return RemediatedCardOut(
        id=item.get("id"),
        action=action.value,
        reason=(_clean_text(item.get("reason")) or "AI remediation")[:300],
        cards=cards,
        mnemonic=mnemonic,
    )


def remediate_leeches(
    cards: list[LeechCardIn],
) -> tuple[list[RemediatedCardOut], str | None]:
    """Return (repaired cards, serving provider label).

    Cards are batched so one request never exceeds the per-call cap. The
    provider label is None when every batch was served from cache.
    """
    if not cards:
        return [], None

    cache = ResponseCache()
    results: list[RemediatedCardOut] = []
    provider: str | None = None

    batch_size = max(1, settings.MAX_LEECHES_PER_CALL)
    for start in range(0, len(cards), batch_size):
        batch = cards[start : start + batch_size]
        payload = json.dumps(
            [
                {
                    "id": card.id,
                    "question": card.question,
                    "answer": card.answer,
                    "streak": card.streak,
                }
                for card in batch
            ],
            indent=2,
        )

        cache_key_payload = json.dumps(
            [c.model_dump() for c in batch], sort_keys=True
        )
        cached = cache.get("leech", cache_key_payload)
        if cached is not None:
            logger.info("Leech cache hit for batch of %d", len(batch))
            results.extend(RemediatedCardOut(**item) for item in cached)
            continue

        prompt = PROMPT.format(cards=payload)
        outcome = get_pool().generate_json(RemediationSchema, prompt)
        provider = outcome.label

        batch_results: list[dict] = []
        valid_ids = {card.id for card in batch}
        for item in outcome.data.get("remediated_cards", []):
            # Drop hallucinated ids rather than creating orphan cards.
            if item.get("id") not in valid_ids:
                continue

            out = _build_result(item)
            if out is None:
                continue

            results.append(out)
            batch_results.append(out.model_dump())

        if batch_results:
            cache.set("leech", cache_key_payload, batch_results)

    return results, provider