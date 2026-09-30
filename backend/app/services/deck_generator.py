"""Deck generation via the provider pool, with disk caching."""

import logging

from app.config import settings
from app.models.schemas import DeckSchema, GeneratedCard
from app.services.cache import ResponseCache
from app.services.providers.pool import get_pool
from app.services.text_utils import normalize_text, truncate_chars

logger = logging.getLogger(__name__)

PROMPT = """You are an expert study assistant. Convert the notes below into a
high-yield flashcard deck.

Rules:
- One fact per card. Split anything testing multiple ideas.
- Questions must be answerable from the answer alone.
- Answers under 25 words, plain prose, no markdown.
- Emit at most {max_cards} cards, prioritising the highest-yield concepts.
- Use the deck title provided by the caller: "{title}"
- Set mnemonic only when a short memory hook genuinely helps; otherwise null.

Notes:
{body}
"""


async def generate_flashcard_deck(raw_text: str, title: str) -> dict:
    """Return {"title", "cards", "cached", "provider"}.

    An identical source always hits the cache, so regenerating the same notes
    costs zero API calls.
    """
    normalized = normalize_text(raw_text)
    if not normalized:
        raise ValueError("No content provided.")

    cache = ResponseCache()
    cached = cache.get("deck", normalized)
    if cached is not None:
        logger.info("Deck cache hit")
        cached["cached"] = True
        return cached

    body = truncate_chars(normalized, settings.MAX_INPUT_CHARS)
    prompt = PROMPT.format(max_cards=settings.MAX_CARDS, title=title, body=body)

    pool = get_pool()
    outcome = await pool.generate_json(DeckSchema, prompt)

    cards: list[GeneratedCard] = []
    for item in outcome.data.get("cards", [])[: settings.MAX_CARDS]:
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        if not question or not answer:
            continue
        mnemonic = item.get("mnemonic")
        cards.append(
            GeneratedCard(
                question=question,
                answer=answer,
                mnemonic=(mnemonic.strip() if isinstance(mnemonic, str) and mnemonic.strip() else None),
            )
        )

    if not cards:
        raise ValueError("The model returned no usable cards for this content.")

    result = {
        "title": (outcome.data.get("title") or title).strip(),
        "cards": [card.model_dump() for card in cards],
        "cached": False,
        "provider": outcome.label,
    }

    cache.set("deck", normalized, result)
    return result