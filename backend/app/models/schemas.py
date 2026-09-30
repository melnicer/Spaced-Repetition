from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# --- Internal LLM response schemas (structured output targets) ---


class GeneratedCard(BaseModel):
    question: str = Field(description="A single, atomic question testing one fact")
    answer: str = Field(description="A concise answer, ideally under 25 words")
    mnemonic: Optional[str] = Field(
        default=None, description="Optional short memory hook, or null"
    )


class DeckSchema(BaseModel):
    title: str = Field(description="A short title for the deck")
    cards: List[GeneratedCard]


class RemediationAction(str, Enum):
    REWRITTEN = "rewritten"
    SPLIT = "split"
    MNEMONIC = "mnemonic"


class RemediatedCard(BaseModel):
    id: str = Field(description="Echo back the input card id exactly")
    action: RemediationAction = Field(
        description=(
            "rewritten if the wording was confusing; "
            "split if the card tested multiple facts; "
            "mnemonic if the content was sound but hard to memorise"
        )
    )
    new_question: str = Field(
        default="",
        description="The clarified question. Used when action is rewritten or mnemonic.",
    )
    new_answer: str = Field(
        default="",
        description="The clarified answer. Used when action is rewritten or mnemonic.",
    )
    sub_cards: List[GeneratedCard] = Field(
        default_factory=list,
        description=(
            "For action=split: the bite-sized cards that replace the original. "
            "Each must test exactly one fact. Leave empty for other actions."
        ),
    )
    mnemonic: Optional[str] = Field(default=None, description="Memory aid, or null")
    reason: str = Field(description="One short sentence explaining the change")


class RemediationSchema(BaseModel):
    remediated_cards: List[RemediatedCard]


# --- Public API models ---


class FlashcardOut(BaseModel):
    question: str
    answer: str
    mnemonic: Optional[str] = None


class DeckOut(BaseModel):
    title: str
    cards: List[FlashcardOut]
    cached: bool = Field(
        default=False, description="True when served from cache without an API call"
    )
    provider: Optional[str] = Field(
        default=None, description="alias/model that generated this deck"
    )


class LeechCardIn(BaseModel):
    id: str
    question: str
    answer: str
    streak: int = 3


class RemediateLeechesRequest(BaseModel):
    cards: List[LeechCardIn]


class RemediatedCardOut(BaseModel):
    id: str
    action: str
    reason: str
    # Exactly what the client should store in place of the original card:
    # one entry for rewritten/mnemonic, N entries for split.
    cards: List[FlashcardOut]
    mnemonic: Optional[str] = None


class RemediateLeechesResponse(BaseModel):
    remediated_cards: List[RemediatedCardOut]
    provider: Optional[str] = None


class ApiError(BaseModel):
    """Structured error body so the app never has to parse a traceback."""

    error: str
    message: str
    retry_after_seconds: Optional[int] = None
    providers: Optional[List[dict]] = None


class ProviderStatusOut(BaseModel):
    alias: str
    model: str
    vendor: str
    fingerprint: str
    available: bool
    disabled: bool
    cooldown_remaining: int
    calls: int
    failures: int
    blacklisted_models: List[str]
    last_error: str
    last_error_kind: str
    last_error_at: float


class StatusOut(BaseModel):
    configured: bool
    providers: List[ProviderStatusOut]
    cache_hits: int
    cache_misses: int