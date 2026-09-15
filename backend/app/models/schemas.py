from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateDeckRequest(BaseModel):
    title: Optional[str] = "Generated Deck"
    text: Optional[str] = None

class FlashcardOut(BaseModel):
    question: str
    answer: str
    mnemonic: Optional[str] = None

class DeckOut(BaseModel):
    title: str
    cards: List[FlashcardOut]

class LeechCardIn(BaseModel):
    id: str
    question: str
    answer: str
    streak: int = 3

class RemediateLeechesRequest(BaseModel):
    cards: List[LeechCardIn]

class RemediatedCardOut(BaseModel):
    id: str
    action: str  # "rewritten", "split", "mnemonic"
    new_question: str
    new_answer: str
    mnemonic: Optional[str] = None
    reason: str

class RemediateLeechesResponse(BaseModel):
    remediated_cards: List[RemediatedCardOut]
