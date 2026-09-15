import json
import google.generativeai as genai
from app.config import settings
from app.models.schemas import LeechCardIn, RemediatedCardOut

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

def get_gemini_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        flash_models = [m for m in models if 'flash' in m.lower()]
        if flash_models:
            return genai.GenerativeModel(flash_models[0])
        elif models:
            return genai.GenerativeModel(models[0])
    except Exception:
        pass
    return genai.GenerativeModel("gemini-1.5-flash")

def remediate_leeches(cards: list[LeechCardIn]) -> list[RemediatedCardOut]:
    if not settings.GEMINI_API_KEY:
        # Fallback dummy remediation if API key not set
        results = []
        for c in cards:
            results.append(RemediatedCardOut(
                id=c.id,
                action="rewritten",
                new_question=f"Simplified: {c.question}",
                new_answer=f"Simplified: {c.answer}",
                mnemonic="Fallback mnemonic: Think of simpler terms.",
                reason="Fallback mock remediation applied because GEMINI_API_KEY is not configured."
            ))
        return results

    model = get_gemini_model()
    
    cards_payload = [{"id": c.id, "question": c.question, "answer": c.answer, "streak": c.streak} for c in cards]
    
    prompt = f"""
You are an expert Spaced Repetition AI agent. The user has repeatedly failed (failure streak >= 3) on the following flashcards (leeches).
For each card, analyze why it might be difficult (too dense, multi-concept, or abstract) and apply one of three remediation strategies:
1. "rewritten": Rewrite the wording to be simpler and clearer.
2. "split": Split a multi-concept question into simpler components (provide the first sub-card or simplified question/answer).
3. "mnemonic": Add a powerful mnemonic memory hint.

Return ONLY valid JSON with this exact structure:
{{
  "remediated_cards": [
    {{
      "id": "card_id_here",
      "action": "rewritten" | "split" | "mnemonic",
      "new_question": "...",
      "new_answer": "...",
      "mnemonic": "...",
      "reason": "..."
    }}
  ]
}}

Leech Cards:
{json.dumps(cards_payload, indent=2)}
"""

    response = model.generate_content(prompt)
    text_resp = response.text.strip()
    if text_resp.startswith("```json"):
        text_resp = text_resp[7:]
    if text_resp.endswith("```"):
        text_resp = text_resp[:-3]
    text_resp = text_resp.strip()

    try:
        data = json.loads(text_resp)
        out = []
        for item in data.get("remediated_cards", []):
            out.append(RemediatedCardOut(
                id=item["id"],
                action=item.get("action", "rewritten"),
                new_question=item["new_question"],
                new_answer=item["new_answer"],
                mnemonic=item.get("mnemonic"),
                reason=item.get("reason", "AI agent remediation")
            ))
        return out
    except Exception as e:
        raise ValueError(f"Failed to parse Gemini JSON response for leech remediation: {e}\nResponse was: {text_resp}")
