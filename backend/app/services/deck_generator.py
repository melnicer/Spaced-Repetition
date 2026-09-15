import json
import google.generativeai as genai
from app.config import settings

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

def generate_flashcard_deck(raw_text: str, title: str) -> dict:
    if not settings.GEMINI_API_KEY:
        # Fallback dummy deck if API key is not set (useful for local dev/testing)
        return {
            "title": title,
            "cards": [
                {
                    "question": f"Sample Question 1 from {title}",
                    "answer": f"Sample Answer derived from text: {raw_text[:50]}...",
                    "mnemonic": "Remember the first 50 chars!"
                },
                {
                    "question": f"Sample Question 2 from {title}",
                    "answer": "Sample Answer 2",
                    "mnemonic": None
                }
            ]
        }

    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    prompt = f"""
You are an expert AI study assistant. Convert the following raw study notes or document text into a concise, high-yield flashcard deck.
Return ONLY valid JSON matching this exact structure:
{{
  "title": "{title}",
  "cards": [
    {{
      "question": "Clear, atomic question",
      "answer": "Concise answer",
      "mnemonic": "Optional memory hook or null"
    }}
  ]
}}

Raw Text:
{raw_text}
"""
    response = model.generate_content(prompt)
    text_resp = response.text.strip()
    
    # Clean markdown code blocks if present
    if text_resp.startswith("```json"):
        text_resp = text_resp[7:]
    if text_resp.endswith("```"):
        text_resp = text_resp[:-3]
    text_resp = text_resp.strip()

    try:
        data = json.loads(text_resp)
        return data
    except Exception as e:
        raise ValueError(f"Failed to parse Gemini JSON response for deck generation: {e}\nResponse was: {text_resp}")
