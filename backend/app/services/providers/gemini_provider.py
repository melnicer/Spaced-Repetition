"""Google Gemini provider built on the current `google-genai` SDK."""

import json

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import settings
from app.services.providers.base import (
    AuthFailed,
    ModelNotFound,
    Provider,
    ProviderError,
    QuotaExceeded,
    TransientError,
)


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    return None


def _retry_after(exc: Exception) -> int | None:
    for attr in ("retry_after", "retry_delay"):
        value = getattr(exc, attr, None)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _translate(exc: Exception) -> ProviderError:
    """Map SDK/transport exceptions onto our classified hierarchy."""
    status = _status_code(exc)
    text = str(exc)
    lowered = text.lower()

    if status == 429 or "resource_exhausted" in lowered or "429" in lowered:
        return QuotaExceeded(text, retry_after=_retry_after(exc))
    if status == 404 or "not found" in lowered or "404" in lowered:
        return ModelNotFound(text)
    if status in (401, 403) or "permission" in lowered or "api key" in lowered:
        return AuthFailed(text)
    if status is not None and 500 <= status < 600:
        return TransientError(text)
    if "timeout" in lowered or "deadline" in lowered or "connection" in lowered:
        return TransientError(text)
    return TransientError(text)


class GeminiProvider(Provider):
    vendor = "gemini"

    def __init__(self, alias: str, model: str, api_key: str, fingerprint: str = ""):
        super().__init__(alias, model, api_key, fingerprint)
        self._client = genai.Client(api_key=api_key)

    @property
    def max_input_chars(self) -> int:
        return settings.MAX_INPUT_CHARS

    def generate_json(self, schema: type[BaseModel], prompt: str, max_output_tokens: int) -> dict:
        self.status.calls += 1
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.3,
                    max_output_tokens=max_output_tokens,
                ),
            )
            text = (response.text or "").strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except json.JSONDecodeError as exc:
            self.status.failures += 1
            raise ProviderError(f"Invalid JSON from {self.model}: {exc}") from exc
        except Exception as exc:
            self.status.failures += 1
            raise _translate(exc) from exc