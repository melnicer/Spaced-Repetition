"""Google Gemini provider built on the current `google-genai` SDK."""

import json

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import settings
from app.services.providers.base import (
    AuthFailed,
    ModelNotFound,
    PermanentError,
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

    # A real status code is authoritative. The substring fallbacks below are
    # for SDK exceptions that omit it, and a bare "400" inside a 503 message
    # would otherwise misclassify the response.
    if status is not None:
        if status == 429:
            return QuotaExceeded(text, retry_after=_retry_after(exc))
        if status == 404:
            return ModelNotFound(text)
        if status in (401, 403):
            return AuthFailed(text)
        if 500 <= status < 600:
            return TransientError(text)
        if 400 <= status < 500:
            return PermanentError(text)

    if "resource_exhausted" in lowered or "rate limit" in lowered or "429" in lowered:
        return QuotaExceeded(text, retry_after=_retry_after(exc))
    if "not found" in lowered or "404" in lowered or "is not supported" in lowered:
        return ModelNotFound(text)
    if "permission" in lowered or "api key" in lowered or "unauthenticated" in lowered:
        return AuthFailed(text)
    # Checked before the generic fallbacks: a safety block or bad argument is
    # a permanent rejection, and retrying it only delays the failover.
    if "safety" in lowered or "blocked" in lowered or "prohibited_content" in lowered:
        return PermanentError(text)
    if "invalid_argument" in lowered or "invalid argument" in lowered:
        return PermanentError(text)
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
            self.status.note_error("invalid_json", f"Invalid JSON from {self.model}: {exc}")
            raise ProviderError(f"Invalid JSON from {self.model}: {exc}") from exc
        except Exception as exc:
            error = _translate(exc)
            self.status.note_error(type(error).__name__, str(error))
            raise error from exc