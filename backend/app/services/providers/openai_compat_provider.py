"""Single adapter for every OpenAI-compatible endpoint.

Covers Groq (GroqCloud) today and Grok/xAI with a different base_url, since
both speak the same wire format.
"""

import json

from openai import OpenAI
from pydantic import BaseModel

from app.services.providers.base import (
    AuthFailed,
    ModelNotFound,
    Provider,
    ProviderError,
    QuotaExceeded,
    TransientError,
)


class OpenAICompatProvider(Provider):
    vendor = "openai_compat"

    def __init__(
        self,
        alias: str,
        model: str,
        api_key: str,
        base_url: str,
        max_input_chars: int,
        fingerprint: str = "",
        vendor_label: str | None = None,
    ):
        super().__init__(alias, model, api_key, fingerprint)
        if vendor_label:
            self.vendor = vendor_label
        self.status.vendor = self.vendor
        self._base_url = base_url
        self._max_input_chars = max_input_chars
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    @property
    def max_input_chars(self) -> int:
        return self._max_input_chars

    def generate_json(self, schema: type[BaseModel], prompt: str, max_output_tokens: int) -> dict:
        self.status.calls += 1
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=max_output_tokens,
            )
            content = completion.choices[0].message.content or ""
            text = content.strip()
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
            raise self._translate(exc) from exc

    @staticmethod
    def _translate(exc: Exception) -> ProviderError:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        text = str(exc)
        lowered = text.lower()

        if status == 429 or "rate limit" in lowered or "429" in lowered:
            retry_after = None
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
            if headers is not None:
                raw = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
                if raw:
                    try:
                        retry_after = int(float(raw))
                    except (TypeError, ValueError):
                        retry_after = None
            return QuotaExceeded(text, retry_after=retry_after)
        if status == 404 or "model_not_found" in lowered or "404" in lowered:
            return ModelNotFound(text)
        if status in (401, 403) or "unauthorized" in lowered or "invalid api key" in lowered:
            return AuthFailed(text)
        return TransientError(text)