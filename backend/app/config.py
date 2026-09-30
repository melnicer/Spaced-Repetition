import os

from dotenv import load_dotenv

load_dotenv()


def _split(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class Credential:
    """A single (alias, api_key) pair.

    Gemini rate limits are applied per GCP *project*, not per API key, so each
    credential should come from a separate project for rotation to be useful.
    """

    def __init__(self, alias: str, api_key: str):
        self.alias = alias
        self.api_key = api_key

    @property
    def fingerprint(self) -> str:
        """Last 4 characters only, safe to expose via /api/status."""
        return self.api_key[-4:] if len(self.api_key) >= 4 else "****"

    def __repr__(self) -> str:
        return f"Credential(alias={self.alias!r}, fingerprint={self.fingerprint!r})"


class Settings:
    # --- Credentials ---
    # Legacy single-key variable, still honoured so existing deployments work.
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Plural form: "alias:key,alias:key" for multi-project rotation.
    GEMINI_CREDENTIALS: list[Credential] = [
        Credential(*entry.split(":", 1)) if ":" in entry else Credential(f"gemini{index + 1}", entry)
        for index, entry in enumerate(_split(os.getenv("GEMINI_CREDENTIALS")))
    ]

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # --- Status endpoint access ---
    # When set, /api/status requires "X-Status-Token" and stops publishing
    # credential fingerprints. When unset it still works, but only ever
    # returns redacted diagnostics, so an unconfigured deploy is safe.
    STATUS_TOKEN: str = os.getenv("STATUS_TOKEN", "")

    # --- Models (env-driven so deprecations never require a code push) ---
    GEMINI_MODEL_CHAIN: list[str] = _split(os.getenv("GEMINI_MODEL_CHAIN")) or [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]

    GROQ_MODEL_CHAIN: list[str] = _split(os.getenv("GROQ_MODEL_CHAIN")) or [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ]

    # How long to cool a model after it burns all its transient retries.
    # 0 disables cooldown and makes the pool fail over on every request.
    TRANSIENT_COOLDOWN_SECONDS: int = int(os.getenv("TRANSIENT_COOLDOWN_SECONDS", "30"))

    # --- Limits ---
    MAX_INPUT_CHARS: int = int(os.getenv("MAX_INPUT_CHARS", "24000"))
    MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "60"))
    MAX_CARDS: int = int(os.getenv("MAX_CARDS", "20"))
    MAX_LEECHES_PER_CALL: int = int(os.getenv("MAX_LEECHES_PER_CALL", "8"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

    # Groq's free tier caps at 8K TPM, so oversized prompts must skip it
    # rather than repeatedly 429.
    GROQ_MAX_INPUT_CHARS: int = int(os.getenv("GROQ_MAX_INPUT_CHARS", "12000"))

    # --- Cache ---
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/flutterstudy_cache")
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}

    # Bump when prompt/schema semantics change so stale entries are ignored.
    PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v2")

    @property
    def effective_gemini_credentials(self) -> list[Credential]:
        """GEMINI_CREDENTIALS when set, otherwise the legacy single key."""
        if self.GEMINI_CREDENTIALS:
            return self.GEMINI_CREDENTIALS
        if self.GEMINI_API_KEY:
            return [Credential("gemini-default", self.GEMINI_API_KEY)]
        return []

    @property
    def has_any_provider(self) -> bool:
        return bool(self.effective_gemini_credentials or self.GROQ_API_KEY)

    def warn_if_unconfigured(self) -> None:
        if not self.has_any_provider:
            print(
                "[WARNING] No API credentials configured. Set GEMINI_API_KEY or "
                "GEMINI_CREDENTIALS (and optionally GROQ_API_KEY) before deploying."
            )


settings = Settings()