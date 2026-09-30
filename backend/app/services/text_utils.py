import re
import unicodedata
from hashlib import sha256


def normalize_text(text: str) -> str:
    """Collapse whitespace and unify unicode so equivalent inputs hash identically."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_chars(text: str, max_chars: int) -> str:
    """Hard cap on prompt size. Approximates tokens at ~4 chars each."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def cache_key(kind: str, payload: str, prompt_version: str) -> str:
    digest = sha256(f"{prompt_version}|{kind}|{normalize_text(payload)}".encode("utf-8"))
    return f"{kind}-{digest.hexdigest()}"