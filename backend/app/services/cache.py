import json
import os
import time

from app.config import settings
from app.services.text_utils import cache_key

_HITS = 0
_MISSES = 0


class ResponseCache:
    """Content-addressed disk cache.

    Deck generation is deterministic for a given source document, so an
    identical upload should never cost another API call.
    """

    def __init__(self, directory: str | None = None, enabled: bool | None = None):
        self._dir = directory if directory is not None else settings.CACHE_DIR
        self._enabled = settings.CACHE_ENABLED if enabled is None else enabled
        if self._enabled:
            try:
                os.makedirs(self._dir, exist_ok=True)
            except OSError:
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.json")

    def get(self, kind: str, payload: str) -> dict | None:
        global _HITS, _MISSES
        if not self._enabled:
            return None
        path = self._path(cache_key(kind, payload, settings.PROMPT_VERSION))
        if not os.path.exists(path):
            _MISSES += 1
            return None
        try:
            # Guard against truncated writes from an interrupted request.
            if time.time() - os.path.getmtime(path) > 30 * 24 * 3600:
                os.remove(path)
                return None
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            _HITS += 1
            return data
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, kind: str, payload: str, value: dict) -> None:
        if not self._enabled:
            return
        path = self._path(cache_key(kind, payload, settings.PROMPT_VERSION))
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(value, handle)
            os.replace(tmp, path)
        except OSError:
            pass

    @staticmethod
    def stats() -> dict:
        return {"cache_hits": _HITS, "cache_misses": _MISSES}