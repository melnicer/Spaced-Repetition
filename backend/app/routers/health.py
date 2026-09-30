import hmac
import logging

from fastapi import APIRouter, Header

from app.config import settings
from app.models.schemas import StatusOut
from app.services.cache import ResponseCache
from app.services.providers.pool import get_pool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Status"])


def _is_authorised(token: str | None) -> bool:
    """Constant-time compare so the token cannot be recovered by timing."""
    if not settings.STATUS_TOKEN:
        # No token configured: stay usable for local debugging, but serve
        # redacted diagnostics rather than refusing outright.
        return False
    if not token:
        return False
    return hmac.compare_digest(token.strip(), settings.STATUS_TOKEN)


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "FlutterStudy API"}


@router.get("/api/status", response_model=StatusOut)
def status(x_status_token: str | None = Header(default=None)):
    """Provider pool health for debugging from the phone.

    Credential fingerprints and raw upstream error text are only included
    when X-Status-Token matches STATUS_TOKEN. An anonymous caller still gets
    availability and counters, which is enough to tell a bad URL from a
    misconfigured provider without exposing anything sensitive.
    """
    detailed = _is_authorised(x_status_token)
    if x_status_token and not detailed:
        logger.warning("Rejected /api/status request with an invalid token")

    pool = get_pool()
    cache_stats = ResponseCache.stats()
    return StatusOut(
        configured=settings.has_any_provider,
        providers=pool.status(detailed=detailed),
        cache_hits=cache_stats["cache_hits"],
        cache_misses=cache_stats["cache_misses"],
        detailed=detailed,
    )