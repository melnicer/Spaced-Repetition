from fastapi import APIRouter

from app.config import settings
from app.models.schemas import StatusOut
from app.services.cache import ResponseCache
from app.services.providers.pool import get_pool

router = APIRouter(tags=["Status"])


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "FlutterStudy API"}


@router.get("/api/status", response_model=StatusOut)
def status():
    """Provider pool health for debugging from the phone.

    Exposes only the last 4 characters of each key, never the full value.
    """
    pool = get_pool()
    cache_stats = ResponseCache.stats()
    return StatusOut(
        configured=settings.has_any_provider,
        providers=pool.status(),
        cache_hits=cache_stats["cache_hits"],
        cache_misses=cache_stats["cache_misses"],
    )