import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import (
    RemediateLeechesRequest,
    RemediateLeechesResponse,
)
from app.services.errors import to_http_response, upstream_error
from app.services.leech_agent import remediate_leeches
from app.services.providers.base import AllProvidersExhausted

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/leech", tags=["Leeches"])


@router.post("/remediate", response_model=RemediateLeechesResponse)
async def remediate(payload: RemediateLeechesRequest):
    if not payload.cards:
        raise HTTPException(status_code=400, detail="No leech cards provided.")

    if not settings.has_any_provider:
        return JSONResponse(
            status_code=503,
            content={
                "error": "no_provider_configured",
                "message": (
                    "Server has no AI provider configured. Set GEMINI_API_KEY or "
                    "GEMINI_CREDENTIALS in the deployment environment."
                ),
                "retry_after_seconds": None,
                "providers": None,
            },
        )

    try:
        remediated, provider = await remediate_leeches(payload.cards)
        return RemediateLeechesResponse(remediated_cards=remediated, provider=provider)
    except AllProvidersExhausted as exc:
        return to_http_response(exc)
    except Exception as exc:
        logger.exception("Leech remediation failed")
        return upstream_error(exc)