"""Shared helpers for surfacing provider failures to HTTP clients."""

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.services.providers.base import AllProvidersExhausted


class InvalidInput(Exception):
    """Client-side problem; maps to 400."""


def to_http_response(exc: AllProvidersExhausted) -> JSONResponse:
    """429 with Retry-After so the app can show a real countdown."""
    retry_after = exc.earliest_retry
    body = {
        "error": "all_providers_exhausted",
        "message": (
            "All AI providers are unavailable right now. "
            + (
                f"Try again in {retry_after}s."
                if retry_after
                else "Check your API keys and try again."
            )
        ),
        "retry_after_seconds": retry_after,
        "providers": exc.attempts,
    }
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    return JSONResponse(status_code=429, content=body, headers=headers)


def upstream_error(exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "error": "upstream_error",
            "message": str(exc)[:500],
            "retry_after_seconds": None,
            "providers": None,
        },
    )


def invalid_input(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)