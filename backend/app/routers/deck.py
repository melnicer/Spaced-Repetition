import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import DeckOut
from app.services.deck_generator import generate_flashcard_deck
from app.services.errors import to_http_response, upstream_error
from app.services.parser import parse_pdf, parse_text_file
from app.services.providers.base import AllProvidersExhausted

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/deck", tags=["Decks"])

ALLOWED_SUFFIXES = (".pdf", ".txt", ".md")


@router.post("/generate", response_model=DeckOut)
async def generate_deck(
    title: str = Form("Generated Deck"),
    text: str = Form(None),
    file: UploadFile = File(None),
):
    raw_content = ""

    if text:
        raw_content += text + "\n"

    if file is not None and file.filename:
        filename = file.filename.lower()
        if not filename.endswith(ALLOWED_SUFFIXES):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use .pdf, .md, or .txt.",
            )
        file_bytes = await file.read()
        if file_bytes:
            try:
                raw_content += (
                    parse_pdf(file_bytes)
                    if filename.endswith(".pdf")
                    else parse_text_file(file_bytes)
                ) + "\n"
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not raw_content.strip():
        raise HTTPException(
            status_code=400,
            detail="No content provided. Paste text or upload a document.",
        )

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
        return generate_flashcard_deck(raw_content, title)
    except AllProvidersExhausted as exc:
        return to_http_response(exc)
    except ValueError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": "invalid_model_output",
                "message": str(exc)[:500],
                "retry_after_seconds": None,
                "providers": None,
            },
        )
    except Exception as exc:
        logger.exception("Deck generation failed")
        return upstream_error(exc)