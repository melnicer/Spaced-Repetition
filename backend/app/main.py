import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import deck, health, leech

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.warn_if_unconfigured()
    credentials = settings.effective_gemini_credentials
    logger.info(
        "Configured %d Gemini credential(s) x %d model(s); Groq %s",
        len(credentials),
        len(settings.GEMINI_MODEL_CHAIN),
        "enabled" if settings.GROQ_API_KEY else "disabled",
    )
    yield


app = FastAPI(
    title="FlutterStudy API",
    version="2.0.0",
    description=(
        "Backend AI agent for FlutterStudy. Supports multi-provider failover "
        "across Gemini (per-project credentials) and optional Groq fallback."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(deck.router)
app.include_router(leech.router)