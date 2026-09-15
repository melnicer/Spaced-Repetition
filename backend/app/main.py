from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, deck, leech

app = FastAPI(
    title="FlutterStudy API",
    version="1.0.0",
    description="Backend AI & Ingestion Agent for FlutterStudy offline-first flashcard app."
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
