from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.schemas import DeckOut
from app.services.parser import parse_pdf, parse_text_file
from app.services.deck_generator import generate_flashcard_deck

router = APIRouter(prefix="/api/deck", tags=["Decks"])

@router.post("/generate", response_model=DeckOut)
async def generate_deck(
    title: str = Form("Generated Deck"),
    text: str = Form(None),
    file: UploadFile = File(None)
):
    raw_content = ""
    
    if text:
        raw_content += text + "\n"
        
    if file:
        file_bytes = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            raw_content += parse_pdf(file_bytes) + "\n"
        elif filename.endswith((".txt", ".md")):
            raw_content += parse_text_file(file_bytes) + "\n"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .pdf, .md, or .txt")

    if not raw_content.strip():
        raise HTTPException(status_code=400, detail="No content provided. Please supply text or upload a document.")

    try:
        deck_data = generate_flashcard_deck(raw_content, title)
        return deck_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
