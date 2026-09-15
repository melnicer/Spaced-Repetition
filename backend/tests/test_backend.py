from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "FlutterStudy API"}

def test_generate_deck_text():
    response = client.post(
        "/api/deck/generate",
        data={"title": "Test Deck", "text": "Photosynthesis is the process used by plants to convert light into energy."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Deck"
    assert len(data["cards"]) > 0
    assert "question" in data["cards"][0]
    assert "answer" in data["cards"][0]

def test_remediate_leeches():
    payload = {
        "cards": [
            {
                "id": "c1",
                "question": "What is the mitochondrial respiratory chain?",
                "answer": "Complex series of electron carriers...",
                "streak": 3
            }
        ]
    }
    response = client.post("/api/leech/remediate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["remediated_cards"]) == 1
    assert data["remediated_cards"][0]["id"] == "c1"
    assert "action" in data["remediated_cards"][0]
