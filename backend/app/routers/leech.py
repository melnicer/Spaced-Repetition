from fastapi import APIRouter, HTTPException
from app.models.schemas import RemediateLeechesRequest, RemediateLeechesResponse
from app.services.leech_agent import remediate_leeches

router = APIRouter(prefix="/api/leech", tags=["Leeches"])

@router.post("/remediate", response_model=RemediateLeechesResponse)
def remediate(payload: RemediateLeechesRequest):
    if not payload.cards:
        raise HTTPException(status_code=400, detail="No leech cards provided.")
    
    try:
        remediated = remediate_leeches(payload.cards)
        return RemediateLeechesResponse(remediated_cards=remediated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
