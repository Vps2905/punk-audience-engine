from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.chat_orchestrator import handle_chat


router = APIRouter(tags=["Module 5 - Conversational Orchestration"])


class ChatRequest(BaseModel):
    message: str = Field(..., description="User command")
    job_id: Optional[str] = Field(default=None, description="Required for creating cohort")
    cohort_name: Optional[str] = Field(default=None)
    top_k: int = Field(default=20, ge=1, le=100)
    seed_limit: int = Field(default=1000, ge=1, le=100000)


@router.post("/chat")
def chat(request: ChatRequest):
    """
    Conversational endpoint that triggers audience intelligence workflows.
    """
    return handle_chat(
        message=request.message,
        job_id=request.job_id,
        cohort_name=request.cohort_name,
        top_k=request.top_k,
        seed_limit=request.seed_limit
    )
