from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.events import record_event
from app.services.tutor import get_tutor

router = APIRouter(prefix="/api/v1/tutor", tags=["tutor"])


class AskIn(BaseModel):
    mode: Literal["hint", "explain", "diagnose"] = "hint"
    question: str = ""
    context: dict = Field(default_factory=dict)


@router.post("/ask")
def ask(body: AskIn):
    tutor = get_tutor()
    text = tutor.ask(body.mode, body.question, body.context)
    record_event("tutor", {"mode": body.mode, "question": body.question})
    return {"mode": body.mode, "provider": tutor.name, "answer": text}
