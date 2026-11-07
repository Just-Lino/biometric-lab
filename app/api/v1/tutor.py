import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.services.events import record_event
from app.services.tutor import MAX_CONTEXT_CHARS, TutorRateLimited, answer

router = APIRouter(prefix="/api/v1/tutor", tags=["tutor"])


class AskIn(BaseModel):
    mode: Literal["hint", "explain", "diagnose"] = "hint"
    question: str = Field("", max_length=2000)  # длинный вопрос уходил бы в платный запрос к модели целиком
    context: dict = Field(default_factory=dict)

    @field_validator("context")
    @classmethod
    def context_not_huge(cls, v: dict) -> dict:
        if len(json.dumps(v, default=str)) > MAX_CONTEXT_CHARS:
            raise ValueError("Контекст эксперимента слишком большой")
        return v


@router.post("/ask")
def ask(body: AskIn):
    try:
        provider, text = answer(body.mode, body.question, body.context)
    except TutorRateLimited as e:
        record_event("tutor_limited", {"mode": body.mode})
        raise HTTPException(status_code=429, detail=str(e), headers={"Retry-After": str(e.retry_after)})
    record_event("tutor", {"mode": body.mode, "question": body.question})
    return {"mode": body.mode, "provider": provider, "answer": text}
