from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.events import record_event
from app.training.trainer import TrainConfig, trainer

router = APIRouter(prefix="/api/v1/training", tags=["training"])


class TrainIn(BaseModel):
    modality: str = Field("fingerprint", pattern="^(fingerprint|face)$")
    epochs: int = Field(30, ge=1, le=80)
    lr: float = Field(0.003, gt=0, le=0.05)
    batch: int = Field(32, ge=4, le=128)
    aug: float = Field(1.0, ge=0, le=1.5)
    copies: int = Field(8, ge=1, le=20)
    n_subjects: int = Field(12, ge=3, le=12)


@router.post("/start")
def start(body: TrainIn):
    try:
        trainer.start(TrainConfig(**body.model_dump()))
    except ValueError as e:
        raise HTTPException(409, str(e))
    record_event("train_start", body.model_dump())
    return trainer.status()


@router.post("/stop")
def stop():
    trainer.stop()
    return {"ok": True}


@router.get("/status")
def status():
    return trainer.status()


@router.get("/snapshot/{epoch}")
def snapshot(epoch: int):
    s = trainer.snapshot(epoch)
    if s is None:
        raise HTTPException(404, "Снимок этой эпохи ещё не готов")
    return s
