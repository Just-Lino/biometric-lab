from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.events import record_event
from app.services import training_runs
from app.training import registry
from app.training.datasets import available_datasets, default_dataset
from app.training.trainer import TrainConfig, trainer

router = APIRouter(prefix="/api/v1/training", tags=["training"])


class TrainIn(BaseModel):
    modality: str = Field("fingerprint", pattern="^(fingerprint|face|iris)$")
    epochs: int = Field(30, ge=1, le=80)
    lr: float = Field(0.003, gt=0, le=0.1)
    batch: int = Field(32, ge=4, le=128)
    aug: float = Field(1.0, ge=0, le=1.5)
    copies: int = Field(8, ge=1, le=20)
    n_subjects: int = Field(12, ge=3, le=40)
    arch: str = Field("small", pattern="^(small|medium|large)$")
    optimizer: str = Field("adam", pattern="^(adam|sgd)$")
    holdout: int = Field(3, ge=1, le=3)
    dataset: str | None = Field(None, max_length=40)  # набор данных; не указан - по умолчанию для биометрии


@router.post("/start")
def start(body: TrainIn):
    cfg = body.model_dump()
    cfg["dataset"] = cfg["dataset"] or default_dataset(body.modality)
    allowed = {d["id"] for d in available_datasets()[body.modality]}
    if cfg["dataset"] not in allowed:
        raise HTTPException(400, f"Набор данных «{cfg['dataset']}» недоступен для этой биометрии. Доступны: {', '.join(sorted(allowed))}")
    try:
        trainer.start(TrainConfig(**cfg))
    except ValueError as e:
        raise HTTPException(409, str(e))
    record_event("train_start", body.model_dump())
    return trainer.status()


@router.post("/stop")
def stop():
    trainer.stop()
    return {"ok": True}


@router.get("/status")
def status(light: bool = False):
    return trainer.status(light)


@router.get("/snapshot/{epoch}")
def snapshot(epoch: int):
    s = trainer.snapshot(epoch)
    if s is None:
        raise HTTPException(404, "Снимок этой эпохи ещё не готов")
    return s


@router.get("/datasets")
def datasets():
    """Наборы данных, на которых можно обучать сеть (только найденные на диске)."""
    return {"datasets": available_datasets(), "default": {m: default_dataset(m) for m in ("fingerprint", "face", "iris")}}


@router.get("/runs")
def runs():
    return training_runs.list_runs()


@router.delete("/runs")
def clear_runs():
    training_runs.clear()
    return {"ok": True}


@router.get("/model/{modality}")
def model_info(modality: str):
    """Есть ли обученная сеть для этих данных и какой порог для неё рекомендован."""
    if modality not in ("fingerprint", "face", "iris"):
        raise HTTPException(404, "Неизвестная биометрия")
    m = registry.get(modality)
    return m.info() if m else {"available": False}
