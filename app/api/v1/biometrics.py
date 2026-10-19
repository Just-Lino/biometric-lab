from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError

from app.modules.registry import get_module, list_modules
from app.services import experiments
from app.services.events import record_event

router = APIRouter(prefix="/api/v1/biometrics", tags=["biometrics"])


class ProcessIn(BaseModel):
    sample_id: str
    params: dict = Field(default_factory=dict)


class CompareIn(BaseModel):
    sample_a: str
    sample_b: str
    params: dict = Field(default_factory=dict)
    threshold: float = Field(0.5, ge=0, le=1)


class EvaluateIn(BaseModel):
    params: dict = Field(default_factory=dict)
    n_subjects: int = Field(8, ge=3, le=12)


def _mod(name: str):
    try:
        return get_module(name)
    except KeyError:
        raise HTTPException(404, f"Модуль «{name}» не найден")


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False, include_context=False))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("")
def modalities():
    return list_modules()


@router.get("/{modality}/info")
def info(modality: str):
    return _mod(modality).info()


@router.get("/{modality}/samples")
def samples(modality: str):
    return _mod(modality).list_samples()


@router.get("/{modality}/sample/{sample_id}.png")
def sample_png(modality: str, sample_id: str, extra_noise: float = 0):
    data = _guard(_mod(modality).get_sample_preview, sample_id, {"extra_noise": extra_noise})
    return Response(data, media_type="image/png")


@router.post("/{modality}/upload")
async def upload(modality: str, file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 5_000_000:
        raise HTTPException(413, "Файл больше 5 МБ")
    sid = _guard(_mod(modality).upload, data)
    record_event("upload", {"modality": modality, "sample_id": sid})
    return {"sample_id": sid}


@router.post("/{modality}/process")
def process(modality: str, body: ProcessIn):
    res = _guard(_mod(modality).process, body.sample_id, body.params)
    record_event("process", {"modality": modality, "sample": body.sample_id, "params": body.params})
    return res


@router.post("/{modality}/compare")
def compare(modality: str, body: CompareIn):
    res = _guard(_mod(modality).compare, body.sample_a, body.sample_b, body.params)
    decision = "MATCH" if res["similarity"] >= body.threshold else "NO MATCH"
    res.update(threshold=body.threshold, decision=decision)
    experiments.add_experiment(modality, body.sample_a, body.sample_b, body.params, res["similarity"],
                               body.threshold, decision, res.get("n_a", 0), res.get("n_b", 0), res.get("matched", 0))
    record_event("compare", {"modality": modality, "a": body.sample_a, "b": body.sample_b,
                             "similarity": res["similarity"], "decision": decision})
    return res


@router.post("/{modality}/evaluate")
def evaluate(modality: str, body: EvaluateIn):
    try:
        res = _guard(_mod(modality).evaluate, body.params, body.n_subjects)
    except NotImplementedError:
        raise HTTPException(501, "Для этого модуля оценка метрик пока не реализована")
    record_event("evaluate", {"modality": modality, "eer": res["eer"]})
    return res


@router.post("/{modality}/relief")
def relief(modality: str, body: ProcessIn):
    try:
        res = _guard(_mod(modality).relief, body.sample_id, body.params)
    except NotImplementedError:
        raise HTTPException(501, "3D-визуализация для этого модуля пока не реализована")
    record_event("relief3d", {"modality": modality, "sample": body.sample_id})
    return res
