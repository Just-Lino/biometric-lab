from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.services import experiments

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


@router.get("")
def list_all(limit: int = 100):
    return experiments.list_experiments(limit)


@router.get("/export.csv")
def export():
    return PlainTextResponse(experiments.export_csv(), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": "attachment; filename=experiments.csv"})


@router.delete("")
def clear():
    experiments.clear_experiments()
    return {"ok": True}
