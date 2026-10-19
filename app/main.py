from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.v1 import biometrics, code, experiments, training, tutor
from app.core.config import settings
from app.core.db import init_db

WEB = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if settings.auto_download_models:
        from app.modules.face import models
        models.download_missing_async()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(biometrics.router)
app.include_router(tutor.router)
app.include_router(experiments.router)
app.include_router(training.router)
app.include_router(code.router)


@app.get("/health")
def health():
    return {"status": "ok", "tutor": settings.tutor_provider}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")
