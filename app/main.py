from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1 import assistant, biometrics, code, experiments, training, tutor
from app.core.config import settings
from app.core.db import close_pools, init_db
from app.core.session import SecurityHeadersMiddleware, SessionMiddleware
from labplus import router as labplus_router  # дополнение LAB+: голос, метрики, устойчивость, стандарты

WEB = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if settings.auto_download_models:
        from app.modules.face import models
        models.download_missing_async()
    yield
    close_pools()  # пулы соединений PostgreSQL закрываются при остановке; для SQLite ничего не делает


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SessionMiddleware)  # анонимная сессия (cookie): загруженные изображения видны только их владельцу
app.add_middleware(GZipMiddleware, minimum_size=1024)  # JSON со снимками обучения и картинками сжимается в разы
# офлайн-режим: если выполнить `python -m scripts.download_assets`, Plotly и Three.js берутся из app/web/vendor, а не из интернета
app.mount("/vendor", StaticFiles(directory=WEB / "vendor", check_dir=False), name="vendor")
CDN_LIBS = {"https://cdn.plot.ly/plotly-2.35.2.min.js": "plotly.min.js",
            "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js": "three.min.js"}
app.include_router(biometrics.router)
app.include_router(tutor.router)
app.include_router(experiments.router)
app.include_router(training.router)
app.include_router(code.router)
app.include_router(assistant.router)
app.include_router(labplus_router)


@app.get("/health")
def health():
    return {"status": "ok", "tutor": settings.tutor_provider}


@app.get("/", include_in_schema=False)
def index():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for url, name in CDN_LIBS.items():
        if (WEB / "vendor" / name).exists():
            html = html.replace(url, f"/vendor/{name}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})
