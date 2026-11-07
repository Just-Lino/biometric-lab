import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1 import assistant, biometrics, code, experiments, training, tutor
from app.core.config import settings
from app.core import i18n
from app.core.db import close_pools, init_db
from app.core.session import SecurityHeadersMiddleware, SessionMiddleware
from labplus import router as labplus_router  # дополнение LAB+: голос, метрики, устойчивость, стандарты

WEB = Path(__file__).parent / "web"
# Pyodide (запуск примеров практикума в браузере): .wasm и .whl должны отдаваться с правильным типом, а .js - как JavaScript
# (в Windows реестр иногда отдаёт .js как text/plain, и браузер отказывается выполнять скрипт воркера)
for _ext, _type in ((".wasm", "application/wasm"), (".whl", "application/zip"), (".js", "text/javascript"), (".mjs", "text/javascript")):
    mimetypes.add_type(_type, _ext)


class SelectiveGZip(GZipMiddleware):
    """Не сжимает файлы Pyodide: .wasm и .whl большие и уже сжаты, а повторное сжатие на каждый запрос только грузит процессор."""

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/vendor/pyodide/"):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


class CachedStatic(StaticFiles):
    """Статика с кэшированием на сутки (Pyodide занимает десятки мегабайт и меняется только при обновлении версии)."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if settings.auto_download_models:
        from app.modules.face import models
        models.download_missing_async()
    yield
    close_pools()  # пулы соединений PostgreSQL закрываются при остановке; для SQLite ничего не делает


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(i18n.I18nMiddleware)  # язык запроса (ru, kk, en) и перевод JSON-ответов; внутренний слой: до сжатия GZip
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SessionMiddleware)  # анонимная сессия (cookie): загруженные изображения видны только их владельцу
app.add_middleware(SelectiveGZip, minimum_size=1024)  # JSON со снимками обучения и картинками сжимается в разы
# офлайн-режим: если выполнить `python -m scripts.download_assets`, Plotly и Three.js берутся из app/web/vendor, а не из интернета
app.mount("/vendor", CachedStatic(directory=WEB / "vendor", check_dir=False), name="vendor")
CDN_LIBS = {"https://cdn.plot.ly/plotly-2.35.2.min.js": "plotly.min.js",
            "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js": "three.min.js"}


@app.get("/codelab/safety.py", include_in_schema=False)
def codelab_safety():
    """Проверка кода и урезанные builtins: тот же файл, что использует сервер, чтобы сообщения о запретах в браузере совпадали."""
    return PlainTextResponse((Path(__file__).parent / "sandbox" / "safety.py").read_text(encoding="utf-8"),
                             media_type="text/x-python; charset=utf-8", headers={"Cache-Control": "no-cache"})


# клиентская часть трёх языков: i18n.js и каталог перевода выбранного языка
app.mount("/i18n", StaticFiles(directory=WEB / "i18n", check_dir=False), name="i18n")


@app.get("/api/v1/i18n/{lang}", tags=["i18n"])
def i18n_catalog(lang: str):
    """Каталог перевода для браузера: словарь «русский текст -> перевод»."""
    if lang not in i18n.LANGS:
        raise HTTPException(404, "Неизвестный язык: " + lang)
    return {"lang": lang, "entries": i18n.client_catalog(lang)}


# воркер Pyodide и браузерный раннер примеров практикума (без эталонных решений и тестов)
app.mount("/codelab", StaticFiles(directory=WEB / "codelab", check_dir=False), name="codelab")
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
    return HTMLResponse(i18n.inject_page(html), headers={"Cache-Control": "no-cache", "Vary": "Cookie, Accept-Language"})
