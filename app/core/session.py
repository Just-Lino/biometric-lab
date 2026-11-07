"""Анонимная сессия посетителя: случайный идентификатор в cookie, без аккаунтов и паролей.

Нужна, чтобы загруженные изображения одного посетителя не были видны другим посетителям того же сервера.
Идентификатор лежит в contextvar, поэтому модули читают его через current_sid() и не знают про HTTP."""
import hashlib
import secrets

from app.core.config import settings
from contextvars import ContextVar

COOKIE = "lab_sid"
_sid: ContextVar[str | None] = ContextVar("lab_sid", default=None)


_subject: ContextVar[str | None] = ContextVar("lab_subject", default=None)


def current_subject() -> str | None:
    """ID пользователя от сайта-хозяина (заголовок SUBJECT_HEADER) или None, если лаборатория используется анонимно."""
    return _subject.get()


def owner() -> tuple[str | None, str | None]:
    """(session_id, subject) для записи в базу. Сам идентификатор cookie в базу не пишем: хранится его короткий хэш."""
    sid = _sid.get()
    return (hashlib.sha256(sid.encode()).hexdigest()[:16] if sid else None), _subject.get()


def limit_key() -> str:
    """Ключ лимитов: пользователь сайта, если он известен (его нельзя обойти очисткой cookie), иначе сессия."""
    return "u:" + _subject.get() if _subject.get() else "s:" + (_sid.get() or "-")


def current_sid() -> str | None:
    """Идентификатор текущего посетителя; вне запроса (тесты, скрипты) None."""
    return _sid.get()


class SessionMiddleware:
    """ASGI-прослойка: читает cookie lab_sid или выдаёт новую и кладёт её в contextvar на время запроса."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        sid = None
        for key, value in scope.get("headers", []):
            if key == b"cookie":
                for part in value.decode("latin-1").split(";"):
                    name, _, val = part.strip().partition("=")
                    if name == COOKIE and 16 <= len(val) <= 64 and val.isalnum():
                        sid = val
        is_new = sid is None
        if is_new:
            sid = secrets.token_hex(16)
        token = _sid.set(sid)
        subject = None
        if settings.subject_header:
            want = settings.subject_header.lower().encode()
            for key, value in scope.get("headers", []):
                if key == want:
                    v = value.decode("latin-1").strip()
                    if 0 < len(v) <= 128 and v.isprintable():
                        subject = v
        stoken = _subject.set(subject)

        async def send_with_cookie(message):
            if is_new and message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"set-cookie", f"{COOKIE}={sid}; Path=/; HttpOnly; SameSite=Lax{'; Secure' if settings.cookie_secure else ''}".encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_cookie)
        finally:
            _sid.reset(token)
            _subject.reset(stoken)


class SecurityHeadersMiddleware:
    """Базовые защитные заголовки. Встраивание в iframe не запрещаем (платформа может показать лабораторию внутри себя),
    CSP не задаём: страница использует встроенные скрипты и библиотеки с CDN. HSTS включается только вместе с COOKIE_SECURE."""

    HEADERS = [
        (b"x-content-type-options", b"nosniff"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"permissions-policy", b"microphone=(self), camera=(), geolocation=()"),
    ]

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", [])) + self.HEADERS
                if settings.cookie_secure:
                    headers.append((b"strict-transport-security", b"max-age=31536000"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)
