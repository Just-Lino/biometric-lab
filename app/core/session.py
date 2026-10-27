"""Анонимная сессия посетителя: случайный идентификатор в cookie, без аккаунтов и паролей.

Нужна, чтобы загруженные изображения одного посетителя не были видны другим посетителям того же сервера.
Идентификатор лежит в contextvar, поэтому модули читают его через current_sid() и не знают про HTTP."""
import secrets
from contextvars import ContextVar

COOKIE = "lab_sid"
_sid: ContextVar[str | None] = ContextVar("lab_sid", default=None)


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

        async def send_with_cookie(message):
            if is_new and message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"set-cookie", f"{COOKIE}={sid}; Path=/; HttpOnly; SameSite=Lax".encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_cookie)
        finally:
            _sid.reset(token)
