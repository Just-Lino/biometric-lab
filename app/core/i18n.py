"""Три языка интерфейса и сообщений: русский (исходный), казахский (kk) и английский (en).

Как это устроено
- Исходный язык кода и данных - русский. Переводы лежат в каталогах app/i18n/*.i18n (блоки «ru> ...», «kk> ...», «en> ...»).
  Ключ записи - русский текст; в нём можно использовать места {0}, {1}, ... для подставляемых значений (числа, названия).
- Язык запроса определяется по cookie lab_lang, параметру ?lang= или заголовку Accept-Language; по умолчанию русский.
- Все JSON-ответы API при языке kk/en переводятся на выходе (I18nMiddleware): точное совпадение, затем шаблоны с {n},
  затем перевод по предложениям. Непереведённый текст остаётся русским, а не ломается.
- Задания практикума переводятся отдельными файлами app/tasks/i18n/<язык>/<id>.json (см. app/sandbox/tasks.py).
- Страница отдаёт клиенту каталог своего языка; в браузере те же правила реализованы в app/web/i18n/i18n.js.

Казахский перевод выполнен автоматически и должен быть проверен носителем языка (см. README, раздел «Три языка»)."""
import json
import re
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

LANGS = ("ru", "kk", "en")
DEFAULT_LANG = "ru"
COOKIE = "lab_lang"
I18N_DIR = Path(__file__).resolve().parents[1] / "i18n"

_lang: ContextVar[str] = ContextVar("lab_lang", default=DEFAULT_LANG)
CYR = re.compile(r"[А-Яа-яЁё]")
_PLACEHOLDER = re.compile(r"\{(\d+)\}")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])(\s+)|(\n+)")


def current_lang() -> str:
    """Язык текущего запроса (вне запроса - русский)."""
    return _lang.get()


def set_lang(lang: str):
    """Устанавливает язык и возвращает токен для сброса (используется middleware и тестами)."""
    return _lang.set(lang if lang in LANGS else DEFAULT_LANG)


def reset_lang(token) -> None:
    _lang.reset(token)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _unescape(v: str) -> str:
    return v.replace("\\n", "\n").replace("\\\\", "\\")


def parse_catalog(text: str) -> list[dict]:
    """Блоки, разделённые пустой строкой: строки «ru> ...», «kk> ...», «en> ...»; строки с # - комментарии."""
    entries, cur = [], {}
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip():
            if cur:
                entries.append(cur)
                cur = {}
            continue
        if line.startswith("#"):
            continue
        m = re.match(r"^(ru|kk|en)>\s?(.*)$", line)
        if not m:
            raise ValueError(f"Неверная строка каталога: {line[:80]!r}")
        cur[m.group(1)] = _unescape(m.group(2))
    if cur:
        entries.append(cur)
    return entries


class Catalog:
    def __init__(self, lang: str, entries: list[dict]):
        self.lang = lang
        self.exact: dict[str, str] = {}
        self.templates: list[tuple[re.Pattern, list[int], str, int]] = []
        for e in entries:
            ru, tr = e.get("ru"), e.get(lang)
            if not ru or not tr:
                continue
            self.exact[ru] = tr
            self.exact.setdefault(norm(ru), tr.strip())
            if _PLACEHOLDER.search(ru):
                key, val = ru.strip(), tr.strip()                 # lookup() всегда работает с обрезанным текстом, пробелы по краям возвращает translate()
                parts = _PLACEHOLDER.split(key)                   # [текст, номер, текст, номер, ..., текст]
                lits, nums = parts[0::2], [int(x) for x in parts[1::2]]
                rx = re.compile("^" + "(.*?)".join(re.escape(p) for p in lits) + "$", re.S if "\n" in key else 0)   # шаблон без \n не должен «съедать» многострочный текст
                self.templates.append((rx, nums, val, sum(len(p) for p in lits)))
        self.templates.sort(key=lambda t: -t[3])                  # сначала самые определённые (больше постоянного текста)

    def lookup(self, text: str):
        """Перевод целой строки без разбиения на предложения или None."""
        core = text.strip()
        hit = self.exact.get(core) or self.exact.get(norm(core))
        if hit is not None:
            return hit
        if core[:1].isalpha():                                    # код выводит название с другой первой буквой (.lower(), _cap): ищем вариант с другим регистром
            alt = core[:1].upper() + core[1:] if core[:1].islower() else core[:1].lower() + core[1:]
            hit = self.exact.get(alt)
            if hit is not None:
                return hit[:1].lower() + hit[1:] if core[:1].islower() else hit[:1].upper() + hit[1:]
        for rx, nums, tr, _ in self.templates:
            m = rx.match(core)
            if m and not any(_SENT_SPLIT.search(g.strip()) for g in m.groups()):    # шаблон не «съедает» несколько предложений: их переведёт разбиение
                groups = m.groups()
                vals = {n: self.translate(g) for n, g in zip(nums, groups)}
                return _PLACEHOLDER.sub(lambda mm: vals.get(int(mm.group(1)), mm.group(0)), tr)
        return None

    def translate(self, text: str) -> str:
        if not text or not CYR.search(text):
            return text
        lead = text[:len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()):]
        core = text.strip()
        hit = self.lookup(core)
        if hit is not None:
            return lead + hit + trail
        # составное сообщение из нескольких предложений: переводим по предложениям
        pieces, last = [], 0
        for m in _SENT_SPLIT.finditer(core):
            pieces.append((core[last:m.start()], m.group(0)))
            last = m.end()
        pieces.append((core[last:], ""))
        if len(pieces) == 1:
            return text
        out, any_hit = [], False
        for sent, sep in pieces:
            h = self.lookup(sent) if sent and CYR.search(sent) else None
            any_hit = any_hit or h is not None
            if h is not None:                                     # сохраняем отступ и хвостовые пробелы строки (код в заданиях)
                h = sent[:len(sent) - len(sent.lstrip())] + h + sent[len(sent.rstrip()):]
            out.append((h if h is not None else sent) + sep)
        return lead + "".join(out) + trail if any_hit else text


@lru_cache(maxsize=8)
def _load(lang: str, stamp: tuple) -> Catalog:
    entries = []
    for f in sorted(I18N_DIR.glob("*.i18n")):
        entries += parse_catalog(f.read_text(encoding="utf-8"))
    return Catalog(lang, entries)


def get_catalog(lang: str) -> Catalog:
    """Каталог языка; при изменении файлов перечитывается (штамп по времени изменения)."""
    stamp = tuple((f.name, f.stat().st_mtime_ns) for f in sorted(I18N_DIR.glob("*.i18n")))
    return _load(lang, stamp)


def translate(text, lang: str | None = None) -> str:
    """Перевод русского текста на язык запроса (или lang). Не-строки возвращаются как есть."""
    lang = lang or current_lang()
    if lang == DEFAULT_LANG or not isinstance(text, str):
        return text
    return get_catalog(lang).translate(text)


def translate_obj(obj, lang: str | None = None):
    """Рекурсивно переводит строки в JSON-структуре (ключи словарей не трогает)."""
    lang = lang or current_lang()
    if lang == DEFAULT_LANG:
        return obj
    cat = get_catalog(lang)

    def walk(x):
        if isinstance(x, str):
            return cat.translate(x) if CYR.search(x) else x
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x
    return walk(obj)


CLIENT_FILES = ("ui_", "client")        # в браузер уходят только тексты интерфейса и сообщения браузерного запуска кода (остальное переводит сервер)


@lru_cache(maxsize=8)
def _load_client(lang: str, stamp: tuple) -> Catalog:
    entries = []
    for f in sorted(I18N_DIR.glob("*.i18n")):
        if f.stem.startswith(CLIENT_FILES):
            entries += parse_catalog(f.read_text(encoding="utf-8"))
    return Catalog(lang, entries)


def client_catalog(lang: str) -> dict:
    """Простой словарь «русский ключ -> перевод» для браузера."""
    if lang == DEFAULT_LANG or lang not in LANGS:
        return {}
    stamp = tuple((f.name, f.stat().st_mtime_ns) for f in sorted(I18N_DIR.glob("*.i18n")))
    return dict(_load_client(lang, stamp).exact)


def inject_page(html: str, lang: str | None = None) -> str:
    """Вставляет в HTML-страницу язык, каталог перевода и клиентскую часть (i18n.js); для русского каталог пуст."""
    lang = lang or current_lang()
    cat = json.dumps(client_catalog(lang), ensure_ascii=False).replace("</", "<\\/")      # «</» внутри <script> закрыл бы тег
    tag = f'<script>window.LAB_LANG={json.dumps(lang)};window.LAB_CAT={cat};</script><script src="/i18n/i18n.js"></script>'
    html = re.sub(r'<html lang="[a-z]{2}"', f'<html lang="{lang}"', html, count=1)
    return html.replace("</head>", tag + "</head>", 1) if "</head>" in html else tag + html


def detect_lang(cookie_header: str = "", query: str = "", accept: str = "") -> str:
    """Порядок: параметр ?lang=, cookie lab_lang, Accept-Language, иначе русский."""
    m = re.search(r"(?:^|&)lang=(ru|kk|en)(?:&|$)", query or "")
    if m:
        return m.group(1)
    for part in (cookie_header or "").split(";"):
        name, _, val = part.strip().partition("=")
        if name == COOKIE and val in LANGS:
            return val
    for tag in (accept or "").split(","):
        code = tag.split(";")[0].strip().lower()[:2]
        if code in LANGS:
            return code
    return DEFAULT_LANG


class I18nMiddleware:
    """ASGI: определяет язык запроса и переводит JSON-ответы (для kk и en) на выходе."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        lang = detect_lang(headers.get("cookie", ""), scope.get("query_string", b"").decode("latin-1"), headers.get("accept-language", ""))
        token = _lang.set(lang)
        state = {"buffer": False, "start": None, "chunks": []}

        async def send_translated(message):
            if message["type"] == "http.response.start":
                hdrs = message.get("headers", [])
                ctype = next((v.decode("latin-1") for k, v in hdrs if k.lower() == b"content-type"), "")
                hdrs = [(k, v) for k, v in hdrs if k.lower() != b"content-language"] + [(b"content-language", lang.encode())]
                message = {**message, "headers": hdrs}
                if ctype.startswith("application/json") and not scope.get("path", "").startswith("/api/v1/i18n/"):
                    hdrs = [(k, v) for k, v in hdrs if k.lower() != b"vary"] + [(b"vary", b"Cookie, Accept-Language")]
                    state["start"] = {**message, "headers": hdrs}
                    state["buffer"] = lang != DEFAULT_LANG
                    if not state["buffer"]:
                        await send(state["start"])
                    return
                await send(message)
                return
            if message["type"] == "http.response.body" and state["start"] is not None and state["buffer"]:
                state["chunks"].append(message.get("body", b""))
                if message.get("more_body"):
                    return
                body = b"".join(state["chunks"])
                try:
                    if CYR.search(body.decode("utf-8", "replace")):
                        body = json.dumps(translate_obj(json.loads(body), lang), ensure_ascii=False).encode("utf-8")
                except (ValueError, UnicodeDecodeError):
                    pass
                start = state["start"]
                start["headers"] = [(k, v) for k, v in start["headers"] if k.lower() != b"content-length"] + [(b"content-length", str(len(body)).encode())]
                await send(start)
                await send({"type": "http.response.body", "body": body})
                return
            await send(message)

        try:
            await self.app(scope, receive, send_translated)
        finally:
            _lang.reset(token)
