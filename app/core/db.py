"""Слой хранения данных: SQLite (по умолчанию) или PostgreSQL по DATABASE_URL.

Весь SQL в app/services/* написан один раз с плейсхолдерами «?» (диалект SQLite). Для PostgreSQL этот слой сам:
  - переписывает «?» в «%s» (только вне строковых литералов);
  - добавляет RETURNING id в connection.insert() вместо cursor.lastrowid;
  - даёт json_field() для чтения поля из JSON-текста (у SQLite json_extract, у PostgreSQL ::jsonb ->>);
  - создаёт ту же схему (таблицы experiments, attempts, events, training_runs) с типами нужного диалекта.
Сервисы работают с обоими вариантами без изменений: connect() отдаёт один и тот же интерфейс (execute, insert).

Библиотека для PostgreSQL: psycopg 3 с пулом соединений (psycopg_pool). Она подключается только при адресе postgresql://,
поэтому локальная работа и тесты на SQLite не требуют её установки."""
import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager

from app.core.config import settings

log = logging.getLogger("uvicorn.error")

# Колонки таблиц без id и ts (они отличаются по диалектам). {real} подставляется как REAL или DOUBLE PRECISION.
TABLES = {
    "experiments": "modality TEXT, sample_a TEXT, sample_b TEXT, params TEXT, similarity {real}, threshold {real}, "
                   "decision TEXT, n_a INTEGER, n_b INTEGER, matched INTEGER, session_id TEXT, subject TEXT",
    "attempts": "task_id TEXT, score {real}, max_score {real}, passed_basic INTEGER, total_basic INTEGER, passed_edge INTEGER, "
                "total_edge INTEGER, run_ok INTEGER, answer_ok INTEGER, code TEXT, details TEXT, session_id TEXT, subject TEXT",
    "events": "type TEXT, payload TEXT, session_id TEXT, subject TEXT",
    "training_runs": "modality TEXT, config TEXT, summary TEXT, curves TEXT, session_id TEXT, subject TEXT",
}
# Индексы под реальные запросы: число и история попыток по заданию, события по типу.
INDEXES = [
    "CREATE INDEX IF NOT EXISTS attempts_task_idx ON attempts(task_id)",
    "CREATE INDEX IF NOT EXISTS events_type_idx ON events(type)",
]
PG_INIT_LOCK = 727274  # ключ advisory-блокировки: несколько экземпляров не создают схему одновременно


def schema_statements(dialect: str) -> list[str]:
    pg = dialect == "postgres"
    pk = "BIGSERIAL PRIMARY KEY" if pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    # Время хранится текстом в UTC в одном формате «ГГГГ-ММ-ДД ЧЧ:ММ:СС» у обоих диалектов, поэтому API и CSV не отличаются.
    ts = "TEXT DEFAULT to_char(timezone('UTC', now()), 'YYYY-MM-DD HH24:MI:SS')" if pg else "TEXT DEFAULT CURRENT_TIMESTAMP"
    real = "DOUBLE PRECISION" if pg else "REAL"
    stmts = [f"CREATE TABLE IF NOT EXISTS {name} (id {pk}, ts {ts}, {cols.format(real=real)})" for name, cols in TABLES.items()]
    return stmts + INDEXES


# ───────────────────────── адреса и диалекты ─────────────────────────
def _scheme(url: str) -> str:
    return url.split(":", 1)[0].lower()


def dialect_of(url: str) -> str:
    scheme = _scheme(url)
    if scheme == "sqlite":
        return "sqlite"
    if scheme in ("postgres", "postgresql") or scheme.startswith("postgresql+"):
        return "postgres"
    raise ValueError(f"Неподдерживаемый адрес базы данных «{scheme}:». Допустимы sqlite:/// и postgresql://")


def current_dialect() -> str:
    return dialect_of(settings.effective_database_url)


def sqlite_path(url: str) -> str:
    """sqlite:///data/lab.db -> data/lab.db; sqlite:////var/lab.db -> /var/lab.db; sqlite:///C:/x/lab.db -> C:/x/lab.db."""
    if not url.lower().startswith("sqlite:///"):
        raise ValueError("Адрес SQLite должен начинаться с sqlite:///")
    path = url[len("sqlite:///"):].split("?", 1)[0]
    if not path or path == ":memory:":
        raise ValueError("Для SQLite нужен путь к файлу: база в памяти не подходит, у каждого соединения она была бы своя")
    return path


def postgres_url(url: str) -> str:
    """postgres:// и postgresql+psycopg:// (форма SQLAlchemy) приводятся к postgresql:// для psycopg."""
    return "postgresql://" + url.split("://", 1)[1]


def to_pg_sql(sql: str) -> str:
    """Плейсхолдеры «?» -> «%s» вне строковых литералов; знак «%» экранируется, потому что psycopg считает его параметром."""
    parts = re.split(r"('(?:[^']|'')*')", sql)
    out = []
    for i, part in enumerate(parts):
        part = part.replace("%", "%%")
        out.append(part if i % 2 else part.replace("?", "%s"))
    return "".join(out)


def json_field(column: str, key: str, dialect: str | None = None) -> str:
    """Фрагмент SQL: поле key из JSON-текста в колонке column."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column) or not re.fullmatch(r"[A-Za-z0-9_]+", key):
        raise ValueError("Недопустимое имя колонки или ключа JSON")
    if (dialect or current_dialect()) == "postgres":
        return f"({column}::jsonb ->> '{key}')"
    return f"json_extract({column}, '$.{key}')"


# ───────────────────────── строки и соединение ─────────────────────────
class Row(tuple):
    """Строка результата PostgreSQL с доступом по номеру и по имени колонки; dict(row) работает, как у sqlite3.Row."""

    def __new__(cls, names, values):
        row = super().__new__(cls, values)
        row._names = tuple(names)
        return row

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return super().__getitem__(self._names.index(key))
            except ValueError:
                raise KeyError(key) from None
        return super().__getitem__(key)

    def keys(self):
        return list(self._names)


def _pg_row_factory(cursor):
    names = [c.name for c in (cursor.description or [])]
    return lambda values: Row(names, values)


class Conn:
    """Единый интерфейс соединения для сервисов: execute(sql, params) и insert(sql, params) -> id."""

    def __init__(self, raw, dialect: str):
        self._raw, self.dialect = raw, dialect

    def execute(self, sql: str, params=()):
        return self._raw.execute(to_pg_sql(sql) if self.dialect == "postgres" else sql, params)

    def insert(self, sql: str, params=()) -> int:
        """INSERT с возвратом id новой строки: у SQLite lastrowid, у PostgreSQL RETURNING id."""
        if self.dialect == "postgres":
            return self._raw.execute(to_pg_sql(sql) + " RETURNING id", params).fetchone()[0]
        return self._raw.execute(sql, params).lastrowid


# ───────────────────────── SQLite ─────────────────────────
_wal_checked: set[str] = set()
_wal_lock = threading.Lock()


def _sqlite_connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)  # ждать освобождения блокировки до 15 секунд, а не падать сразу
    conn.row_factory = sqlite3.Row
    with _wal_lock:
        first = path not in _wal_checked
        _wal_checked.add(path)
    if first:
        # WAL: читатели не блокируют писателя, при 20-30 студентах это убирает «database is locked». Режим хранится в файле базы.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as e:
            log.warning("SQLite: не удалось включить режим WAL (%s), работаем в обычном режиме", e)
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ───────────────────────── PostgreSQL ─────────────────────────
_pools: dict = {}
_pools_lock = threading.Lock()


def _pg_pool(url: str):
    with _pools_lock:
        pool = _pools.get(url)
        if pool is None:
            try:
                from psycopg_pool import ConnectionPool
            except ImportError as e:
                raise RuntimeError('Для PostgreSQL установите пакеты: python -m pip install "psycopg[binary,pool]"') from e
            kwargs = {}
            if hasattr(ConnectionPool, "check_connection"):
                kwargs["check"] = ConnectionPool.check_connection  # отбрасывает соединения, закрытые сервером по бездействию
            pool = ConnectionPool(url, min_size=1, max_size=settings.db_pool_max_size, timeout=15, open=True,
                                  kwargs={"row_factory": _pg_row_factory}, **kwargs)
            _pools[url] = pool
        return pool


def close_pools() -> None:
    """Закрыть пулы соединений PostgreSQL (при остановке приложения и в тестах)."""
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.close()


# ───────────────────────── публичный интерфейс ─────────────────────────
@contextmanager
def connect():
    """Соединение на одну операцию: commit при успехе, rollback при исключении. Диалект выбирается по DATABASE_URL при каждом вызове."""
    url = settings.effective_database_url
    if dialect_of(url) == "postgres":
        with _pg_pool(postgres_url(url)).connection() as raw:  # пул сам делает commit или rollback
            yield Conn(raw, "postgres")
        return
    raw = _sqlite_connect(sqlite_path(url))
    try:
        yield Conn(raw, "sqlite")
        raw.commit()
    except BaseException:
        raw.rollback()
        raise
    finally:
        raw.close()


OWNER_COLUMNS = ("session_id", "subject")  # кто выполнил действие: хэш анонимной сессии и ID пользователя сайта (может быть пустым)


def _add_owner_columns(c) -> None:
    """Базы, созданные до появления колонок session_id и subject, дополняются ими при запуске."""
    for table in TABLES:
        if c.dialect == "postgres":
            for col in OWNER_COLUMNS:
                c.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} TEXT")
        else:
            have = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
            for col in OWNER_COLUMNS:
                if col not in have:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")


def init_db() -> None:
    with connect() as c:
        if c.dialect == "postgres":
            c.execute(f"SELECT pg_advisory_xact_lock({PG_INIT_LOCK})")
        for stmt in schema_statements(c.dialect):
            c.execute(stmt)
        _add_owner_columns(c)
