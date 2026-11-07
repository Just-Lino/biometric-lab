"""Слой хранения: SQLite по умолчанию и PostgreSQL по DATABASE_URL.

Чистые функции (перевод SQL, схема, адреса) проверяются всегда. Сервисные тесты идут на SQLite всегда,
а на PostgreSQL только если задана переменная окружения TEST_DATABASE_URL (пустая тестовая база: таблицы пересоздаются)."""
import os
import re
import threading

import pytest

from app.core import db
from app.core.config import settings
from app.services import attempts, events, experiments, training_runs

TEST_PG = os.environ.get("TEST_DATABASE_URL")
TABLE_NAMES = ("experiments", "attempts", "events", "training_runs")


# ───────────────────────── чистые функции ─────────────────────────
def test_placeholders_are_translated_outside_string_literals_only():
    sql = "SELECT * FROM t WHERE a=? AND b LIKE '50%?' AND c=? LIMIT ?"
    assert db.to_pg_sql(sql) == "SELECT * FROM t WHERE a=%s AND b LIKE '50%%?' AND c=%s LIMIT %s"
    assert db.to_pg_sql("SELECT 'it''s ?', ?") == "SELECT 'it''s ?', %s"        # удвоенная кавычка не закрывает литерал
    assert db.to_pg_sql("SELECT 1") == "SELECT 1"


def test_schema_per_dialect():
    sqlite = " ".join(db.schema_statements("sqlite"))
    pg = " ".join(db.schema_statements("postgres"))
    assert "AUTOINCREMENT" in sqlite and "CURRENT_TIMESTAMP" in sqlite and "DOUBLE PRECISION" not in sqlite
    assert "AUTOINCREMENT" not in pg and "BIGSERIAL PRIMARY KEY" in pg and "DOUBLE PRECISION" in pg
    assert not re.search(r"\bREAL\b", pg)                          # REAL в PostgreSQL это float4: теряется точность оценок
    for dialect in ("sqlite", "postgres"):
        stmts = db.schema_statements(dialect)
        assert sum(s.startswith("CREATE TABLE") for s in stmts) == 4
        for name in TABLE_NAMES:
            assert any(f"EXISTS {name} " in s for s in stmts)
        assert any("attempts_task_idx" in s for s in stmts) and any("events_type_idx" in s for s in stmts)


@pytest.mark.parametrize("url, dialect", [
    ("sqlite:///data/lab.db", "sqlite"), ("postgresql://u:p@h:5432/db", "postgres"), ("postgres://u:p@h/db", "postgres"),
    ("postgresql+psycopg://u:p@h/db", "postgres")])
def test_dialect_detection(url, dialect):
    assert db.dialect_of(url) == dialect


def test_unknown_scheme_is_rejected_with_clear_message():
    with pytest.raises(ValueError, match="sqlite:/// и postgresql://"):
        db.dialect_of("mysql://u:p@h/db")


def test_sqlite_path_from_url():
    assert db.sqlite_path("sqlite:///data/lab.db") == "data/lab.db"
    assert db.sqlite_path("sqlite:////var/lab/lab.db") == "/var/lab/lab.db"
    assert db.sqlite_path("sqlite:///C:/Users/x/lab.db") == "C:/Users/x/lab.db"
    for bad in ("sqlite://", "sqlite:///:memory:", "data/lab.db"):
        with pytest.raises(ValueError):
            db.sqlite_path(bad)


def test_postgres_url_normalisation():
    assert db.postgres_url("postgres://u:p@h/db") == "postgresql://u:p@h/db"
    assert db.postgres_url("postgresql+psycopg://u:p@h/db?host=/cloudsql/x") == "postgresql://u:p@h/db?host=/cloudsql/x"


def test_json_field_per_dialect_and_injection_guard():
    assert db.json_field("payload", "task", "sqlite") == "json_extract(payload, '$.task')"
    assert db.json_field("payload", "task", "postgres") == "(payload::jsonb ->> 'task')"
    for col, key in (("payload; DROP TABLE x", "task"), ("payload", "task' OR '1'='1"), ("1payload", "task")):
        with pytest.raises(ValueError):
            db.json_field(col, key, "sqlite")


def test_row_supports_index_name_and_dict():
    r = db.Row(["id", "code"], (7, "x"))
    assert r[0] == 7 and r["code"] == "x" and r[-1] == "x" and dict(r) == {"id": 7, "code": "x"} and r.keys() == ["id", "code"]
    with pytest.raises(KeyError):
        r["nope"]


def test_postgres_without_driver_gives_russian_hint(monkeypatch):
    """Если psycopg не установлен, ошибка говорит, что поставить, а не падает ImportError без объяснения."""
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("psycopg"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    monkeypatch.setattr(db, "_pools", {})
    with pytest.raises(RuntimeError, match="psycopg"):
        db._pg_pool("postgresql://u:p@h/db")


# ───────────────────────── сервисы на обеих базах ─────────────────────────
@pytest.fixture(params=["sqlite", "postgres"])
def backend(request, tmp_path, monkeypatch):
    if request.param == "postgres":
        if not TEST_PG:
            pytest.skip("TEST_DATABASE_URL не задан: проверка PostgreSQL пропущена")
        monkeypatch.setattr(settings, "database_url", TEST_PG)
        with db.connect() as c:
            for t in TABLE_NAMES:
                c.execute(f"DROP TABLE IF EXISTS {t}")
    else:
        monkeypatch.setattr(settings, "database_url", None)
        monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"))
    db.init_db()
    yield request.param
    db.close_pools()


G = dict(score=6.5, max_score=10.0, passed_basic=3, total_basic=4, passed_edge=2, total_edge=4, run_ok=True, answer_ok=False)


def test_init_db_is_idempotent(backend):
    db.init_db()
    db.init_db()
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0


def test_experiments_roundtrip_and_csv(backend):
    experiments.add_experiment("fingerprint", "f01_i1", "f01_i2", {"matcher": "hough"}, 0.92, 0.5, "MATCH", 37, 39, 35)
    experiments.add_experiment("face", "a", "b", {}, 0.3, 0.4, "NO MATCH", 0, 0, 0)
    rows = experiments.list_experiments(10)
    assert [r["modality"] for r in rows] == ["face", "fingerprint"]                    # новые сверху
    assert rows[1]["similarity"] == pytest.approx(0.92) and rows[1]["matched"] == 35 and '"hough"' in rows[1]["params"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", rows[0]["ts"])             # у обеих баз один формат времени
    csv_text = experiments.export_csv()
    assert csv_text.splitlines()[0].startswith("id,ts,modality") and "f01_i1" in csv_text
    experiments.clear_experiments()
    assert experiments.list_experiments() == []


def test_attempts_ids_counts_history_and_summary(backend):
    first = attempts.add_attempt("binarize", "code1", G, {"k": 1})
    second = attempts.add_attempt("binarize", "code2", {**G, "score": 9.0, "total_basic": 0}, {"k": 2})
    attempts.add_attempt("verify", "code3", G, {})
    assert isinstance(first, int) and second > first                                    # insert() возвращает id новой строки
    assert attempts.count("binarize") == 2 and attempts.count_valid("binarize") == 1    # попытка без тестов не считается
    assert [h["id"] for h in attempts.history("binarize")] == [second, first]
    assert attempts.last_code("binarize") == "code2" and attempts.last_code("nope") is None
    events.record_event("hint", {"task": "binarize", "level": 1})
    events.record_event("hint", {"task": "binarize", "level": 2})
    events.record_event("compare", {"task": "binarize"})                                # другой тип не учитывается
    s = attempts.summary()
    assert s["binarize"] == {"attempts": 2, "best": 9.0, "hints": 2} and s["verify"]["hints"] == 0


def test_training_runs_roundtrip(backend):
    rid = training_runs.save("fingerprint", {"epochs": 20}, {"eer": 0.06}, {"loss": [3.0, 1.0]})
    training_runs.save("face", {}, {}, {})
    runs = training_runs.list_runs()
    assert runs[1]["id"] == rid and runs[1]["config"] == {"epochs": 20} and runs[1]["curves"] == {"loss": [3.0, 1.0]}
    training_runs.clear()
    assert training_runs.list_runs() == []


def test_failed_transaction_is_rolled_back(backend):
    with pytest.raises(RuntimeError):
        with db.connect() as c:
            c.insert("INSERT INTO events(type, payload) VALUES (?, ?)", ("x", "{}"))
            raise RuntimeError("сбой посреди операции")
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_many_concurrent_writers_lose_nothing(backend):
    """20-30 студентов пишут в журнал одновременно: ни «database is locked», ни потерянных записей."""
    errors, per_thread, threads = [], 25, 8

    def work():
        try:
            for i in range(per_thread):
                events.record_event("compare", {"i": i})
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=work) for _ in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:3]
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM events").fetchone()[0] == per_thread * threads


# ───────────────────────── только SQLite ─────────────────────────
def test_sqlite_uses_wal_and_busy_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_url", None)
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "wal.db"))
    db.init_db()
    with db.connect() as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_sqlite_url_overrides_db_path(tmp_path, monkeypatch):
    target = tmp_path / "from_url.db"
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "ignored.db"))
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{target.as_posix()}")
    db.init_db()
    events.record_event("x", {})
    assert target.exists() and not (tmp_path / "ignored.db").exists()
