"""Кто выполнил действие: хэш анонимной сессии и (необязательно) ID пользователя сайта в заголовке SUBJECT_HEADER."""
import os
import sqlite3
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core import db  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.session import SessionMiddleware, current_subject, limit_key  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def own_db(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "own.db"))
    monkeypatch.setattr(settings, "database_url", None)
    return tmp_path / "own.db"


def last_event(path):
    c = sqlite3.connect(path)
    try:
        return c.execute("SELECT type, session_id, subject FROM events ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        c.close()


def do_action(client, **kw):
    r = client.post("/api/v1/tutor/ask", json={"mode": "explain", "question": "что такое EER", "context": {}}, **kw)
    assert r.status_code == 200, r.text


def test_event_records_session_hash_not_cookie_and_no_subject_by_default(own_db, monkeypatch):
    monkeypatch.setattr(settings, "subject_header", None)
    with TestClient(app) as c:
        do_action(c, headers={"X-User-Id": "student-7"})       # заголовок без настройки игнорируется
        sid = c.cookies.get("lab_sid")
    typ, session_id, subject = last_event(own_db)
    assert typ == "tutor" and subject is None
    assert session_id and session_id != sid and len(session_id) == 16     # сам идентификатор cookie в базу не попадает


def test_subject_from_configured_header(own_db, monkeypatch):
    monkeypatch.setattr(settings, "subject_header", "X-User-Id")
    with TestClient(app) as c:
        do_action(c, headers={"X-User-Id": "student-7"})
    assert last_event(own_db)[2] == "student-7"


@pytest.mark.parametrize("bad", ["", "x" * 129])
def test_bad_subject_is_ignored(own_db, monkeypatch, bad):
    monkeypatch.setattr(settings, "subject_header", "X-User-Id")
    with TestClient(app) as c:
        do_action(c, headers={"X-User-Id": bad})
    assert last_event(own_db)[2] is None


def test_limit_key_prefers_subject_over_session(own_db, monkeypatch):
    from app.core import session
    monkeypatch.setattr(settings, "subject_header", "X-User-Id")
    t1, t2 = session._sid.set("abc"), session._subject.set("u1")
    try:
        assert limit_key() == "u:u1"
    finally:
        session._sid.reset(t1), session._subject.reset(t2)
    t1 = session._sid.set("abc")
    try:
        assert limit_key() == "s:abc"
    finally:
        session._sid.reset(t1)


def test_old_database_without_owner_columns_is_migrated(monkeypatch, tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP, type TEXT, payload TEXT)")
    conn.execute("INSERT INTO events(type, payload) VALUES ('old', '{}')")
    conn.commit(); conn.close()
    monkeypatch.setattr(settings, "db_path", str(path))
    monkeypatch.setattr(settings, "database_url", None)
    db.init_db()
    db.init_db()                                                           # повторный запуск безопасен
    conn = sqlite3.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    assert {"session_id", "subject"} <= cols
    assert conn.execute("SELECT type, subject FROM events").fetchone() == ("old", None)   # старые записи сохранены
    conn.close()


def test_owner_fields_are_not_exposed_by_api(own_db):
    with TestClient(app) as c:
        c.post("/api/v1/biometrics/fingerprint/compare", json={"sample_a": "f01_i1", "sample_b": "f01_i2"})
        rows = c.get("/api/v1/experiments").json()
    assert rows and all("session_id" not in r and "subject" not in r for r in rows)
