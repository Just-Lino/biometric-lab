"""Минимальный слой хранения на SQLite (stdlib).
Позже заменяется на PostgreSQL: весь доступ идёт через репозитории в app/services."""
import os
import sqlite3
from contextlib import contextmanager

from app.core.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT CURRENT_TIMESTAMP,
    modality TEXT, sample_a TEXT, sample_b TEXT,
    params TEXT, similarity REAL, threshold REAL, decision TEXT,
    n_a INTEGER, n_b INTEGER, matched INTEGER
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT CURRENT_TIMESTAMP,
    task_id TEXT, score REAL, max_score REAL,
    passed_basic INTEGER, total_basic INTEGER, passed_edge INTEGER, total_edge INTEGER,
    run_ok INTEGER, answer_ok INTEGER, code TEXT, details TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT CURRENT_TIMESTAMP,
    type TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT CURRENT_TIMESTAMP,
    modality TEXT, config TEXT, summary TEXT, curves TEXT
);
"""


@contextmanager
def connect():
    os.makedirs(os.path.dirname(settings.db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)
