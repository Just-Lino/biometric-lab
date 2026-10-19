"""Репозиторий журнала экспериментов."""
import csv
import io
import json

from app.core.db import connect

COLUMNS = ["id", "ts", "modality", "sample_a", "sample_b", "similarity",
           "threshold", "decision", "n_a", "n_b", "matched", "params"]


def add_experiment(modality: str, sample_a: str, sample_b: str, params: dict,
                   similarity: float, threshold: float, decision: str,
                   n_a: int, n_b: int, matched: int) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO experiments(modality, sample_a, sample_b, params, similarity,"
            " threshold, decision, n_a, n_b, matched) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (modality, sample_a, sample_b, json.dumps(params), similarity,
             threshold, decision, n_a, n_b, matched))


def list_experiments(limit: int = 100) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def export_csv() -> str:
    rows = list_experiments(10_000)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    for r in reversed(rows):
        w.writerow({k: r.get(k) for k in COLUMNS})
    return buf.getvalue()


def clear_experiments() -> None:
    with connect() as c:
        c.execute("DELETE FROM experiments")
