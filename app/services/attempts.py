"""Репозиторий попыток студента (основа для Learning Analytics)."""
import json

from app.core.db import connect, json_field
from app.core.session import owner


def add_attempt(task_id: str, code: str, g: dict, details: dict) -> int:
    with connect() as c:
        return c.insert(
            "INSERT INTO attempts(task_id, score, max_score, passed_basic, total_basic, passed_edge, total_edge, run_ok, answer_ok, code, details, session_id, subject)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (task_id, g["score"], g["max_score"], g["passed_basic"], g["total_basic"], g["passed_edge"], g["total_edge"],
             int(g["run_ok"]), int(g["answer_ok"]), code, json.dumps(details, ensure_ascii=False), *owner()))


def count(task_id: str) -> int:
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM attempts WHERE task_id=?", (task_id,)).fetchone()[0]


def count_valid(task_id: str) -> int:
    """Попытки, в которых код дошёл до тестов (не заблокирован песочницей и без синтаксической ошибки): по ним открывается полное решение."""
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM attempts WHERE task_id=? AND total_basic>0", (task_id,)).fetchone()[0]


def history(task_id: str, limit: int = 20) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT id, ts, score, max_score, passed_basic, total_basic, passed_edge, total_edge, run_ok, answer_ok "
                         "FROM attempts WHERE task_id=? ORDER BY id DESC LIMIT ?", (task_id, limit)).fetchall()
    return [dict(r) for r in rows]


def last_code(task_id: str) -> str | None:
    with connect() as c:
        r = c.execute("SELECT code FROM attempts WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    return r["code"] if r else None


def summary() -> dict[str, dict]:
    with connect() as c:
        rows = c.execute("SELECT task_id, COUNT(*) AS n, MAX(score) AS best FROM attempts GROUP BY task_id").fetchall()
        hints = c.execute(f"SELECT {json_field('payload', 'task')} AS t, COUNT(*) AS n FROM events WHERE type='hint' GROUP BY 1").fetchall()
    out = {r["task_id"]: {"attempts": r["n"], "best": r["best"], "hints": 0} for r in rows}
    for h in hints:
        if h["t"]:
            out.setdefault(h["t"], {"attempts": 0, "best": None, "hints": 0})["hints"] = h["n"]
    return out
