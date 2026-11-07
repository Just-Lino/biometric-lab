"""Журнал запусков обучения: настройки, итоги и кривые каждого запуска (для сравнения запусков друг с другом)."""
import json

from app.core.db import connect
from app.core.session import owner


def save(modality: str, config: dict, summary: dict, curves: dict) -> int:
    with connect() as c:
        return c.insert("INSERT INTO training_runs(modality, config, summary, curves, session_id, subject) VALUES (?,?,?,?,?,?)",
                        (modality, json.dumps(config, ensure_ascii=False), json.dumps(summary, ensure_ascii=False), json.dumps(curves), *owner()))


def list_runs(limit: int = 40) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r["id"], "ts": r["ts"], "modality": r["modality"], "config": json.loads(r["config"]),
             "summary": json.loads(r["summary"]), "curves": json.loads(r["curves"])} for r in rows]


def clear() -> None:
    with connect() as c:
        c.execute("DELETE FROM training_runs")
