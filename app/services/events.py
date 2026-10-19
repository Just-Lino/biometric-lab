"""Журнал действий студента (основа для Learning Analytics)."""
import json

from app.core.db import connect


def record_event(event_type: str, payload: dict) -> None:
    with connect() as c:
        c.execute("INSERT INTO events(type, payload) VALUES (?, ?)",
                  (event_type, json.dumps(payload, ensure_ascii=False)))
