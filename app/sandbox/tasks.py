"""Загрузка заданий из app/tasks/*.json (машиночитаемые: преподаватель может менять тексты, подсказки, веса рубрики)."""
import json
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


def load_tasks() -> dict[str, dict]:
    out = {}
    for f in sorted(TASKS_DIR.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        out[t["id"]] = t
    return out


def get_task(task_id: str) -> dict:
    tasks = load_tasks()
    if task_id not in tasks:
        raise KeyError(f"Неизвестное задание: {task_id}")
    return tasks[task_id]


def public_task(t: dict) -> dict:
    """Версия для студента: без решения, подсказок и правильного ответа."""
    return {"id": t["id"], "title": t["title"], "topic": t["topic"], "difficulty": t["difficulty"],
            "theory": t["theory"], "description": t["description"], "starter_code": t["starter_code"],
            "question": {"text": t["question"]["text"], "options": t["question"]["options"]},
            "rubric": t["rubric"], "max_score": sum(t["rubric"].values())}
