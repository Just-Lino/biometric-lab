from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.sandbox.executor import execute
from app.sandbox.tasks import get_task, load_tasks, public_task
from app.services import attempts, code_tutor
from app.services.events import record_event
from app.services.grader import grade

router = APIRouter(prefix="/api/v1/code", tags=["code"])


class RunIn(BaseModel):
    task_id: str
    code: str = Field(max_length=20000)


class SubmitIn(RunIn):
    answer: int | None = None


class HintIn(BaseModel):
    task_id: str
    level: int = Field(ge=1, le=4)


def _task(task_id: str) -> dict:
    try:
        return get_task(task_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.get("/tasks")
def tasks():
    summ = attempts.summary()
    out = []
    for t in load_tasks().values():
        p = public_task(t)
        s = summ.get(t["id"], {})
        out.append({k: p[k] for k in ("id", "title", "topic", "difficulty", "max_score")} |
                   {"attempts": s.get("attempts", 0), "best": s.get("best"), "hints": s.get("hints", 0)})
    return out


@router.get("/tasks/{task_id}")
def task(task_id: str):
    p = public_task(_task(task_id))
    p["attempts"] = attempts.count(task_id)
    p["last_code"] = attempts.last_code(task_id)
    return p


@router.post("/run")
def run(body: RunIn):
    t = _task(body.task_id)
    res = execute(t["spec"], body.code, "example")
    res["feedback"] = code_tutor.feedback(t, res)
    record_event("code_run", {"task": body.task_id, "ok": bool(res.get("example") and res["example"]["ok"])})
    return res


@router.post("/submit")
def submit(body: SubmitIn):
    t = _task(body.task_id)
    res = execute(t["spec"], body.code, "grade")
    g = grade(t, res, body.answer)
    res["grading"] = g
    res["feedback"] = code_tutor.feedback(t, res, g)
    res["attempt_id"] = attempts.add_attempt(body.task_id, body.code, g,
                                             {"error": res.get("error"), "tests": [{k: x[k] for k in ("name", "kind", "ok")} for x in res["tests"]]})
    res["attempts"] = attempts.count(body.task_id)
    record_event("code_submit", {"task": body.task_id, "score": g["score"]})
    return res


@router.post("/hint")
def hint(body: HintIn):
    t = _task(body.task_id)
    n = attempts.count_valid(body.task_id)
    ok, text = code_tutor.hint(t, body.level, n)
    if ok:
        record_event("hint", {"task": body.task_id, "level": body.level})
    return {"level": body.level, "title": code_tutor.LEVELS[body.level], "available": ok, "text": text, "attempts": n}


@router.get("/attempts/{task_id}")
def history(task_id: str):
    _task(task_id)
    return attempts.history(task_id)
