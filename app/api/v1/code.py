import copy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.sandbox.executor import SandboxBusy, execute
from app.sandbox.reference import CMP_DESC, get_spec, short
from app.sandbox.serialize import encode
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


class BrowserError(BaseModel):
    """Ошибка из браузерного запуска. Тип ограничен именем класса исключения: он подставляется в текст разбора."""
    type: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
    message: str = Field("", max_length=500)
    line: int | None = Field(None, ge=0, le=100000)


class BrowserExample(BaseModel):
    input: str = Field("", max_length=600)
    expected: str = Field("", max_length=600)
    ok: bool
    message: str = Field("", max_length=1000)
    error: BrowserError | None = None


class BrowserResult(BaseModel):
    error: BrowserError | None = None
    example: BrowserExample | None = None
    stdout: str = Field("", max_length=30000)
    defined: bool = False
    time: float = 0.0


class FeedbackIn(BaseModel):
    task_id: str
    result: BrowserResult


def _run_code(spec: str, code: str, mode: str) -> dict:
    """Запуск кода в песочнице; при перегрузке ответ 503 с подсказкой повторить (попытка при этом не засчитывается)."""
    try:
        return execute(spec, code, mode)
    except SandboxBusy as e:
        raise HTTPException(status_code=503, detail=str(e), headers={"Retry-After": "5"})


def _task(task_id: str) -> dict:
    try:
        return get_task(task_id)
    except KeyError as e:
        raise HTTPException(404, e.args[0] if e.args and isinstance(e.args[0], str) else str(e))  # без кавычек, которые KeyError добавляет сам


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


@router.get("/tasks/{task_id}/example")
def example(task_id: str):
    """Пример задания для запуска в браузере: вход и ожидаемый результат (то же, что студент видит в ответе /run).
    Базовые и граничные тесты, решение, подсказки и правильный ответ на вопрос здесь не отдаются."""
    t = _task(task_id)
    sp = get_spec(t["spec"])
    case = sp.example
    expected = sp.ref(*copy.deepcopy(case.args))
    return {"task_id": task_id, "function": sp.func, "args": encode(list(case.args)),
            "input": short(case.args if len(case.args) > 1 else case.args[0], 300),
            "expected": short(expected), "expected_value": encode(expected), "cmp": CMP_DESC[t["spec"]]}


@router.post("/feedback")
def browser_feedback(body: FeedbackIn):
    """Разбор результата запуска примера, выполненного в браузере. Баллы и попытки не затрагиваются: оценивает только /submit."""
    t = _task(body.task_id)
    res = body.result.model_dump()
    res["tests"] = []
    record_event("code_run", {"task": body.task_id, "ok": bool(res["example"] and res["example"]["ok"]), "where": "browser"})
    return {"feedback": code_tutor.feedback(t, res)}


@router.post("/run")
def run(body: RunIn):
    t = _task(body.task_id)
    res = _run_code(t["spec"], body.code, "example")
    res["feedback"] = code_tutor.feedback(t, res)
    record_event("code_run", {"task": body.task_id, "ok": bool(res.get("example") and res["example"]["ok"]), "where": "server"})
    return res


@router.post("/submit")
def submit(body: SubmitIn):
    t = _task(body.task_id)
    res = _run_code(t["spec"], body.code, "grade")
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
