import os
import tempfile

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "code.db"))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sandbox.executor import execute
from app.sandbox.safety import SafetyError, check_code
from app.sandbox.tasks import get_task, load_tasks
from app.services.grader import grade

TASKS = list(load_tasks())


@pytest.mark.parametrize("tid", TASKS)
def test_reference_solution_gets_full_score(tid):
    t = get_task(tid)
    res = execute(t["spec"], t["solution"], "grade")
    g = grade(t, res, t["question"]["correct"])
    assert res["error"] is None
    assert g["score"] == g["max_score"] == 10.0 and g["run_ok"] and g["answer_ok"]


@pytest.mark.parametrize("tid", TASKS)
def test_starter_code_gets_zero(tid):
    t = get_task(tid)
    g = grade(t, execute(t["spec"], t["starter_code"], "grade"), None)
    assert g["score"] == 0  # пустой шаблон не должен получать баллы


def test_partial_credit_for_edge_case_bug():
    t = get_task("binarize")
    code = "import numpy as np\ndef binarize(g, t):\n    return np.where(g <= t, 0, 255).astype(np.uint8)\n"
    g = grade(t, execute(t["spec"], code, "grade"), None)
    assert g["run_ok"] and 0 < g["score"] < 8 and g["passed_edge"] < g["total_edge"]


@pytest.mark.parametrize("code", [
    "import os\ndef binarize(g, t):\n    return g",
    "import subprocess\ndef binarize(g, t):\n    return g",
    "def binarize(g, t):\n    return open('x').read()",
    "def binarize(g, t):\n    return ().__class__",
    "import numpy as np\ndef binarize(g, t):\n    return np.load('a.npy')",
    "import numpy.ctypeslib\ndef binarize(g, t):\n    return g",
    "def binarize(g, t):\n    return eval('1')",
])
def test_dangerous_code_is_blocked(code):
    with pytest.raises(SafetyError):
        check_code(code)
    res = execute("binarize", code, "grade")
    assert res["error"]["type"] == "Blocked" and res["tests"] == []


def test_syntax_error_reports_line():
    res = execute("binarize", "def binarize(g, t)\n    return g\n", "grade")
    assert res["error"]["type"] == "SyntaxError" and res["error"]["line"] == 1


def test_infinite_loop_times_out_quickly():
    code = "def binarize(g, t):\n    while True:\n        pass\n"
    res = execute("binarize", code, "grade", timeout=15)
    errs = [t for t in res["tests"] if t["error"]]
    assert (res["error"] and res["error"]["type"] == "Timeout") or (errs and errs[0]["error"]["type"] == "Timeout")


def test_student_prints_are_captured_and_input_not_mutated():
    code = "def binarize(g, t):\n    print('отладка')\n    g[:] = 0\n    return g\n"
    res = execute("binarize", code, "example")
    assert "отладка" in res["stdout"] and res["example"]["ok"] is False


def test_api_flow_and_hint_gating():
    with TestClient(app) as c:
        lst = c.get("/api/v1/code/tasks").json()
        assert {t["id"] for t in lst} == set(TASKS)
        t = c.get("/api/v1/code/tasks/binarize").json()
        assert "solution" not in t and "hints" not in t and "correct" not in t["question"]  # решение студенту не отдаётся
        assert t["max_score"] == 10 and "def binarize" in t["starter_code"]

        r = c.post("/api/v1/code/run", json={"task_id": "binarize", "code": t["starter_code"]}).json()
        assert r["example"]["ok"] is False and r["feedback"]

        # полное решение закрыто до трёх попыток
        h4 = c.post("/api/v1/code/hint", json={"task_id": "binarize", "level": 4}).json()
        assert h4["available"] is False and "np.where" not in h4["text"]
        h1 = c.post("/api/v1/code/hint", json={"task_id": "binarize", "level": 1}).json()
        assert h1["available"] is True and h1["title"] == "направление"

        for i in range(3):
            s = c.post("/api/v1/code/submit", json={"task_id": "binarize", "code": t["starter_code"], "answer": 1}).json()
            assert s["grading"]["score"] == 1.0 and s["attempts"] == i + 1  # только балл за вопрос
        h4 = c.post("/api/v1/code/hint", json={"task_id": "binarize", "level": 4}).json()
        assert h4["available"] is True and "np.where" in h4["text"]

        sol = get_task("binarize")["solution"]
        s = c.post("/api/v1/code/submit", json={"task_id": "binarize", "code": sol, "answer": 1}).json()
        assert s["grading"]["score"] == 10.0 and s["feedback"].startswith("Все тесты пройдены")
        assert c.get("/api/v1/code/tasks/binarize").json()["last_code"] == sol
        hist = c.get("/api/v1/code/attempts/binarize").json()
        assert [h["score"] for h in hist] == [10.0, 1.0, 1.0, 1.0]
        row = next(x for x in c.get("/api/v1/code/tasks").json() if x["id"] == "binarize")
        assert row["best"] == 10.0 and row["attempts"] == 4 and row["hints"] >= 2
        assert c.post("/api/v1/code/submit", json={"task_id": "nope", "code": "x"}).status_code == 404


def test_runner_output_is_ascii_only_regardless_of_console_encoding():
    """Регрессия для Windows: вывод раннера должен читаться при любой кодировке консоли (cp1251, ascii и т.п.)."""
    import json
    import subprocess
    import sys

    from app.sandbox.executor import MARK, RUNNER
    req = json.dumps({"spec": "binarize", "mode": "grade",
                      "code": "def binarize(g, t):\n    print('отладка ×')\n    return g\n"})
    env = {"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C",
           "LAB_SYS_PATH": os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))}
    for flags in (["-X", "utf8=0"], ["-X", "utf8"]):
        p = subprocess.run([sys.executable, "-I", *flags, str(RUNNER)], input=req.encode(), capture_output=True, env=env)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[-300:]
        line = [l for l in p.stdout.decode("ascii").splitlines() if l.startswith(MARK)][0]  # decode("ascii") не должен упасть
        assert r"\u043e\u0442\u043b\u0430\u0434\u043a\u0430" in line  # «отладка» в виде экранированных символов
