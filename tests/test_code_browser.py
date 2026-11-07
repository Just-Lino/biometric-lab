"""Запуск примера практикума в браузере: эндпоинты example и feedback, совпадение браузерного раннера с серверным /run,
отсутствие секретов в ответах клиенту, раздача файлов Pyodide. Сама оценка (/submit) не меняется."""
import importlib.util
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "code.db"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.sandbox import safety  # noqa: E402
from app.sandbox.executor import execute  # noqa: E402
from app.sandbox.reference import CMP_DESC, SPECS, get_spec  # noqa: E402
from app.sandbox.serialize import decode, encode  # noqa: E402
from app.sandbox.tasks import get_task, load_tasks  # noqa: E402
from app.services import code_tutor  # noqa: E402
from tests.test_code_tasks import WRONG  # noqa: E402

CODELAB = Path(__file__).resolve().parents[1] / "app" / "web" / "codelab"
VENDOR = Path(__file__).resolve().parents[1] / "app" / "web" / "vendor" / "pyodide"
TASKS = load_tasks()

_spec = importlib.util.spec_from_file_location("lab_runner_under_test", CODELAB / "lab_runner.py")
lab_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab_runner)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def browser_run(client, task_id: str, code: str) -> dict:
    """То же, что делает страница: берёт пример с сервера и выполняет раннер (здесь в обычном CPython)."""
    ex = client.get(f"/api/v1/code/tasks/{task_id}/example").json()
    payload = json.dumps({"code": code, "function": ex["function"], "args": ex["args"], "expected_value": ex["expected_value"],
                          "cmp": ex["cmp"], "input": ex["input"], "expected": ex["expected"]})
    return json.loads(lab_runner.run_example(payload))


# ───────────────────────── эндпоинт example ─────────────────────────
@pytest.mark.parametrize("tid", list(TASKS))
def test_example_endpoint_returns_only_the_public_example(client, tid):
    r = client.get(f"/api/v1/code/tasks/{tid}/example")
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"task_id", "function", "args", "input", "expected", "expected_value", "cmp"}
    t, sp = get_task(tid), get_spec(TASKS[tid]["spec"])
    assert d["function"] == sp.func and d["cmp"] == CMP_DESC[t["spec"]]
    # вход и ожидаемое значение - ровно пример, видимый в /run; значения раскодируются в исходные
    srv = execute(t["spec"], t["starter_code"], "example")["example"]
    assert d["input"] == srv["input"] and d["expected"] == srv["expected"]
    dec = decode(d["args"])
    assert len(dec) == len(sp.example.args)
    for a, b in zip(dec, sp.example.args):
        assert np.array_equal(np.asarray(a), np.asarray(b)) if isinstance(b, np.ndarray) else a == b
    text = json.dumps(d, ensure_ascii=False)
    assert t["solution"].strip() not in text
    for c in sp.basic + sp.edge:                                    # названия граничных и базовых тестов не раскрываются
        assert c.name not in text


def test_example_endpoint_unknown_task_is_404(client):
    assert client.get("/api/v1/code/tasks/nope/example").status_code == 404


def test_serializer_roundtrips_arrays_tuples_and_scalars():
    value = [np.array([[1, 2], [3, 4]], np.uint8), (0.5, "MATCH"), [1, 2, 3], None, True, np.float64(2.5), np.int64(7)]
    back = decode(encode(value))
    assert np.array_equal(back[0], value[0]) and back[0].dtype == np.uint8
    assert back[1] == (0.5, "MATCH") and back[2:5] == [[1, 2, 3], None, True] and back[5] == 2.5 and back[6] == 7
    with pytest.raises(TypeError):
        encode({1, 2})


# ───────────────────────── эндпоинт feedback ─────────────────────────
def test_feedback_matches_server_feedback_for_the_same_result(client):
    for tid in TASKS:
        t = get_task(tid)
        for code in (t["starter_code"], t["solution"], "def x(:\n  pass"):
            res = execute(t["spec"], code, "example")
            body = {"task_id": tid, "result": {k: res[k] for k in ("error", "example", "stdout", "defined", "time")}}
            r = client.post("/api/v1/code/feedback", json=body)
            assert r.status_code == 200
            assert r.json()["feedback"] == code_tutor.feedback(t, res)


@pytest.mark.parametrize("bad_type", ["{msg.__class__}", "Err or", "x" * 80, "", "1abc", "a-b"])
def test_feedback_rejects_unsafe_error_type(client, bad_type):
    body = {"task_id": "binarize", "result": {"error": {"type": bad_type, "message": "m", "line": 1}}}
    assert client.post("/api/v1/code/feedback", json=body).status_code == 422


def test_feedback_validates_sizes_and_task(client):
    big = {"task_id": "binarize", "result": {"example": {"ok": False, "message": "я" * 5000}}}
    assert client.post("/api/v1/code/feedback", json=big).status_code == 422
    assert client.post("/api/v1/code/feedback", json={"task_id": "nope", "result": {}}).status_code == 404
    assert client.post("/api/v1/code/feedback", json={"task_id": "binarize"}).status_code == 422


def test_feedback_does_not_touch_attempts_and_logs_event(client):
    from app.core.db import connect
    before = client.get("/api/v1/code/attempts/binarize").json()
    client.post("/api/v1/code/feedback", json={"task_id": "binarize", "result": {"example": {"ok": True, "input": "i", "expected": "e"}}})
    assert client.get("/api/v1/code/attempts/binarize").json() == before            # попытки и баллы не меняются
    with connect() as c:
        rows = c.execute("SELECT payload FROM events WHERE type='code_run' ORDER BY id DESC LIMIT 5").fetchall()
    assert any(json.loads(r[0]).get("where") == "browser" for r in rows)


def test_submit_still_grades_on_the_server_and_ignores_browser_data(monkeypatch, tmp_path):
    from app.core.config import settings
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "own.db"))   # своя база: чужие тесты считают попытки и не должны видеть эти
    monkeypatch.setattr(settings, "database_url", None)
    t = get_task("binarize")
    with TestClient(app) as c:
        r = c.post("/api/v1/code/submit", json={"task_id": "binarize", "code": t["solution"], "answer": t["question"]["correct"],
                                                 "browser_result": {"example": {"ok": True}}}).json()     # лишние поля игнорируются
        assert r["grading"]["score"] == 10.0 and r["attempts"] == 1 and len(r["tests"]) >= 8
        wrong = c.post("/api/v1/code/submit", json={"task_id": "binarize", "code": t["starter_code"], "answer": 0}).json()
        assert wrong["grading"]["score"] < 10.0 and wrong["attempts"] == 2


# ───────────────────────── браузерный раннер совпадает с серверным /run ─────────────────────────
EXTRA = [
    ("синтаксическая ошибка", "def {f}(:\n    pass\n"),
    ("запрещённый импорт", "import os\ndef {f}(*a):\n    return 1\n"),
    ("функции нет", "x = 1\n"),
    ("NameError", "def {f}(*a):\n    return undefined_name\n"),
    ("print и возврат None", "def {f}(*a):\n    print('отладка')\n"),
    ("исключение в функции", "def {f}(*a):\n    raise ValueError('плохо')\n"),
    ("ошибка на уровне модуля", "raise RuntimeError('при импорте')\n"),
]


def parity_cases():
    out = []
    for tid, t in TASKS.items():
        f = t["function"]
        out.append((tid, "эталонное решение", t["solution"]))
        out.append((tid, "заготовка", t["starter_code"]))
        out += [(w[0], w[1], w[2]) for w in WRONG if w[0] == tid]
        out += [(tid, name, code.format(f=f)) for name, code in EXTRA]
    return out


PARITY = parity_cases()


@pytest.mark.parametrize("tid,why,code", PARITY, ids=[f"{p[0]}-{p[1]}" for p in PARITY])
def test_browser_runner_matches_server_run(client, tid, why, code):
    srv = execute(get_task(tid)["spec"], code, "example")
    br = browser_run(client, tid, code)
    for key in ("error", "defined", "example", "stdout", "tests"):
        assert br[key] == srv[key], f"{key}: браузер {br[key]!r} != сервер {srv[key]!r}"


def test_output_is_capped_like_on_the_server(client):
    code = "def binarize(g, t):\n    print('x' * 30000)\n    return g\n"
    srv = execute("binarize", code, "example")
    br = browser_run(client, "binarize", code)
    assert len(br["stdout"]) == len(srv["stdout"]) and 20000 <= len(br["stdout"]) <= 30000


@pytest.mark.parametrize("name", list(SPECS))
def test_comparators_match_reference_comparators(name):
    sp = get_spec(name)
    exp = sp.ref(*sp.example.args)
    cands = [None, 0, 1, 1.5, -1, "x", [1, 2], (1, 2), (0.1, "MATCH"), np.zeros(3), np.zeros((2, 2), np.uint8), exp, True]
    if isinstance(exp, np.ndarray):
        cands += [exp + 1, exp.astype(float), exp.astype(np.uint8), exp.reshape(-1), exp[:1], exp * 0, exp.astype(np.int64)]
    elif isinstance(exp, tuple):
        cands += [(exp[0] + 1, exp[1]), (exp[0], exp[1] if isinstance(exp[1], str) else exp[1] + 1), exp[:1], exp[::-1], list(exp)]
    else:
        cands += [exp + 1, float(exp), exp - 0.5, exp + 1e-12, np.float64(exp), -exp]
    for got in cands:
        try:
            want = sp.cmp(exp, got)
        except Exception as e:                                                # эталонная проверка падает - раннер тоже должен
            with pytest.raises(type(e)):
                lab_runner.compare(CMP_DESC[name], exp, got)
            continue
        assert lab_runner.compare(CMP_DESC[name], exp, got) == want, (name, got)


# ───────────────────────── секреты и раздача файлов ─────────────────────────
def test_browser_files_do_not_contain_solutions_or_reference_code():
    texts = {p.name: p.read_text(encoding="utf-8") for p in CODELAB.iterdir() if p.is_file()}
    assert {"worker.js", "lab_runner.py"} <= set(texts)
    for name, text in texts.items():
        assert "app.sandbox.reference" not in text and "SPECS" not in text, name
        for t in TASKS.values():
            assert t["solution"].strip() not in text, name
            for h in t["hints"]:
                assert h not in text


def test_safety_module_is_served_unchanged(client):
    r = client.get("/codelab/safety.py")
    assert r.status_code == 200 and r.text == Path(safety.__file__).read_text(encoding="utf-8")
    assert "text/x-python" in r.headers["content-type"]


def test_worker_and_runner_are_served_with_correct_types(client):
    w = client.get("/codelab/worker.js")
    assert w.status_code == 200 and "javascript" in w.headers["content-type"] and "loadPyodide" in w.text and "pyodide.mjs" in w.text
    assert client.get("/codelab/lab_runner.py").status_code == 200


def test_no_client_response_leaks_solutions_hints_or_answers(client):
    blobs = [client.get("/api/v1/code/tasks").text]
    for tid, t in TASKS.items():
        blobs += [client.get(f"/api/v1/code/tasks/{tid}").text, client.get(f"/api/v1/code/tasks/{tid}/example").text,
                  client.post("/api/v1/code/run", json={"task_id": tid, "code": t["starter_code"]}).text,
                  client.post("/api/v1/code/feedback", json={"task_id": tid, "result": {"example": {"ok": False, "message": "m"}}}).text]
        text = "\n".join(blobs[-4:])
        assert '"solution"' not in text and '"hints"' not in text and '"correct"' not in text
        assert t["solution"].strip() not in text
        for h in t["hints"]:
            assert h not in text


@pytest.mark.skipif(not (VENDOR / "pyodide.asm.wasm").exists(), reason="Pyodide не скачан: python -m scripts.download_pyodide")
def test_pyodide_files_have_right_mime_types_and_are_not_recompressed(client):
    h = {"Accept-Encoding": "gzip"}
    wasm = client.get("/vendor/pyodide/pyodide.asm.wasm", headers=h)
    assert wasm.status_code == 200 and wasm.headers["content-type"] == "application/wasm"
    assert "content-encoding" not in wasm.headers and "max-age" in wasm.headers["cache-control"]
    js = client.get("/vendor/pyodide/pyodide.mjs", headers=h)
    assert "javascript" in js.headers["content-type"]
    lock = json.loads(client.get("/vendor/pyodide/pyodide-lock.json").text)
    wheels = [p["file_name"] for k, p in lock["packages"].items() if k in ("numpy", "scipy", "scikit-image")]
    for w in wheels:
        r = client.head(f"/vendor/pyodide/{w}")
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip", w


def test_pyodide_version_is_pinned_in_script_and_readme():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "download_pyodide.py").read_text(encoding="utf-8")
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    import re
    ver = re.search(r'PYODIDE_VERSION = "([0-9.]+)"', script).group(1)
    assert ver in readme, "версия Pyodide должна быть указана в README"


def test_import_whitelist_is_the_server_one():
    """Браузер использует тот же safety.py, поэтому список разрешённых импортов совпадает автоматически."""
    assert lab_runner.check_code is safety.check_code or lab_runner.check_code.__module__ in ("safety", "app.sandbox.safety")
    assert "numpy" in safety.ALLOWED_IMPORTS and "os" not in safety.ALLOWED_IMPORTS
