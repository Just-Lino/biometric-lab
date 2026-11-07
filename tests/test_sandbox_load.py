"""Песочница под нагрузкой: настраиваемое число запусков, «сервер занят» вместо ожидания, изоляция студентов друг от друга."""
import os
import tempfile
import threading
import time

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.sandbox import executor  # noqa: E402
from app.sandbox.executor import SandboxBusy, execute  # noqa: E402
from app.sandbox.tasks import get_task  # noqa: E402
from app.services import attempts  # noqa: E402

TASK = get_task("binarize")
LOOP = "def binarize(gray, threshold):\n    while True:\n        pass\n"       # занимает место, пока не сработает ограничение времени


@pytest.fixture(autouse=True)
def fast_busy(monkeypatch):
    monkeypatch.setattr(executor, "QUEUE_WAIT", 0.2)   # в тестах не ждём три секунды перед отказом
    yield
    deadline = time.time() + 30                         # запуски, начатые тестом, должны успеть закончиться
    while executor._slots.active and time.time() < deadline:
        time.sleep(0.1)


def hold(n, timeout=10):
    """Занять n мест запусками бесконечного цикла; возвращает потоки и список результатов."""
    results, threads = [], []
    for _ in range(n):
        t = threading.Thread(target=lambda: results.append(execute(TASK["spec"], LOOP, "example", timeout=timeout)), daemon=True)
        t.start()
        threads.append(t)
    deadline = time.time() + 15
    while executor._slots.active < n and time.time() < deadline:
        time.sleep(0.02)
    assert executor._slots.active == n
    return threads, results


def stopped(res):
    """Результат запуска, где код был остановлен: ошибка верхнего уровня (например, Timeout) или проваленный пример."""
    return res["error"] is not None or (res["example"] is not None and not res["example"]["ok"])


def finish(threads):
    for t in threads:
        t.join(40)
        assert not t.is_alive()


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_concurrency_is_configurable_and_excess_gets_busy_quickly(monkeypatch, limit):
    monkeypatch.setattr(settings, "sandbox_max_concurrent", limit)
    threads, results = hold(limit)
    t0 = time.time()
    with pytest.raises(SandboxBusy, match="Сервер занят"):
        execute(TASK["spec"], TASK["solution"], "example")
    assert time.time() - t0 < 2                          # отказ почти сразу, а не после очереди в несколько секунд
    finish(threads)
    assert executor._slots.active == 0
    res = execute(TASK["spec"], TASK["solution"], "example")      # места освободились: следующий запуск проходит
    assert res["example"]["ok"]
    assert len(results) == limit and all(stopped(r) for r in results)     # бесконечные циклы остановлены ограничением времени


def test_infinite_loop_of_one_student_does_not_stop_another(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_max_concurrent", 2)
    threads, results = hold(1)
    res = execute(TASK["spec"], TASK["solution"], "grade")        # второй студент в это время проверяет верное решение
    assert res["error"] is None and all(t["ok"] for t in res["tests"])
    finish(threads)
    assert stopped(results[0])                                   # цикл первого студента остановлен ограничением времени


def test_blocked_code_gets_instant_feedback_even_when_all_places_are_busy(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_max_concurrent", 1)
    threads, _ = hold(1)
    t0 = time.time()
    res = execute(TASK["spec"], "import os\n", "example")         # проверка безопасности идёт до очереди и места не занимает
    assert res["error"]["type"] == "Blocked" and time.time() - t0 < 1
    assert execute(TASK["spec"], "def binarize(g, t)\n  return g\n", "example")["error"]["type"] == "SyntaxError"
    finish(threads)


def test_place_is_released_after_timeout_and_after_unexpected_error(monkeypatch):
    res = execute(TASK["spec"], LOOP, "example", timeout=0.5)     # процесс не успевает даже запуститься: сработает Timeout
    assert res["error"]["type"] == "Timeout" and executor._slots.active == 0

    def boom(*a, **k):
        raise OSError("не удалось запустить процесс")
    monkeypatch.setattr(executor.subprocess, "run", boom)
    with pytest.raises(OSError):
        execute(TASK["spec"], TASK["solution"], "example")
    assert executor._slots.active == 0                            # место не «утекает» при сбое


@pytest.mark.parametrize("code", [
    "def binarize(g, t):\n    open('x.txt', 'w').write('1')\n    return g\n",
    "import numpy as np\ndef binarize(g, t):\n    np.save('x.npy', g)\n    return g\n",
    "import numpy as np\ndef binarize(g, t):\n    np.memmap('x.bin', mode='w+', shape=(10,))\n    return g\n",
    "import cv2\ndef binarize(g, t):\n    cv2.imwrite('x.png', g)\n    return g\n",
    "import os\ndef binarize(g, t):\n    os.remove('x')\n    return g\n",
])
def test_file_writes_are_blocked_and_leave_no_trace(code):
    before = sorted(os.listdir("."))
    res = execute(TASK["spec"], code, "example")
    assert res["error"]["type"] == "Blocked"
    assert sorted(os.listdir(".")) == before                       # в рабочей папке сервера ничего не появилось


@pytest.mark.skipif(os.name != "posix", reason="лимит памяти песочницы работает только на Linux и macOS (на Windows его нет)")
def test_memory_hog_is_stopped_and_does_not_affect_the_next_run():
    hog = "import numpy as np\ndef binarize(gray, threshold):\n    x = np.ones((80000, 80000))\n    return gray\n"   # около 51 ГБ при лимите 3 ГБ
    res = execute(TASK["spec"], hog, "example")
    assert res["error"] or (res["example"] and not res["example"]["ok"])
    assert "allocate" in str(res) or "Memory" in str(res)
    assert execute(TASK["spec"], TASK["solution"], "example")["example"]["ok"]


def test_api_returns_503_when_busy_and_does_not_count_an_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "sandbox_max_concurrent", 1)
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "own.db"))   # своя база: чужие тесты считают попытки и не должны видеть эти
    monkeypatch.setattr(settings, "database_url", None)
    with TestClient(app) as c:
        before = attempts.count("binarize")
        threads, _ = hold(1)
        for path in ("run", "submit"):
            r = c.post(f"/api/v1/code/{path}", json={"task_id": "binarize", "code": TASK["solution"]})
            assert r.status_code == 503 and "Сервер занят" in r.json()["detail"] and r.headers["Retry-After"] == "5"
        assert attempts.count("binarize") == before                 # отказ из-за нагрузки не съедает попытку студента
        finish(threads)
        r = c.post("/api/v1/code/submit", json={"task_id": "binarize", "code": TASK["solution"], "answer": TASK["question"]["correct"]})
        assert r.status_code == 200 and r.json()["grading"]["score"] == r.json()["grading"]["max_score"]
        assert attempts.count("binarize") == before + 1
