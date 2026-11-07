"""Запуск кода студента в отдельном процессе с ограничениями (время, память на Linux/macOS, чистое окружение)."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from app.core.config import settings
from app.sandbox.safety import SafetyError, check_code

RUNNER = Path(__file__).with_name("runner.py")
MARK = "@@RESULT@@"
QUEUE_WAIT = 3.0  # секунд: столько запрос ждёт свободного места, затем получает «сервер занят» (а не стоит в очереди минутами)


class SandboxBusy(Exception):
    """Все места для запуска кода заняты: студенту нужно повторить через несколько секунд."""


class _Slots:
    """Места для одновременных запусков. Их число (SANDBOX_MAX_CONCURRENT) читается при каждом обращении, поэтому
    изменение настройки в тестах и при переконфигурации действует сразу. Счётчик в памяти процесса: у каждого экземпляра сервера свой."""

    def __init__(self):
        self._cond = threading.Condition()
        self._active = 0

    def acquire(self, wait: float) -> bool:
        deadline = time.monotonic() + wait
        with self._cond:
            while self._active >= max(1, settings.sandbox_max_concurrent):
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
            self._active += 1
            return True

    def release(self) -> None:
        with self._cond:
            self._active -= 1
            self._cond.notify()

    @property
    def active(self) -> int:
        with self._cond:
            return self._active


_slots = _Slots()


def _limits():  # только POSIX
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024 ** 3, 3 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


def _blocked(error_type: str, message: str, line=None) -> dict:
    return {"error": {"type": error_type, "message": message, "line": line}, "stdout": "", "defined": False,
            "tests": [], "example": None, "time": 0.0}


def execute(spec: str, code: str, mode: str = "grade", timeout: float = 20.0) -> dict:
    """mode: example (один видимый пример) или grade (все тесты)."""
    if len(code) > 20000:
        return _blocked("Blocked", "Код слишком длинный (максимум 20000 символов).")
    try:
        check_code(code)
    except SafetyError as e:
        return _blocked("Blocked", str(e), e.line)
    except SyntaxError as e:
        return _blocked("SyntaxError", e.msg, e.lineno)
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ}
    env.update(PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="0")
    # режим -I не видит пакеты из пользовательской папки (pip install --user, AppData на Windows):
    # передаём дочернему процессу пути поиска пакетов родителя
    env["LAB_SYS_PATH"] = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
    if not _slots.acquire(QUEUE_WAIT):
        raise SandboxBusy("Сервер занят: сейчас проверяется много решений. Повторите через несколько секунд.")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                p = subprocess.run([sys.executable, "-I", "-X", "utf8", str(RUNNER)],
                                   input=json.dumps({"spec": spec, "code": code, "mode": mode}),
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=tmp, env=env,
                                   preexec_fn=_limits if os.name == "posix" else None)
            except subprocess.TimeoutExpired:
                return _blocked("Timeout", f"Превышено время выполнения ({timeout:g} с): возможно, бесконечный цикл "
                                           "или слишком медленный алгоритм. Попробуйте векторизацию numpy.")
    finally:
        _slots.release()
    for line in reversed((p.stdout or "").splitlines()):
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    tail = (p.stderr or "").strip().splitlines()[-1:] or ["процесс завершился без результата"]
    return _blocked("RunnerError", f"Не удалось выполнить код: {tail[0][:200]}")
