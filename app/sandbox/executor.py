"""Запуск кода студента в отдельном процессе с ограничениями (время, память на Linux/macOS, чистое окружение)."""
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from app.sandbox.safety import SafetyError, check_code

RUNNER = Path(__file__).with_name("runner.py")
MARK = "@@RESULT@@"
_slots = threading.BoundedSemaphore(2)  # не больше двух одновременных запусков


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
    with _slots, tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-X", "utf8", str(RUNNER)],
                               input=json.dumps({"spec": spec, "code": code, "mode": mode}),
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=tmp, env=env,
                               preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return _blocked("Timeout", f"Превышено время выполнения ({timeout:g} с): возможно, бесконечный цикл "
                                       "или слишком медленный алгоритм. Попробуйте векторизацию numpy.")
    for line in reversed((p.stdout or "").splitlines()):
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    tail = (p.stderr or "").strip().splitlines()[-1:] or ["процесс завершился без результата"]
    return _blocked("RunnerError", f"Не удалось выполнить код: {tail[0][:200]}")
