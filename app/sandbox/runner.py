"""Процесс-раннер: выполняет код студента и тесты. Запускается отдельно: python -I runner.py (JSON в stdin)."""
import copy
import io
import json
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

for _p in os.environ.get("LAB_SYS_PATH", "").split(os.pathsep):  # пути пакетов родителя (см. executor.py)
    if _p and _p not in sys.path:
        sys.path.append(_p)

from app.sandbox.reference import get_spec, short  # noqa: E402
from app.sandbox.safety import SafetyError, check_code, safe_builtins  # noqa: E402

MARK = "@@RESULT@@"
CASE_SECONDS = 3


class _Timeout(BaseException):
    pass


class Cap(io.StringIO):
    def write(self, s):
        if self.tell() < 20000:
            return super().write(s)
        return 0


def _err(e: BaseException) -> dict:
    line = None
    tb = e.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_filename == "<student>":
            line = tb.tb_lineno
        tb = tb.tb_next
    if isinstance(e, SyntaxError):
        line = e.lineno
    return {"type": type(e).__name__, "message": str(e)[:300], "line": line}


# состояние для сторожевого таймера (Windows): что сейчас выполняется и что уже накоплено
STATE: dict = {"out": None, "buf": None, "t0": 0.0, "cases": [], "i": -1, "mode": "grade", "spec": None, "timer": None}


def _watchdog_fire():
    """Windows: сигналов SIGALRM нет, поэтому по таймеру в отдельном потоке пишем итог (текущий тест = Timeout, остальные пропущены) и завершаем процесс."""
    out, cases, i, spec = STATE["out"], STATE["cases"], STATE["i"], STATE["spec"]
    msg = f"тест выполнялся дольше {CASE_SECONDS} с"
    err = {"type": "Timeout", "message": msg, "line": None}
    if 0 <= i < len(cases):
        if STATE["mode"] == "example":
            c = cases[i]
            out["example"] = {"input": short(c.args if len(c.args) > 1 else c.args[0], 300), "expected": short(spec.ref(*copy.deepcopy(c.args))),
                              "ok": False, "message": msg, "error": err}
        else:
            for k, c in enumerate(cases[i:]):
                out["tests"].append({"name": c.name, "kind": c.kind, "ok": False, "returned_none": False,
                                     "error": err if k == 0 else None, "message": msg if k == 0 else "пропущен: ранее превышено время выполнения"})
    else:
        out["error"] = {"type": "Timeout", "message": "код выполнялся слишком долго (бесконечный цикл?)", "line": None}
    out["stdout"] = STATE["buf"].getvalue()
    out["time"] = round(time.time() - STATE["t0"], 3)
    sys.__stdout__.write(MARK + json.dumps(out, default=str) + "\n")
    sys.__stdout__.flush()
    import os
    os._exit(0)


def _arm():
    if hasattr(signal, "setitimer"):
        def h(*_):
            raise _Timeout()
        signal.signal(signal.SIGALRM, h)
        signal.setitimer(signal.ITIMER_REAL, CASE_SECONDS)
    else:
        t = threading.Timer(CASE_SECONDS, _watchdog_fire)
        t.daemon = True
        STATE["timer"] = t
        t.start()


def _disarm():
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, 0)
    elif STATE["timer"] is not None:
        STATE["timer"].cancel()
        STATE["timer"] = None


def run_case(fn, spec, case, reveal: bool) -> dict:
    exp_args = copy.deepcopy(case.args)
    expected = spec.ref(*exp_args)
    args = copy.deepcopy(case.args)
    out = {"name": case.name, "kind": case.kind, "ok": False, "message": "", "error": None, "returned_none": False}
    try:
        _arm()
        got = fn(*args)
        _disarm()
    except _Timeout:
        out["error"] = {"type": "Timeout", "message": f"тест выполнялся дольше {CASE_SECONDS} с", "line": None}
        out["message"] = out["error"]["message"]
        return out
    except BaseException as e:
        _disarm()
        out["error"] = _err(e)
        out["message"] = f"{out['error']['type']}: {out['error']['message']}"
        return out
    out["returned_none"] = got is None
    ok, msg = spec.cmp(expected, got)
    out["ok"] = bool(ok)
    if not ok:
        # для граничных случаев ожидаемое значение не раскрываем
        out["message"] = msg if reveal else "результат не совпал с эталоном для этого случая"
        if reveal:
            out["input"] = short(case.args if len(case.args) > 1 else case.args[0], 200)
    return out


def main():
    req = json.loads(sys.stdin.read())
    spec = get_spec(req["spec"])
    out = {"error": None, "stdout": "", "defined": False, "tests": [], "example": None, "time": 0.0}
    code = req["code"]
    t0 = time.time()
    buf = Cap()
    STATE.update(out=out, buf=buf, t0=t0, spec=spec, mode=req["mode"], cases=[spec.example] if req["mode"] == "example" else list(spec.basic) + list(spec.edge), i=-1)
    try:
        check_code(code)
        ns = {"__builtins__": safe_builtins(), "__name__": "student"}
        old = sys.stdout
        sys.stdout = buf
        try:
            _arm()
            exec(compile(code, "<student>", "exec"), ns)
            _disarm()
        finally:
            sys.stdout = old
        fn = ns.get(spec.func)
        out["defined"] = callable(fn)
        if not out["defined"]:
            out["error"] = {"type": "NoFunction", "message": f"Не найдена функция {spec.func}", "line": None}
        else:
            sys.stdout = buf
            try:
                if req["mode"] == "example":
                    STATE["i"] = 0
                    r = run_case(fn, spec, spec.example, True)
                    exp = spec.ref(*copy.deepcopy(spec.example.args))
                    out["example"] = {"input": short(spec.example.args if len(spec.example.args) > 1 else spec.example.args[0], 300),
                                      "expected": short(exp), "ok": r["ok"], "message": r["message"], "error": r["error"]}
                else:
                    timed_out = False
                    idx = -1
                    for cases, reveal in ((spec.basic, True), (spec.edge, False)):
                        for case in cases:
                            idx += 1
                            STATE["i"] = idx
                            if timed_out:  # после таймаута остальные тесты не гоняем, чтобы не ждать
                                out["tests"].append({"name": case.name, "kind": case.kind, "ok": False, "error": None,
                                                     "returned_none": False, "message": "пропущен: ранее превышено время выполнения"})
                                continue
                            r = run_case(fn, spec, case, reveal)
                            timed_out = bool(r["error"] and r["error"]["type"] == "Timeout")
                            out["tests"].append(r)
            finally:
                sys.stdout = old
    except SafetyError as e:
        out["error"] = {"type": "Blocked", "message": str(e), "line": e.line}
    except SyntaxError as e:
        out["error"] = {"type": "SyntaxError", "message": e.msg, "line": e.lineno}
    except _Timeout:
        out["error"] = {"type": "Timeout", "message": "код выполнялся слишком долго (бесконечный цикл?)", "line": None}
    except BaseException as e:
        out["error"] = _err(e)
    finally:
        _disarm()
    out["stdout"] = buf.getvalue()
    out["time"] = round(time.time() - t0, 3)
    # только ASCII: кириллица экранируется (\uXXXX), поэтому вывод не зависит от кодировки консоли (Windows: cp1251/cp866)
    print(MARK + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
