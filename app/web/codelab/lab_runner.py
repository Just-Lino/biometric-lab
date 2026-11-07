"""Запуск ПРИМЕРА задания практикума в браузере (Pyodide, Web Worker).

Модуль выполняет код студента на примере и сравнивает результат с ожидаемым значением, которое присылает сервер
(GET /api/v1/code/tasks/{id}/example). Оценка, баллы, граничные тесты и эталонные решения остаются на сервере и сюда не попадают:
кнопка «Проверить решение» по-прежнему работает через POST /api/v1/code/submit.

Формат результата совпадает с ответом серверного /api/v1/code/run (без поля feedback: его добавляет сервер в POST /feedback).
Файл выполняется и в Pyodide (Python 3.14), и в обычном CPython на сервере (тесты сравнивают оба пути)."""
import copy
import io
import json
import sys
import time

import numpy as np

try:                                              # в браузере воркер кладёт рядом safety.py, скачанный с сервера
    from safety import SafetyError, check_code, safe_builtins
except ImportError:                               # на сервере (тесты)
    from app.sandbox.safety import SafetyError, check_code, safe_builtins

STDOUT_LIMIT = 20000


def _warm_numpy():
    """numpy подгружает numpy._core._methods лениво при первом a.mean(), a.std(), a.var(), a.clip() ... Если это случится внутри кода
    студента, ограниченный __import__ песочницы блокирует служебный импорт («Импорт numpy._core._methods запрещён»).
    Поэтому такие модули загружаются заранее, до запуска кода студента."""
    import importlib
    for name in ("numpy._core._methods", "numpy._core._internal", "numpy._core._exceptions", "numpy._core._dtype",
                 "numpy.core._methods", "numpy.core._internal", "numpy.core._exceptions", "numpy.core._dtype"):
        try:
            importlib.import_module(name)
        except Exception:
            pass


_warm_numpy()


class Cap(io.StringIO):
    """Вывод print(): после 20000 символов запись прекращается (как Cap в серверном раннере). emit сразу отдаёт текст воркеру."""

    def __init__(self, emit=None):
        super().__init__()
        self.emit = emit

    def write(self, s):
        if self.tell() < STDOUT_LIMIT:
            n = super().write(s)
            if self.emit is not None and n:
                self.emit(s)
            return n
        return 0


def short(x, limit: int = 260) -> str:
    s = np.array2string(x, threshold=24, edgeitems=2, max_line_width=120) if isinstance(x, np.ndarray) else repr(x)
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "..."


def decode(x):
    if isinstance(x, dict):
        if x.get("__nd__"):
            return np.array(x["data"], dtype=x["dtype"]).reshape(x["shape"])
        if "__tuple__" in x:
            return tuple(decode(v) for v in x["__tuple__"])
        raise ValueError("Неизвестный формат значения")
    if isinstance(x, list):
        return [decode(v) for v in x]
    return x


# ───────────── сравнение результата с ожидаемым (те же тексты, что и на сервере) ─────────────
def _cmp_array(exp, got, dtype=None, tol=None):
    if got is None:
        return False, "функция вернула None (забыт return?)"
    try:
        g = np.asarray(got)
    except Exception:
        return False, f"результат не является массивом: {short(got)}"
    if g.shape != exp.shape:
        return False, f"неверная форма результата: ожидалась {exp.shape}, получена {g.shape}"
    if dtype is not None and g.dtype != dtype:
        return False, f"неверный тип данных: ожидался {np.dtype(dtype)}, получен {g.dtype}"
    ok = np.allclose(g, exp, atol=tol, rtol=0) if tol is not None else np.array_equal(g, exp)
    return (True, "") if ok else (False, f"значения не совпадают: ожидалось {short(exp)}, получено {short(g)}")


def _cmp_int(exp, got):
    if got is None:
        return False, "функция вернула None (забыт return?)"
    try:
        ok = float(got) == float(exp)
    except Exception:
        return False, f"результат не число: {short(got)}"
    return (True, "") if ok else (False, f"ожидалось {exp}, получено {short(got)}")


def _cmp_float(exp, got, tol=1e-9):
    if got is None:
        return False, "функция вернула None (забыт return?)"
    try:
        ok = abs(float(got) - float(exp)) <= tol
    except Exception:
        return False, f"результат не число: {short(got)}"
    return (True, "") if ok else (False, f"ожидалось {float(exp):.6f}, получено {short(got)}")


def _cmp_verify(exp, got):
    if got is None:
        return False, "функция вернула None (забыт return?)"
    try:
        s, d = got
        if abs(float(s) - exp[0]) < 1e-6 and d == exp[1]:
            return True, ""
    except Exception:
        return False, f"нужен кортеж (score, decision), получено {short(got)}"
    return False, f"ожидалось ({exp[0]:.6f}, '{exp[1]}'), получено {short(got)}"


def _cmp_pair(exp, got):
    if got is None:
        return False, "функция вернула None (забыт return?)"
    try:
        a, b = got
        if abs(float(a) - exp[0]) < 1e-9 and abs(float(b) - exp[1]) < 1e-9:
            return True, ""
    except Exception:
        return False, f"нужен кортеж (far, frr), получено {short(got)}"
    return False, f"ожидалось ({exp[0]:.4f}, {exp[1]:.4f}), получено {short(got)}"


def _cmp_eer(exp, got):
    if got is None:
        return False, "функция вернула None (забыт return?)"
    try:
        e, t = got
        if abs(float(e) - exp[0]) < 1e-9 and abs(float(t) - exp[1]) < 1e-9:
            return True, ""
    except Exception:
        return False, f"нужен кортеж (eer, порог), получено {short(got)}"
    return False, f"ожидалось ({exp[0]:.4f}, {exp[1]}), получено {short(got)}"


def _cmp_angle(exp, got):
    if got is None:
        return False, "функция вернула None (забыт return?)"
    try:
        v = float(got)
    except Exception:
        return False, f"результат не число: {short(got)}"
    if not (-1e-9 <= v < np.pi + 1e-9):
        return False, f"угол должен лежать в диапазоне [0, π), получено {v:.4f}"
    d = abs(v - exp) % np.pi
    d = min(d, np.pi - d)
    return (True, "") if d < 1e-6 else (False, f"ожидалось {exp:.4f} рад, получено {v:.4f}")


def compare(desc: dict, expected, got):
    """desc приходит с сервера: {"kind": "array", "dtype": "uint8" | None, "tol": float | None} и т.п."""
    kind = desc["kind"]
    if kind == "array":
        return _cmp_array(expected, got, np.dtype(desc["dtype"]) if desc.get("dtype") else None, desc.get("tol"))
    if kind == "int":
        return _cmp_int(expected, got)
    if kind == "float":
        return _cmp_float(expected, got, desc.get("tol", 1e-9))
    if kind == "verify":
        return _cmp_verify(expected, got)
    if kind == "pair":
        return _cmp_pair(expected, got)
    if kind == "eer":
        return _cmp_eer(expected, got)
    if kind == "angle":
        return _cmp_angle(expected, got)
    raise ValueError(f"Неизвестный способ сравнения: {kind}")


# ───────────── запуск ─────────────
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


def _blocked(error_type: str, message: str, line=None) -> dict:
    return {"error": {"type": error_type, "message": message, "line": line}, "stdout": "", "defined": False,
            "tests": [], "example": None, "time": 0.0}


def run_example(payload_json: str, emit=None, phase=None) -> str:
    """payload: {code, function, args, expected_value, cmp, input, expected} -> JSON результата, как у серверного /run.
    emit(текст) получает вывод print() по мере появления, phase("call") вызывается перед запуском функции на примере."""
    req = json.loads(payload_json)
    code, func = req["code"], req["function"]
    if len(code) > 20000:
        return json.dumps(_blocked("Blocked", "Код слишком длинный (максимум 20000 символов)."), ensure_ascii=False)
    try:
        check_code(code)
    except SafetyError as e:
        return json.dumps(_blocked("Blocked", str(e), e.line), ensure_ascii=False)
    except SyntaxError as e:
        return json.dumps(_blocked("SyntaxError", e.msg, e.lineno), ensure_ascii=False)
    args, expected = decode(req["args"]), decode(req["expected_value"])
    out = {"error": None, "stdout": "", "defined": False, "tests": [], "example": None, "time": 0.0}
    t0 = time.time()
    buf = Cap(emit)
    old = sys.stdout
    try:
        ns = {"__builtins__": safe_builtins(), "__name__": "student"}
        sys.stdout = buf
        try:
            exec(compile(code, "<student>", "exec"), ns)
        finally:
            sys.stdout = old
        fn = ns.get(func)
        out["defined"] = callable(fn)
        if not out["defined"]:
            out["error"] = {"type": "NoFunction", "message": f"Не найдена функция {func}", "line": None}
        else:
            ex = {"input": req["input"], "expected": req["expected"], "ok": False, "message": "", "error": None}
            sys.stdout = buf
            if phase is not None:
                phase("call")
            try:
                try:
                    got = fn(*copy.deepcopy(args))
                except BaseException as e:
                    ex["error"] = _err(e)
                    ex["message"] = f"{ex['error']['type']}: {ex['error']['message']}"
                else:
                    ok, msg = compare(req["cmp"], expected, got)
                    ex["ok"] = bool(ok)
                    if not ok:
                        ex["message"] = msg
            finally:
                sys.stdout = old
            out["example"] = ex
    except SafetyError as e:
        out["error"] = {"type": "Blocked", "message": str(e), "line": e.line}
    except SyntaxError as e:
        out["error"] = {"type": "SyntaxError", "message": e.msg, "line": e.lineno}
    except BaseException as e:
        out["error"] = _err(e)
    finally:
        sys.stdout = old
    out["stdout"] = buf.getvalue()
    out["time"] = round(time.time() - t0, 3)
    return json.dumps(out, default=str, ensure_ascii=False)
