"""JSON-представление входа и ожидаемого результата ПРИМЕРА задания (для запуска примера в браузере через Pyodide).

Кодируются только значения примера, который и так виден студенту в ответе /run. Граничные и базовые тесты, эталонные
решения и подсказки сюда не попадают. Браузерная сторона (app/web/codelab/lab_runner.py) раскодирует значения тем же форматом."""
from typing import Any

import numpy as np


def encode(x: Any) -> Any:
    """Python-значение -> JSON-совместимая структура: массивы numpy, кортежи, списки, числа, строки, None."""
    if isinstance(x, np.ndarray):
        return {"__nd__": True, "dtype": str(x.dtype), "shape": list(x.shape), "data": x.ravel().tolist()}
    if isinstance(x, tuple):
        return {"__tuple__": [encode(v) for v in x]}
    if isinstance(x, list):
        return [encode(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    raise TypeError(f"Нельзя закодировать значение типа {type(x).__name__}")


def decode(x: Any) -> Any:
    """Обратное преобразование (используется в тестах; в браузере то же делает lab_runner.decode)."""
    if isinstance(x, dict):
        if x.get("__nd__"):
            return np.array(x["data"], dtype=x["dtype"]).reshape(x["shape"])
        if "__tuple__" in x:
            return tuple(decode(v) for v in x["__tuple__"])
        raise ValueError("Неизвестный формат значения")
    if isinstance(x, list):
        return [decode(v) for v in x]
    return x
