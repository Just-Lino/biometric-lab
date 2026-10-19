"""Эталонные реализации и тестовые случаи для заданий (доверенный код, выполняется в процессе-раннере)."""
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Case:
    name: str
    args: tuple
    kind: str = "basic"  # basic | edge


@dataclass
class Spec:
    func: str
    ref: Callable
    cmp: Callable
    example: Case
    basic: list = field(default_factory=list)
    edge: list = field(default_factory=list)


def short(x: Any, limit: int = 260) -> str:
    s = np.array2string(x, threshold=24, edgeitems=2, max_line_width=120) if isinstance(x, np.ndarray) else repr(x)
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "..."


def _fp_crop(n: int = 40) -> np.ndarray:
    try:
        import cv2
        from app.modules.fingerprint import synthetic
        p = synthetic.DATASET_DIR / "f01_i1.png"
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)[100:100 + n, 100:100 + n].copy()
    except Exception:
        pass
    yy, xx = np.mgrid[0:n, 0:n]
    return (127 + 100 * np.sin(xx / 2.3 + 0.3 * yy)).astype(np.uint8)


def _rand(shape, seed):
    return np.random.RandomState(seed).randint(0, 256, shape).astype(np.uint8)


# ---------- binarize ----------
def _binarize(gray, threshold):
    return np.where(gray < threshold, 0, 255).astype(np.uint8)


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


def spec_binarize() -> Spec:
    return Spec("binarize", _binarize, lambda e, g: _cmp_array(e, g, np.uint8),
        Case("пример", (np.array([[10, 200, 128], [127, 129, 0]], np.uint8), 128)),
        [Case("случайный массив 8×8, порог 100", (_rand((8, 8), 1), 100)),
         Case("фрагмент отпечатка, порог 128", (_fp_crop(), 128)),
         Case("случайный массив 6×6, порог 200", (_rand((6, 6), 2), 200)),
         Case("массив 5×7, порог 50", (_rand((5, 7), 3), 50))],
        [Case("порог 0: все пиксели светлые", (_rand((4, 4), 4), 0), "edge"),
         Case("пиксель, равный порогу", (np.array([[127, 128, 129]], np.uint8), 128), "edge"),
         Case("порог 255 и крайние значения", (np.array([[254, 255, 0]], np.uint8), 255), "edge"),
         Case("изображение 1×1", (np.array([[7]], np.uint8), 8), "edge")])


# ---------- normalize ----------
def _normalize(gray, m0=128.0, s0=50.0):
    g = np.asarray(gray, dtype=float)
    std = g.std()
    if std == 0:
        return np.full(g.shape, m0, dtype=float)
    return (g - g.mean()) / std * s0 + m0


def spec_normalize() -> Spec:
    return Spec("normalize", _normalize, lambda e, g: _cmp_array(e, g, None, 1e-6),
        Case("пример", (np.array([[0, 255], [100, 200]], np.uint8), 128.0, 50.0)),
        [Case("случайный массив 5×5", (_rand((5, 5), 5), 128.0, 50.0)),
         Case("фрагмент отпечатка, m0=100, s0=30", (_fp_crop(), 100.0, 30.0)),
         Case("массив 4×6", (_rand((4, 6), 6), 128.0, 50.0))],
        [Case("все пиксели одинаковы (std = 0)", (np.full((3, 3), 77, np.uint8), 128.0, 50.0), "edge"),
         Case("изображение 1×1", (np.array([[9]], np.uint8), 128.0, 50.0), "edge"),
         Case("два значения 2×2", (np.array([[0, 10], [10, 0]], np.uint8), 90.0, 20.0), "edge")])


# ---------- crossing_number ----------
_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _crossing(skel, y, x):
    h, w = skel.shape
    p = [int(skel[y + dy, x + dx]) if 0 <= y + dy < h and 0 <= x + dx < w else 0 for dy, dx in _OFFS]
    return sum(abs(p[i] - p[(i + 1) % 8]) for i in range(8)) // 2


def _cmp_int(exp, got):
    if got is None:
        return False, "функция вернула None (забыт return?)"
    try:
        ok = float(got) == float(exp)
    except Exception:
        return False, f"результат не число: {short(got)}"
    return (True, "") if ok else (False, f"ожидалось {exp}, получено {short(got)}")


def spec_crossing() -> Spec:
    a = lambda rows: np.array(rows, np.uint8)
    rnd = lambda seed: (np.random.RandomState(seed).rand(7, 7) > 0.5).astype(np.uint8)
    return Spec("crossing_number", _crossing, _cmp_int,
        Case("пример: окончание линии", (a([[0, 0, 0], [0, 1, 1], [0, 0, 0]]), 1, 1)),
        [Case("окончание линии (CN=1)", (a([[0, 0, 0], [0, 1, 1], [0, 0, 0]]), 1, 1)),
         Case("точка на линии (CN=2)", (a([[0, 0, 0], [1, 1, 1], [0, 0, 0]]), 1, 1)),
         Case("разветвление (CN=3)", (a([[0, 0, 1], [1, 1, 0], [0, 0, 1]]), 1, 1)),
         Case("изолированный пиксель", (a([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), 1, 1)),
         Case("случайный узор, точка (3,3)", (rnd(11), 3, 3)),
         Case("случайный узор, точка (2,4)", (rnd(12), 2, 4))],
        [Case("угол (0,0) в заполненном массиве", (np.ones((4, 4), np.uint8), 0, 0), "edge"),
         Case("правый нижний угол", (np.ones((4, 4), np.uint8), 3, 3), "edge"),
         Case("верхняя строка", (np.ones((3, 5), np.uint8), 0, 2), "edge"),
         Case("последний столбец", (a([[1, 0, 1], [1, 1, 1], [1, 0, 1]]), 1, 2), "edge")])


# ---------- verify ----------
def _verify(a, b, threshold):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    score = 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))
    return score, ("MATCH" if score >= threshold else "NO MATCH")


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


def spec_verify() -> Spec:
    r = np.random.RandomState(21)
    return Spec("verify", _verify, _cmp_verify,
        Case("пример", ([1, 2, 3], [2, 4, 6], 0.9)),
        [Case("случайные векторы, порог 0.5", (r.randn(8), r.randn(8), 0.5)),
         Case("ортогональные векторы", ([1, 0], [0, 1], 0.1)),
         Case("противоположные векторы", ([1, 2], [-1, -2], 0.0)),
         Case("похожие векторы-списки", ([3, 4, 5], [3, 4, 6], 0.95))],
        [Case("нулевой вектор", ([0, 0, 0], [1, 2, 3], 0.5), "edge"),
         Case("сходство ровно равно порогу", ([1, 0], [1, 0], 1.0), "edge"),
         Case("разный масштаб векторов", ([1e-3, 2e-3], [10, 20], 0.99), "edge"),
         Case("отрицательный порог", ([1, 0], [-1, 0], -1.0), "edge")])


# ---------- far_frr ----------
def _far_frr(genuine, impostor, threshold):
    far = sum(s >= threshold for s in impostor) / len(impostor) if len(impostor) else 0.0
    frr = sum(s < threshold for s in genuine) / len(genuine) if len(genuine) else 0.0
    return far, frr


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


def spec_far_frr() -> Spec:
    r = np.random.RandomState(31)
    mk = lambda: (list(np.round(r.uniform(0.4, 1.0, 12), 3)), list(np.round(r.uniform(0.0, 0.7, 15), 3)))
    g1, i1 = mk(); g2, i2 = mk(); g3, i3 = mk()
    return Spec("far_frr", _far_frr, _cmp_pair,
        Case("пример", ([0.9, 0.8, 0.4], [0.1, 0.5, 0.3], 0.5)),
        [Case("порог 0.5", (g1, i1, 0.5)), Case("порог 0.3", (g2, i2, 0.3)),
         Case("порог 0.7", (g3, i3, 0.7)), Case("малые списки", ([0.6, 0.9], [0.2, 0.4, 0.8], 0.6))],
        [Case("список чужих пуст", ([0.9, 0.2], [], 0.5), "edge"),
         Case("список своих пуст", ([], [0.9, 0.2], 0.5), "edge"),
         Case("оценка равна порогу", ([0.5, 0.5], [0.5, 0.5], 0.5), "edge"),
         Case("оба списка пусты", ([], [], 0.5), "edge")])


SPECS = {"binarize": spec_binarize, "normalize": spec_normalize, "crossing_number": spec_crossing,
         "verify": spec_verify, "far_frr": spec_far_frr}


def get_spec(name: str) -> Spec:
    return SPECS[name]()
