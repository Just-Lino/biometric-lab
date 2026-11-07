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
         Case("массив 5×7, порог 50", (_rand((5, 7), 3), 50)),
         Case("массив 3×9, порог 177", (_rand((3, 9), 7), 177)),
         Case("фрагмент отпечатка, порог 90", (_fp_crop(), 90))],
        [Case("порог 0: все пиксели светлые", (_rand((4, 4), 4), 0), "edge"),
         Case("пиксель, равный порогу", (np.array([[127, 128, 129]], np.uint8), 128), "edge"),
         Case("порог 255 и крайние значения", (np.array([[254, 255, 0]], np.uint8), 255), "edge"),
         Case("изображение 1×1", (np.array([[7]], np.uint8), 8), "edge"),
         Case("порог выше всех значений", (np.full((2, 2), 100, np.uint8), 101), "edge")])


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
         Case("массив 4×6", (_rand((4, 6), 6), 128.0, 50.0)),
         Case("массив 3×8, m0=0, s0=1", (_rand((3, 8), 8), 0.0, 1.0)),
         Case("фрагмент отпечатка, m0=128, s0=50", (_fp_crop(20), 128.0, 50.0))],
        [Case("все пиксели одинаковы (std = 0)", (np.full((3, 3), 77, np.uint8), 128.0, 50.0), "edge"),
         Case("изображение 1×1", (np.array([[9]], np.uint8), 128.0, 50.0), "edge"),
         Case("два значения 2×2", (np.array([[0, 10], [10, 0]], np.uint8), 90.0, 20.0), "edge"),
         Case("отрицательное целевое среднее", (_rand((3, 3), 9), -50.0, 10.0), "edge")])


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
         Case("случайный узор, точка (2,4)", (rnd(12), 2, 4)),
         Case("окончание линии сверху", (a([[0, 1, 0], [0, 1, 0], [0, 0, 0]]), 1, 1)),
         Case("перекрёсток (CN=4)", (a([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), 1, 1))],
        [Case("угол (0,0) в заполненном массиве", (np.ones((4, 4), np.uint8), 0, 0), "edge"),
         Case("правый нижний угол", (np.ones((4, 4), np.uint8), 3, 3), "edge"),
         Case("верхняя строка", (np.ones((3, 5), np.uint8), 0, 2), "edge"),
         Case("последний столбец", (a([[1, 0, 1], [1, 1, 1], [1, 0, 1]]), 1, 2), "edge"),
         Case("массив 1×1", (np.ones((1, 1), np.uint8), 0, 0), "edge")])


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
         Case("похожие векторы-списки", ([3, 4, 5], [3, 4, 6], 0.95)),
         Case("одинаковые векторы", ([2, 3, 4], [2, 3, 4], 0.99)),
         Case("угол 30°, порог 0.9", ([1, 0], [float(np.cos(np.pi / 6)), float(np.sin(np.pi / 6))], 0.9))],
        [Case("нулевой вектор", ([0, 0, 0], [1, 2, 3], 0.5), "edge"),
         Case("сходство ровно равно порогу", ([1, 0], [1, 0], 1.0), "edge"),
         Case("разный масштаб векторов", ([1e-3, 2e-3], [10, 20], 0.99), "edge"),
         Case("отрицательный порог", ([1, 0], [-1, 0], -1.0), "edge"),
         Case("оба вектора нулевые", ([0, 0], [0, 0], 0.0), "edge")])


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
         Case("порог 0.7", (g3, i3, 0.7)), Case("малые списки", ([0.6, 0.9], [0.2, 0.4, 0.8], 0.6)),
         Case("порог 0.1", (g2, i2, 0.1)), Case("порог 0.55", (g3, i3, 0.55))],
        [Case("список чужих пуст", ([0.9, 0.2], [], 0.5), "edge"),
         Case("список своих пуст", ([], [0.9, 0.2], 0.5), "edge"),
         Case("оценка равна порогу", ([0.5, 0.5], [0.5, 0.5], 0.5), "edge"),
         Case("оба списка пусты", ([], [], 0.5), "edge"),
         Case("все оценки выше порога", ([0.9, 0.8], [0.7, 0.6], 0.1), "edge")])


def _cmp_float(exp, got, tol=1e-9):
    if got is None:
        return False, "функция вернула None (забыт return?)"
    try:
        ok = abs(float(got) - float(exp)) <= tol
    except Exception:
        return False, f"результат не число: {short(got)}"
    return (True, "") if ok else (False, f"ожидалось {float(exp):.6f}, получено {short(got)}")


# ---------- hamming (радужка) ----------
def _hamming(code_a, code_b, mask_a, mask_b):
    valid = (np.asarray(mask_a) == 1) & (np.asarray(mask_b) == 1)
    n = int(valid.sum())
    if n == 0:
        return 1.0
    return float(((np.asarray(code_a) != np.asarray(code_b)) & valid).sum()) / n


def spec_hamming() -> Spec:
    bits = lambda shape, seed: np.random.RandomState(seed).randint(0, 2, shape).astype(np.uint8)
    ones = lambda shape: np.ones(shape, np.uint8)
    a = bits((4, 16), 41)
    return Spec("hamming_distance", _hamming, _cmp_float,
        Case("пример", (np.array([1, 0, 1, 1], np.uint8), np.array([1, 1, 1, 0], np.uint8), np.ones(4, np.uint8), np.ones(4, np.uint8))),
        [Case("случайные коды 4×16 без масок", (a, bits((4, 16), 43), ones((4, 16)), ones((4, 16)))),
         Case("одинаковые коды", (a, a.copy(), ones((4, 16)), ones((4, 16)))),
         Case("противоположные коды", (a, 1 - a, ones((4, 16)), ones((4, 16)))),
         Case("случайные коды с масками", (bits((8, 8), 44), bits((8, 8), 45), bits((8, 8), 46), bits((8, 8), 47))),
         Case("маска закрывает половину бит", (np.array([1, 1, 0, 0], np.uint8), np.array([0, 1, 1, 0], np.uint8), np.array([1, 1, 0, 0], np.uint8), np.array([1, 1, 1, 1], np.uint8)))],
        [Case("нет общих надёжных бит", (np.array([1, 0], np.uint8), np.array([0, 1], np.uint8), np.array([1, 0], np.uint8), np.array([0, 1], np.uint8)), "edge"),
         Case("один бит, различается", (np.array([1], np.uint8), np.array([0], np.uint8), np.array([1], np.uint8), np.array([1], np.uint8)), "edge"),
         Case("ненадёжные биты различаются, но игнорируются", (np.array([0, 0, 0], np.uint8), np.array([1, 0, 0], np.uint8), np.array([0, 1, 1], np.uint8), np.array([0, 1, 1], np.uint8)), "edge"),
         Case("двумерные коды 2×3", (np.array([[1, 0, 1], [0, 0, 1]], np.uint8), np.array([[1, 1, 0], [0, 0, 0]], np.uint8), ones((2, 3)), ones((2, 3))), "edge")])


# ---------- eer_threshold ----------
def _eer(genuine, impostor):
    g, i = [float(x) for x in genuine], [float(x) for x in impostor]
    cands = sorted(set(g) | set(i))
    if not cands:
        return 0.0, 0.0
    best = None
    for t in cands:
        far = sum(s >= t for s in i) / len(i) if i else 0.0
        frr = sum(s < t for s in g) / len(g) if g else 0.0
        d = abs(far - frr)
        if best is None or d < best[0] - 1e-12:
            best = (d, (far + frr) / 2, t)
    return best[1], best[2]


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


def spec_eer() -> Spec:
    r = np.random.RandomState(51)
    mk = lambda n, m: (list(np.round(r.uniform(0.35, 1.0, n), 3)), list(np.round(r.uniform(0.0, 0.7, m), 3)))
    g1, i1 = mk(10, 14); g2, i2 = mk(9, 11); g3, i3 = mk(12, 12)
    return Spec("eer_threshold", _eer, _cmp_eer,
        Case("пример", ([0.9, 0.8, 0.7], [0.1, 0.2, 0.75])),
        [Case("случайные оценки 10 и 14", (g1, i1)), Case("случайные оценки 9 и 11", (g2, i2)),
         Case("случайные оценки 12 и 12", (g3, i3)),
         Case("небольшие списки", ([0.6, 0.9, 0.7, 0.8], [0.2, 0.4, 0.65, 0.1]))],
        [Case("полное разделение (EER = 0)", ([0.8, 0.9, 0.95], [0.1, 0.2, 0.3]), "edge"),
         Case("по одной оценке в списках", ([0.9], [0.1]), "edge"),
         Case("списки совпадают целиком", ([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]), "edge"),
         Case("свои оценки ниже чужих", ([0.1, 0.2], [0.8, 0.9]), "edge"),
         Case("оба списка пусты", ([], []), "edge")])


# ---------- ridge_orientation ----------
def _orient(gray):
    g = np.asarray(gray, dtype=float)
    gy, gx = np.gradient(g)
    gxx, gyy, gxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    if gxx + gyy == 0:
        return 0.0
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    return float((theta + np.pi / 2) % np.pi)


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


def spec_orientation() -> Spec:
    yy, xx = np.mgrid[0:8, 0:8]
    vert = (127 + 100 * np.sin(xx * 1.1)).astype(np.uint8)            # гребни вертикальные
    horiz = (127 + 100 * np.sin(yy * 1.1)).astype(np.uint8)           # гребни горизонтальные
    diag = (127 + 100 * np.sin((xx + yy) * 0.9)).astype(np.uint8)
    anti = (127 + 100 * np.sin((xx - yy) * 0.9)).astype(np.uint8)
    return Spec("ridge_orientation", _orient, _cmp_angle,
        Case("пример: вертикальные полосы", (np.array([[0, 255, 0, 255]] * 4, np.uint8),)),
        [Case("вертикальные гребни", (vert,)), Case("горизонтальные гребни", (horiz,)),
         Case("диагональ вида «\\»", (diag,)), Case("диагональ вида «/»", (anti,)),
         Case("фрагмент отпечатка", (_fp_crop(24),)), Case("случайный блок 6×6", (_rand((6, 6), 52),))],
        [Case("постоянное изображение", (np.full((5, 5), 90, np.uint8),), "edge"),
         Case("блок 2×2", (np.array([[0, 255], [0, 255]], np.uint8),), "edge"),
         Case("блок 2×2 с диагональным перепадом", (np.array([[0, 0], [0, 255]], np.uint8),), "edge"),
         Case("значения uint8 с большим перепадом", (np.array([[255, 0, 255], [255, 0, 255], [255, 0, 255]], np.uint8),), "edge")])


# ---------- lbp_code ----------
_LBP_ORDER = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]


def _lbp(patch):
    p = np.asarray(patch)
    c = p[1, 1]
    return int(sum(2 ** i for i, (y, x) in enumerate(_LBP_ORDER) if p[y, x] >= c))


def spec_lbp() -> Spec:
    a = lambda rows: np.array(rows, np.uint8)
    rp = lambda seed: _rand((3, 3), seed)
    return Spec("lbp_code", _lbp, _cmp_int,
        Case("пример", (a([[9, 3, 7], [1, 5, 6], [2, 8, 4]]),)),
        [Case("случайный фрагмент 1", (rp(61),)), Case("случайный фрагмент 2", (rp(62),)),
         Case("случайный фрагмент 3", (rp(63),)), Case("градиент яркости слева направо", (a([[1, 5, 9], [1, 5, 9], [1, 5, 9]]),)),
         Case("яркий верхний ряд", (a([[200, 200, 200], [10, 100, 10], [10, 10, 10]]),))],
        [Case("все пиксели равны (сосед >= центра)", (np.full((3, 3), 50, np.uint8),), "edge"),
         Case("центр самый яркий", (a([[1, 2, 3], [4, 250, 6], [7, 8, 9]]),), "edge"),
         Case("центр самый тёмный", (a([[200, 201, 202], [203, 0, 204], [205, 206, 207]]),), "edge"),
         Case("соседи ровно равны центру на одной стороне", (a([[5, 5, 5], [0, 5, 0], [0, 0, 0]]),), "edge")])


# ---------- mel_points ----------
def _mel_points(f_min, f_max, n):
    to_mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)
    to_hz = lambda m: 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)
    return to_hz(np.linspace(to_mel(f_min), to_mel(f_max), int(n) + 2))


def spec_mel() -> Spec:
    return Spec("mel_points", _mel_points, lambda e, g: _cmp_array(e, g, None, 1e-6),
        Case("пример", (0, 8000, 4)),
        [Case("диапазон 0-8000 Гц, 10 фильтров", (0, 8000, 10)), Case("телефонный диапазон 300-3400, 5 фильтров", (300, 3400, 5)),
         Case("диапазон 100-4000, 8 фильтров", (100, 4000, 8)), Case("один фильтр", (0, 8000, 1))],
        [Case("нулевое число фильтров", (0, 8000, 0), "edge"),
         Case("границы совпадают", (1000, 1000, 3), "edge"),
         Case("узкий диапазон", (500.0, 520.0, 2), "edge"),
         Case("много фильтров", (0, 16000, 40), "edge")])


# ---------- rank1_accuracy ----------
def _rank1(scores, true_idx):
    s = np.asarray(scores, dtype=float)
    if s.shape[0] == 0:
        return 0.0
    return float(np.mean(np.argmax(s, axis=1) == np.asarray(true_idx)))


def spec_rank1() -> Spec:
    r = np.random.RandomState(71)
    mk = lambda n, k: ([[float(x) for x in row] for row in np.round(r.rand(n, k), 3)], [int(x) for x in r.randint(0, k, n)])
    s1, t1 = mk(8, 5); s2, t2 = mk(10, 4); s3, t3 = mk(6, 6)
    return Spec("rank1_accuracy", _rank1, _cmp_float,
        Case("пример", ([[0.9, 0.1], [0.2, 0.8], [0.6, 0.5]], [0, 1, 1])),
        [Case("случайная таблица 8×5", (s1, t1)), Case("случайная таблица 10×4", (s2, t2)),
         Case("случайная таблица 6×6", (s3, t3)),
         Case("все пробы распознаны", ([[0.9, 0.2, 0.1], [0.1, 0.8, 0.3], [0.2, 0.3, 0.7]], [0, 1, 2]))],
        [Case("нет проб", ([], []), "edge"),
         Case("одна проба и одна запись", ([[0.5]], [0]), "edge"),
         Case("равные оценки: берётся первая запись", ([[0.5, 0.5, 0.5], [0.4, 0.4, 0.1]], [0, 1]), "edge"),
         Case("ни одна проба не распознана", ([[0.1, 0.9], [0.8, 0.2]], [0, 1]), "edge"),
         Case("отрицательные оценки", ([[-0.5, -0.1, -0.9], [-0.3, -0.7, -0.2]], [1, 2]), "edge")])


# ---------- count_matched ----------
def _matched(points_a, points_b, radius):
    cnt = 0
    for ax, ay in points_a:
        if any((ax - bx) ** 2 + (ay - by) ** 2 <= radius ** 2 for bx, by in points_b):
            cnt += 1
    return cnt


def spec_matched() -> Spec:
    r = np.random.RandomState(81)
    pts = lambda n: [(int(x), int(y)) for x, y in r.randint(0, 60, (n, 2))]
    return Spec("count_matched", _matched, _cmp_int,
        Case("пример", ([(0, 0), (10, 10), (50, 50)], [(1, 1), (30, 30)], 3)),
        [Case("случайные точки, допуск 8", (pts(12), pts(15), 8)), Case("случайные точки, допуск 5", (pts(10), pts(10), 5)),
         Case("случайные точки, допуск 15", (pts(9), pts(14), 15)),
         Case("несколько точек B рядом с одной точкой A", ([(10, 10)], [(11, 10), (10, 12), (9, 9)], 3))],
        [Case("расстояние ровно равно допуску (3-4-5)", ([(0, 0)], [(3, 4)], 5), "edge"),
         Case("второй набор пуст", ([(1, 1), (2, 2)], [], 5), "edge"),
         Case("первый набор пуст", ([], [(1, 1)], 5), "edge"),
         Case("допуск 0: учитываются только совпадающие точки", ([(1, 1), (2, 2), (3, 3)], [(1, 1), (3, 4)], 0), "edge"),
         Case("одна точка B рядом с двумя точками A", ([(0, 0), (2, 0)], [(1, 0)], 1), "edge")])


SPECS = {"binarize": spec_binarize, "normalize": spec_normalize, "crossing_number": spec_crossing,
         "verify": spec_verify, "far_frr": spec_far_frr, "hamming": spec_hamming, "eer": spec_eer,
         "orientation": spec_orientation, "lbp": spec_lbp, "mel": spec_mel, "rank1": spec_rank1, "matched": spec_matched}


def get_spec(name: str) -> Spec:
    return SPECS[name]()


# Описание сравнения результата с ожидаемым для примера, выполняемого в браузере (см. app/web/codelab/lab_runner.py:compare).
# Это только способ сравнения (вид, допуск, тип данных), а не эталонное решение.
CMP_DESC = {
    "binarize": {"kind": "array", "dtype": "uint8", "tol": None},
    "normalize": {"kind": "array", "dtype": None, "tol": 1e-6},
    "crossing_number": {"kind": "int"},
    "verify": {"kind": "verify"},
    "far_frr": {"kind": "pair"},
    "hamming": {"kind": "float", "tol": 1e-9},
    "eer": {"kind": "eer"},
    "orientation": {"kind": "angle"},
    "lbp": {"kind": "int"},
    "mel": {"kind": "array", "dtype": None, "tol": 1e-6},
    "rank1": {"kind": "float", "tol": 1e-9},
    "matched": {"kind": "int"},
}
