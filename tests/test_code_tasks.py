"""Практикум по коду: структура всех заданий, типичные неверные решения и согласованность с эталонами."""
import os
import tempfile

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "code.db"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.sandbox.executor import execute  # noqa: E402
from app.sandbox.reference import SPECS, get_spec  # noqa: E402
from app.sandbox.tasks import get_task, load_tasks, public_task  # noqa: E402
from app.services.grader import grade  # noqa: E402

TASKS = load_tasks()


def _norm(code: str) -> str:
    """Код без пустых строк: в подсказке 4 отступы между функциями могут отличаться от файла решения."""
    return "\n".join(l.rstrip() for l in code.strip().splitlines() if l.strip())


def test_there_are_at_least_twelve_tasks_in_a_stable_order():
    assert len(TASKS) >= 12
    orders = [t["order"] for t in TASKS.values()]
    assert orders == sorted(orders) and len(set(orders)) == len(orders)


@pytest.mark.parametrize("tid", list(TASKS))
def test_task_json_is_complete_and_consistent(tid):
    t = TASKS[tid]
    assert t["spec"] in SPECS and t["function"] == get_spec(t["spec"]).func
    assert f"def {t['function']}(" in t["starter_code"] and f"def {t['function']}(" in t["solution"]
    assert len(t["hints"]) == 4 and all(h.strip() for h in t["hints"])
    assert _norm(t["solution"]) in _norm(t["hints"][3])               # подсказка 4 содержит полное решение
    assert _norm(t["solution"]) not in _norm("\n".join(t["hints"][:3]))  # а подсказки 1-3 решения не раскрывают
    q = t["question"]
    assert len(q["options"]) == 4 and 0 <= q["correct"] < 4 and len(set(q["options"])) == 4
    assert sum(t["rubric"].values()) == 10 and set(t["rubric"]) == {"run", "basic", "edge", "question"}
    assert t["difficulty"] in ("базовый", "средний", "сложный")
    for key in ("title", "topic", "theory", "description"):
        assert t[key].strip()


@pytest.mark.parametrize("name", list(SPECS))
def test_every_spec_has_enough_cases_with_unique_names(name):
    sp = get_spec(name)
    assert len(sp.basic) >= 4 and len(sp.edge) >= 3, name
    names = [c.name for c in sp.basic + sp.edge]
    assert len(names) == len(set(names)), "названия тестов должны быть различны"
    assert all(c.kind == "basic" for c in sp.basic) and all(c.kind == "edge" for c in sp.edge)


@pytest.mark.parametrize("tid", list(TASKS))
def test_reference_solution_passes_every_case_and_example(tid):
    t = get_task(tid)
    res = execute(t["spec"], t["solution"], "grade")
    sp = get_spec(t["spec"])
    assert res["error"] is None and len(res["tests"]) == len(sp.basic) + len(sp.edge)
    assert all(x["ok"] for x in res["tests"]), [x for x in res["tests"] if not x["ok"]]
    ex = execute(t["spec"], t["solution"], "example")
    assert ex["example"]["ok"] is True and ex["example"]["expected"]


def test_public_task_never_contains_secrets():
    for t in TASKS.values():
        p = public_task(t)
        assert not ({"solution", "hints"} & set(p)) and "correct" not in p["question"]
        assert t["solution"].strip() not in str(p) and _norm(t["solution"]) not in str(p).replace("\\n", "\n")


# типичные ошибки студентов: (задание, название, код)
WRONG = [
    ("binarize", "<= вместо <", "import numpy as np\ndef binarize(g, t):\n    return np.where(g <= t, 0, 255).astype(np.uint8)\n"),
    ("binarize", "тип не uint8", "import numpy as np\ndef binarize(g, t):\n    return np.where(g < t, 0, 255)\n"),
    ("normalize", "std с ddof=1", "import numpy as np\ndef normalize(gray, m0=128.0, s0=50.0):\n    g = np.asarray(gray, dtype=float)\n    return (g - g.mean()) / g.std(ddof=1) * s0 + m0\n"),
    ("normalize", "без защиты от std=0", "import numpy as np\ndef normalize(gray, m0=128.0, s0=50.0):\n    g = np.asarray(gray, dtype=float)\n    return (g - g.mean()) / g.std() * s0 + m0\n"),
    ("crossing_number", "нет проверки границ", "def crossing_number(skel, y, x):\n    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]\n    p = [int(skel[y + dy, x + dx]) for dy, dx in offs]\n    return sum(abs(p[i] - p[(i + 1) % 8]) for i in range(8)) // 2\n"),
    ("verify", "нет защиты от нулевого вектора", "import numpy as np\ndef verify(a, b, threshold):\n    a = np.asarray(a, float)\n    b = np.asarray(b, float)\n    s = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))\n    return s, ('MATCH' if s >= threshold else 'NO MATCH')\n"),
    ("verify", "> вместо >=", "import numpy as np\ndef verify(a, b, threshold):\n    a = np.asarray(a, float)\n    b = np.asarray(b, float)\n    na, nb = np.linalg.norm(a), np.linalg.norm(b)\n    s = 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))\n    return s, ('MATCH' if s > threshold else 'NO MATCH')\n"),
    ("far_frr", "деление на длину пустого списка", "def far_frr(genuine, impostor, threshold):\n    far = sum(s >= threshold for s in impostor) / len(impostor)\n    frr = sum(s < threshold for s in genuine) / len(genuine)\n    return far, frr\n"),
    ("hamming", "маски игнорируются", "import numpy as np\ndef hamming_distance(code_a, code_b, mask_a, mask_b):\n    return float((code_a != code_b).mean())\n"),
    ("hamming", "деление на ноль без проверки", "import numpy as np\ndef hamming_distance(code_a, code_b, mask_a, mask_b):\n    valid = (mask_a == 1) & (mask_b == 1)\n    return float(((code_a != code_b) & valid).sum()) / int(valid.sum())\n"),
    ("eer", "> вместо >= для чужих", "def eer_threshold(genuine, impostor):\n    cands = sorted(set(genuine) | set(impostor))\n    if not cands:\n        return 0.0, 0.0\n    best = None\n    for t in cands:\n        far = sum(s > t for s in impostor) / len(impostor) if impostor else 0.0\n        frr = sum(s < t for s in genuine) / len(genuine) if genuine else 0.0\n        d = abs(far - frr)\n        if best is None or d < best[0] - 1e-12:\n            best = (d, (far + frr) / 2, t)\n    return best[1], best[2]\n"),
    ("eer", "фиксированный порог 0.5", "def eer_threshold(genuine, impostor):\n    far = sum(s >= 0.5 for s in impostor) / len(impostor) if impostor else 0.0\n    frr = sum(s < 0.5 for s in genuine) / len(genuine) if genuine else 0.0\n    return (far + frr) / 2, 0.5\n"),
    ("orientation", "угол градиента без + pi/2", "import numpy as np\ndef ridge_orientation(gray):\n    gy, gx = np.gradient(gray.astype(float))\n    sxx, syy, sxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()\n    if sxx + syy == 0:\n        return 0.0\n    return float((0.5 * np.arctan2(2 * sxy, sxx - syy)) % np.pi)\n"),
    ("orientation", "arctan2 без удвоения (sxy вместо 2*sxy)", "import numpy as np\ndef ridge_orientation(gray):\n    gy, gx = np.gradient(gray.astype(float))\n    sxx, syy, sxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()\n    if sxx + syy == 0:\n        return 0.0\n    return float((0.5 * np.arctan2(sxy, sxx - syy) + np.pi / 2) % np.pi)\n"),
    ("lbp", "> вместо >=", "def lbp_code(patch):\n    c = patch[1, 1]\n    order = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]\n    return sum(2 ** i for i, (y, x) in enumerate(order) if patch[y, x] > c)\n"),
    ("lbp", "обратный порядок бит", "def lbp_code(patch):\n    c = patch[1, 1]\n    order = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]\n    return sum(2 ** (7 - i) for i, (y, x) in enumerate(order) if patch[y, x] >= c)\n"),
    ("mel", "равномерно по герцам", "import numpy as np\ndef mel_points(f_min, f_max, n):\n    return np.linspace(f_min, f_max, n + 2)\n"),
    ("mel", "n точек вместо n + 2", "import numpy as np\ndef mel_points(f_min, f_max, n):\n    to_mel = lambda f: 2595 * np.log10(1 + f / 700)\n    to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)\n    return to_hz(np.linspace(to_mel(f_min), to_mel(f_max), n))\n"),
    ("rank1", "argmin вместо argmax", "import numpy as np\ndef rank1_accuracy(scores, true_idx):\n    s = np.asarray(scores, dtype=float)\n    if s.shape[0] == 0:\n        return 0.0\n    return float(np.mean(np.argmin(s, axis=1) == np.asarray(true_idx)))\n"),
    ("rank1", "нет защиты от пустой таблицы", "import numpy as np\ndef rank1_accuracy(scores, true_idx):\n    s = np.asarray(scores, dtype=float)\n    return float(np.mean(np.argmax(s, axis=1) == np.asarray(true_idx)))\n"),
    ("matched", "считаются пары, а не точки", "def count_matched(points_a, points_b, radius):\n    return sum(1 for ax, ay in points_a for bx, by in points_b if (ax - bx) ** 2 + (ay - by) ** 2 <= radius ** 2)\n"),
    ("matched", "строгое сравнение с допуском", "def count_matched(points_a, points_b, radius):\n    return sum(1 for ax, ay in points_a if any((ax - bx) ** 2 + (ay - by) ** 2 < radius ** 2 for bx, by in points_b))\n"),
]


@pytest.mark.parametrize("tid,why,code", WRONG, ids=[f"{w[0]}-{w[1]}" for w in WRONG])
def test_typical_mistake_is_not_graded_as_correct(tid, why, code):
    t = get_task(tid)
    res = execute(t["spec"], code, "grade")
    g = grade(t, res, t["question"]["correct"])
    assert g["score"] < g["max_score"], f"ошибка «{why}» не обнаружена тестами"
    assert g["score"] >= 0


def test_each_task_has_a_mistake_in_the_catalog():
    covered = {w[0] for w in WRONG}
    assert covered == set(TASKS), "для каждого задания должна быть проверка типичной ошибки"


def test_new_tasks_visible_through_api_without_secrets():
    with TestClient(app) as c:
        lst = c.get("/api/v1/code/tasks").json()
        assert len(lst) >= 12
        for row in lst:
            t = c.get(f"/api/v1/code/tasks/{row['id']}").json()
            assert t["max_score"] == 10 and "solution" not in t and "hints" not in t and "correct" not in t["question"]
            assert f"def {get_task(row['id'])['function']}(" in t["starter_code"]


@pytest.mark.parametrize("expr", ["(a != b).mean()", "x.mean()", "x.std()", "x.var()", "x.clip(0, 1).sum()", "float(x.max() - x.min())"])
def test_numpy_methods_that_load_internal_modules_lazily_are_not_blocked(expr):
    """Регрессия: a.mean(), a.std() и др. раньше блокировались сообщением «Импорт numpy._core._methods запрещён»."""
    code = "import numpy as np\ndef hamming_distance(a, b, ma, mb):\n    x = np.asarray(a, float)\n    return float(" + expr + ")\n"
    res = execute("hamming", code, "example")
    assert res["error"] is None and res["example"]["error"] is None, res


@pytest.mark.parametrize("code", ["import numpy._core._methods\n", "from numpy._core import _methods\n", "import numpy.core._methods as m\n"])
def test_explicit_import_of_numpy_internals_is_still_blocked(code):
    from app.sandbox.safety import SafetyError, check_code
    with pytest.raises(SafetyError):
        check_code(code)
