"""Реальные открытые наборы отпечатков: FVC2002 и FVC2004 (набор «B» каждой из баз DB1-DB4) и SOCOFing.

Файлы лежат в datasets/real/ (см. app/core/realdata.py), в Git не попадают. Идентификаторы образцов:
  fvc2004:DB1:101_3                      - FVC, палец 101, оттиск 3
  soco:100__M:Left_index                  - SOCOFing, реальный отпечаток
  soco:100__M:Left_index:Hard:Obl         - SOCOFing, искажённая копия (уровень и вид искажения)
Все изображения приводятся к кадру 256×256 (так работает остальной конвейер): FVC вырезается по центру в масштабе 1:1,
чтобы не менять период гребней, SOCOFing (около 100×100 пикселей) увеличивается в 2 раза."""
import os
import re
from functools import lru_cache

import cv2
import numpy as np

from app.core import realdata

SIZE = 256
SOCO = "socofing"
FVC_SOURCES = {f"fvc{y}_db{d}": (y, d) for y in (2002, 2004) for d in (1, 2, 3, 4)}
HANDS = {"Left": "левый", "Right": "правый"}
FINGERS = {"thumb": "большой", "index": "указательный", "middle": "средний", "ring": "безымянный", "little": "мизинец"}
LEVELS = ("Easy", "Medium", "Hard")
KINDS = {"Obl": "стирание", "CR": "центральный поворот", "Zcut": "Z-разрез"}
SOCO_SCALE = 2.0
LEVELS_RU = {"Easy": "лёгкое", "Medium": "среднее", "Hard": "сильное"}
# пороги при EER для методов на реальных наборах (замерены на самих наборах, набор B, 10 пальцев × 8 оттисков); шкалы методов разные
THRESHOLDS: dict[str, dict[str, float]] = {
    "fvc2002_db1": {"hough": 0.22, "local": 0.49, "poc": 0.05, "sift": 0.11},
    "fvc2002_db2": {"hough": 0.24, "local": 0.53, "poc": 0.04, "sift": 0.08},
    "fvc2002_db3": {"hough": 0.2, "local": 0.24, "poc": 0.05, "sift": 0.06},
    "fvc2002_db4": {"hough": 0.22, "local": 0.49, "poc": 0.04, "sift": 0.06},
    "fvc2004_db1": {"hough": 0.19, "local": 0.42, "poc": 0.04, "sift": 0.08},
    "fvc2004_db2": {"hough": 0.19, "local": 0.39, "poc": 0.04, "sift": 0.06},
    "fvc2004_db3": {"hough": 0.24, "local": 0.5, "poc": 0.04, "sift": 0.11},
    "fvc2004_db4": {"hough": 0.19, "local": 0.46, "poc": 0.04, "sift": 0.06},
}
_SOCO_ALT = re.compile(r"(\d+__[MF])_(Left|Right)_(thumb|index|middle|ring|little)_finger_(Obl|CR|Zcut)\.bmp", re.I)
_SOCO_REAL = re.compile(r"(\d+__[MF])_(Left|Right)_(thumb|index|middle|ring|little)_finger\.bmp", re.I)


def title(source: str) -> str:
    if source == SOCO:
        return "SOCOFing (реальные отпечатки, 600 человек × 10 пальцев)"
    y, d = FVC_SOURCES[source]
    return f"FVC{y}, база DB{d} (набор B: 10 пальцев × 8 оттисков)"


def _fvc_dir(source: str):
    y, d = FVC_SOURCES[source]
    return realdata.subdir(f"fvc{y}") / f"DB{d}"


def _soco_root():
    base = realdata.subdir(SOCO)
    for cand in (base, base / "SOCOFing", base / "SOCOFing" / "SOCOFing"):
        if (cand / "Real").is_dir():
            return cand
    return None


def is_real_id(sample_id: str) -> bool:
    return sample_id.startswith(("fvc2002:", "fvc2004:", "soco:"))


def is_real_source(source: str | None) -> bool:
    return bool(source) and (source in FVC_SOURCES or source == SOCO)


# ---------- списки ----------
@lru_cache(maxsize=16)
def _fvc_items(path: str) -> tuple:
    out = []
    for f in os.listdir(path):
        m = re.fullmatch(r"(\d+)_(\d+)\.(tif|tiff|bmp|png)", f, re.I)
        if m:
            out.append((int(m[1]), int(m[2])))
    return tuple(sorted(out))


@lru_cache(maxsize=2)
def _soco_items(root: str) -> tuple:
    out = []
    real = os.path.join(root, "Real")
    for pid in sorted(os.listdir(real)):
        d = os.path.join(real, pid)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            m = _SOCO_REAL.fullmatch(f)
            if m:
                out.append((m[1], m[2].capitalize(), m[3].lower()))
    return tuple(out)


@lru_cache(maxsize=2)
def _soco_altered(root: str) -> dict:
    """(человек, рука, палец) -> [(уровень, вид искажения)] по папкам Altered/Altered-Easy|Medium|Hard."""
    kinds = {k.lower(): k for k in KINDS}
    out: dict[tuple, list] = {}
    for level in LEVELS:
        base = os.path.join(root, "Altered", f"Altered-{level}")
        if not os.path.isdir(base):
            continue
        for pid in sorted(os.listdir(base)):
            d = os.path.join(base, pid)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    m = _SOCO_ALT.fullmatch(f)
                    if m:
                        out.setdefault((m[1], m[2].capitalize(), m[3].lower()), []).append((level, kinds[m[4].lower()]))
    return out


def clear_cache() -> None:
    _fvc_items.cache_clear()
    _soco_items.cache_clear()
    _soco_altered.cache_clear()


def source_available(source: str) -> bool:
    if source == SOCO:
        return _soco_root() is not None
    return source in FVC_SOURCES and _fvc_dir(source).is_dir() and bool(_fvc_items(str(_fvc_dir(source))))


def sources_info() -> list[dict]:
    """Доступные реальные наборы (только те, чьи файлы есть на диске)."""
    out = []
    for s in list(FVC_SOURCES) + [SOCO]:
        if not source_available(s):
            continue
        n = len(_soco_items(str(_soco_root()))) + sum(len(v) for v in _soco_altered(str(_soco_root())).values()) if s == SOCO else len(_fvc_items(str(_fvc_dir(s))))
        note = ("Реальные отпечатки, по одному снимку на палец. После каждого оригинала в списке идут его искажённые копии (стирание, центральный "
                "поворот, Z-разрез): выберите оригинал в A и его копию в B, это «свои» пары. Два оригинала разных пальцев - «чужие». "
                "EER не считается (копии не независимы), используйте «Стресс-тест искажений»." if s == SOCO else
                "Реальные оттиски (FVC). Метрики хуже, чем на синтетике: это ожидаемо. Параметры конвейера подобраны под синтетику, при необходимости их стоит менять.")
        out.append({"id": s, "title": title(s), "real": True, "images": n, "note": note, "thresholds": THRESHOLDS.get(s, {})})
    return out


def _soco_label(pid: str, hand: str, finger: str) -> str:
    return f"{pid.replace('__', ' ')}: {HANDS[hand]} {FINGERS[finger]}"


def list_samples(source: str) -> list[dict]:
    if not is_real_source(source) or not source_available(source):
        raise KeyError(f"Набор «{source}» не найден в datasets/real/. Подготовьте данные: python -m scripts.prepare_real")
    if source == SOCO:
        alt = _soco_altered(str(_soco_root()))
        out = []
        for pid, hand, fing in _soco_items(str(_soco_root())):  # оригинал, затем его искажённые копии: пара «свой» всегда рядом
            base = f"soco:{pid}:{hand}_{fing}"
            out.append({"id": base, "label": _soco_label(pid, hand, fing), "person": base, "finger": None, "impression": None})
            for level, kind in sorted(alt.get((pid, hand, fing), []), key=lambda lk: (LEVELS.index(lk[0]), list(KINDS).index(lk[1]))):
                out.append({"id": f"{base}:{level}:{kind}", "label": f"{_soco_label(pid, hand, fing)} - искажение {LEVELS_RU[level]}, {KINDS[kind]}",
                            "person": base, "finger": None, "impression": None})
        return out
    y, d = FVC_SOURCES[source]
    return [{"id": f"fvc{y}:DB{d}:{f}_{i}", "label": f"FVC{y} DB{d}: палец {f}, оттиск {i}", "person": f"{source}:{f}",
             "finger": f, "impression": i} for f, i in _fvc_items(str(_fvc_dir(source)))]


def fvc_fingers(source: str) -> dict[int, list[str]]:
    """Палец -> идентификаторы его оттисков по порядку (для оценки качества)."""
    y, d = FVC_SOURCES[source]
    out: dict[int, list[str]] = {}
    for f, i in _fvc_items(str(_fvc_dir(source))):
        out.setdefault(f, []).append(f"fvc{y}:DB{d}:{f}_{i}")
    return out


def soco_stress_samples(n: int, finger: str = "Right_index") -> list[dict]:
    """Первые n человек, у которых есть реальный отпечаток этого пальца: идентификатор оригинала и его искажённых копий."""
    root = _soco_root()
    if root is None:
        raise KeyError("SOCOFing не найден в datasets/real/. Подготовьте данные: python -m scripts.prepare_real")
    hand, fing = finger.split("_")
    out = []
    for pid, h, f in _soco_items(str(root)):
        if (h, f.lower()) != (hand, fing):
            continue
        alt = {}
        for level in LEVELS:
            for kind in KINDS:
                p = root / "Altered" / f"Altered-{level}" / pid / f"{pid}_{hand}_{fing}_finger_{kind}.BMP"
                if p.exists():
                    alt[(level, kind)] = f"soco:{pid}:{finger}:{level}:{kind}"
        out.append({"pid": pid, "real": f"soco:{pid}:{finger}", "altered": alt})
        if len(out) >= n:
            break
    return out


# ---------- загрузка ----------
def _read(path) -> np.ndarray:
    if not path.exists():
        raise KeyError(f"Неизвестный образец: файл {path.name} не найден")
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение {path.name}")
    return img


def to_canvas(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Кадр 256×256: изображение масштабируется, вырезается по центру (или дополняется фоном по краям)."""
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape
    fill = int(np.median(np.r_[img[0], img[-1], img[:, 0], img[:, -1]]))
    out = np.full((SIZE, SIZE), fill, np.uint8)
    sy, sx = max(0, (h - SIZE) // 2), max(0, (w - SIZE) // 2)
    crop = img[sy:sy + SIZE, sx:sx + SIZE]
    dy, dx = (SIZE - crop.shape[0]) // 2, (SIZE - crop.shape[1]) // 2
    out[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop
    return out


def load(sample_id: str) -> np.ndarray:
    p = sample_id.split(":")
    bad = KeyError(f"Неизвестный образец: {sample_id}")
    if p[0] in ("fvc2002", "fvc2004") and len(p) == 3:
        if p[1] not in ("DB1", "DB2", "DB3", "DB4") or not re.fullmatch(r"\d+_\d+", p[2]):
            raise bad
        folder = realdata.subdir(p[0]) / p[1]
        for ext in (".tif", ".tiff", ".bmp", ".png"):
            if (folder / (p[2] + ext)).exists():
                return to_canvas(_read(folder / (p[2] + ext)))
        raise bad
    if p[0] == "soco" and len(p) in (3, 5):
        root = _soco_root()
        if root is None or not re.fullmatch(r"\d+__[MF]", p[1]) or not re.fullmatch(r"(Left|Right)_(thumb|index|middle|ring|little)", p[2]):
            raise bad
        if len(p) == 3:
            path = root / "Real" / p[1] / f"{p[1]}_{p[2]}_finger.BMP"
        else:
            if p[3] not in LEVELS or p[4] not in KINDS:
                raise bad
            path = root / "Altered" / f"Altered-{p[3]}" / p[1] / f"{p[1]}_{p[2]}_finger_{p[4]}.BMP"
        return to_canvas(_read(path), SOCO_SCALE)
    raise bad
