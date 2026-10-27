"""Реальные снимки радужки: CASIA-Iris-Interval (ближний ИК, 320×280).

Файлы лежат в datasets/real/casia_iris_interval/<человек>/<L|R>/S1001L01.jpg (в Git не попадают, копируются скриптом
`python -m scripts.prepare_real`). Образец - один снимок: casia:S1001L01. Глаз - человек и сторона (S1001L); снимки одного
глаза играют роль «оттисков». Истинных окружностей в наборе нет, поэтому ошибку локализации в пикселях показать нельзя."""
import os
import re
from functools import lru_cache

import cv2
import numpy as np

from app.core import realdata

CASIA = "casia_interval"
TITLE = "CASIA-Iris-Interval (реальные снимки, ближний ИК)"
_NAME = re.compile(r"(S\d\d{3})([LR])(\d{2})\.(jpg|jpeg|png|bmp)", re.I)
_SIDE = {"L": "левый", "R": "правый"}
# пороги при EER для кодировщиков на этом наборе (замерены на первых 12 глазах × 3 снимка); у каждого кодировщика своя шкала
THRESHOLDS: dict[str, dict[str, float]] = {"casia_interval": {"gabor": 0.69, "loggabor": 0.66, "ordinal": 0.81, "dct": 0.01, "haar": 0.74}}


def _root():
    base = realdata.subdir("casia_iris_interval")
    for cand in (base, base / "CASIA-Iris-Interval"):
        if cand.is_dir() and any(c.is_dir() for c in cand.iterdir()):
            return cand
    return None


def is_real_id(sample_id: str) -> bool:
    return sample_id.startswith("casia:")


def is_real_source(source: str | None) -> bool:
    return source == CASIA


def source_available() -> bool:
    return _root() is not None


@lru_cache(maxsize=2)
def _eyes(root: str) -> dict:
    """Глаз (S1001L) -> имена файлов по порядку."""
    out: dict[str, list[tuple[str, str]]] = {}
    for subj in sorted(os.listdir(root)):
        for side in ("L", "R"):
            d = os.path.join(root, subj, side)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                m = _NAME.fullmatch(f)
                if m:
                    out.setdefault(m[1] + m[2].upper(), []).append((f"S{m[1][1:]}{m[2].upper()}{m[3]}", os.path.join(d, f)))
    return out


def clear_cache() -> None:
    _eyes.cache_clear()


def sources_info() -> list[dict]:
    root = _root()
    if root is None:
        return []
    eyes = _eyes(str(root))
    return [{"id": CASIA, "title": TITLE, "real": True, "images": sum(len(v) for v in eyes.values()),
             "note": "Реальные снимки (CASIA-Iris-Interval). Метрики хуже, чем на синтетике: это ожидаемо. Истинных окружностей в наборе нет, "
                     "поэтому ошибка локализации в пикселях не показывается.", "thresholds": THRESHOLDS.get(CASIA, {})}]


def list_samples(source: str = CASIA) -> list[dict]:
    root = _root()
    if source != CASIA or root is None:
        raise KeyError(f"Набор «{source}» не найден в datasets/real/. Подготовьте данные: python -m scripts.prepare_real")
    out = []
    for eye, files in _eyes(str(root)).items():
        side = _SIDE[eye[-1]]
        for k, (stem, _) in enumerate(files, 1):
            out.append({"id": f"casia:{stem}", "label": f"Человек {eye[2:5]}, {side} глаз, снимок {k}", "person": eye})
    return out


def eye_images(min_images: int = 3, n: int = 40) -> dict[str, list[str]]:
    """Первые n глаз, у которых не меньше min_images снимков: глаз -> идентификаторы первых min_images снимков."""
    root = _root()
    if root is None:
        raise KeyError("CASIA не найден в datasets/real/. Подготовьте данные: python -m scripts.prepare_real")
    out = {}
    for eye, files in _eyes(str(root)).items():
        if len(files) >= min_images:
            out[eye] = [f"casia:{stem}" for stem, _ in files[:min_images]]
            if len(out) >= n:
                break
    return out


def load(sample_id: str) -> np.ndarray:
    m = re.fullmatch(r"casia:(S\d{4})([LR])(\d{2})", sample_id)
    root = _root()
    if not m or root is None:
        raise KeyError(f"Неизвестный образец: {sample_id}")
    path = root / m[1][2:] / m[2] / f"{m[1]}{m[2]}{m[3]}.jpg"
    if not path.exists():
        raise KeyError(f"Неизвестный образец: {sample_id}")
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение {path.name}")
    return img
