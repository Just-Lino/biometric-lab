"""Искажения изображений для проверки устойчивости + базовый сопоставитель + синтетический отпечаток.
Свои алгоритмы подключаются через register_matcher(имя, функция(img_a, img_b) -> 0..1)."""
from __future__ import annotations

import base64
from typing import Callable

import cv2
import numpy as np


class ImagingError(ValueError):
    pass


# ───────────────────────── синтетический отпечаток ─────────────────────────
def synthetic_fingerprint(seed: int = 1, size: int = 192) -> np.ndarray:
    """Учебный «отпечаток»: линии вокруг смещённого центра с плавными искажениями. Не реальная биометрия."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    cx, cy = 0.5 + rng.uniform(-0.12, 0.12), 0.5 + rng.uniform(-0.12, 0.12)
    ax = rng.uniform(0.7, 1.3)
    r = np.sqrt(((x - cx) * ax) ** 2 + ((y - cy) / ax) ** 2)
    warp = np.zeros_like(r)
    for _ in range(4):
        fx, fy = rng.uniform(1, 4, 2)
        warp += rng.uniform(0.01, 0.04) * np.sin(2 * np.pi * (fx * x + fy * y) + rng.uniform(0, 6.28))
    ang = np.arctan2(y - cy, x - cx)
    phase = (r + warp) * rng.uniform(22, 30) + 0.35 * np.sin(ang * rng.integers(1, 3) + rng.uniform(0, 6))
    img = 0.5 + 0.5 * np.cos(2 * np.pi * phase)
    mask = (((x - cx) / 0.42) ** 2 + ((y - cy) / 0.48) ** 2) < 1
    img = np.where(mask, img, 0.5)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return (img * 255).clip(0, 255).astype(np.uint8)


def decode_image(data: bytes, max_side: int = 512) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ImagingError("Не удалось прочитать изображение (PNG/JPG/BMP).")
    h, w = img.shape
    if min(h, w) < 32:
        raise ImagingError("Изображение слишком маленькое.")
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def to_b64_png(img: np.ndarray, side: int = 110) -> str:
    h, w = img.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", small)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ───────────────────────── искажения ─────────────────────────
# имя: (описание, единица, значения от слабого к сильному)
ATTACKS: dict[str, tuple[str, str, list[float]]] = {
    "noise": ("Гауссов шум", "σ", [0, 10, 20, 35, 50, 70]),
    "blur": ("Размытие", "σ, px", [0, 1, 2, 3.5, 5, 7]),
    "rotate": ("Поворот", "градусы", [0, 5, 10, 20, 30, 45]),
    "brightness": ("Смена яркости", "сдвиг", [0, 30, 60, 90, 120, 150]),
    "contrast": ("Потеря контраста", "множитель", [1.0, 0.7, 0.5, 0.35, 0.2, 0.1]),
    "occlusion": ("Закрытая область", "доля площади", [0, 0.1, 0.2, 0.35, 0.5, 0.65]),
    "jpeg": ("Сжатие JPEG", "качество", [95, 60, 35, 20, 10, 4]),
    "downscale": ("Низкое разрешение", "масштаб", [1.0, 0.6, 0.4, 0.25, 0.15, 0.1]),
}


def apply_attack(img: np.ndarray, kind: str, v: float, seed: int = 0) -> np.ndarray:
    h, w = img.shape
    if kind == "noise":
        rng = np.random.default_rng(seed)
        return np.clip(img.astype(np.float32) + rng.normal(0, v, img.shape), 0, 255).astype(np.uint8)
    if kind == "blur":
        return img if v <= 0 else cv2.GaussianBlur(img, (0, 0), v)
    if kind == "rotate":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), v, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if kind == "brightness":
        return np.clip(img.astype(np.float32) + v, 0, 255).astype(np.uint8)
    if kind == "contrast":
        return np.clip((img.astype(np.float32) - 128) * v + 128, 0, 255).astype(np.uint8)
    if kind == "occlusion":
        out = img.copy()
        side = np.sqrt(v * h * w)
        bw, bh = int(min(w, side * 1.2)), int(min(h, side / 1.2))
        x0, y0 = (w - bw) // 2, (h - bh) // 2
        out[y0:y0 + bh, x0:x0 + bw] = 0
        return out
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, v)))])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if kind == "downscale":
        if v >= 1:
            return img
        small = cv2.resize(img, (max(4, int(w * v)), max(4, int(h * v))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ImagingError("Неизвестный тип искажения.")


def shuffled_reference(img: np.ndarray, seed: int = 0, tiles: int = 8) -> np.ndarray:
    """«Чужой» ориентир: те же пиксели, но плитки перемешаны — содержание не совпадает."""
    h, w = img.shape
    th, tw = h // tiles, w // tiles
    parts = [img[r * th:(r + 1) * th, c * tw:(c + 1) * tw] for r in range(tiles) for c in range(tiles)]
    order = np.random.default_rng(seed).permutation(len(parts))
    out = img.copy()
    for k, idx in enumerate(order):
        r, c = divmod(k, tiles)
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = parts[idx]
    return out


# ───────────────────────── сопоставители ─────────────────────────
_HOG = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)


def _hog_vec(img: np.ndarray) -> np.ndarray:
    v = _HOG.compute(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)).ravel().astype(np.float64)
    return v - v.mean()


def matcher_hog(a: np.ndarray, b: np.ndarray) -> float:
    """Базовый учебный сопоставитель: корреляция HOG-признаков. Не устойчив к повороту — это видно на графике."""
    va, vb = _hog_vec(a), _hog_vec(b)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(max(0.0, va @ vb / d)) if d > 0 else 0.0


def matcher_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Нормированная корреляция пикселей (самый простой эталон для сравнения)."""
    a = cv2.resize(a, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float64)
    b = cv2.resize(b, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(max(0.0, (a * b).sum() / d)) if d > 0 else 0.0


MATCHERS: dict[str, tuple[str, Callable[[np.ndarray, np.ndarray], float]]] = {
    "hog": ("HOG-корреляция (базовый)", matcher_hog),
    "ncc": ("Корреляция пикселей (простейший)", matcher_ncc),
}


def register_matcher(key: str, title: str, fn: Callable[[np.ndarray, np.ndarray], float]) -> None:
    """Подключить свой алгоритм: fn(серое_изображение_A, серое_изображение_B) -> оценка сходства 0..1."""
    MATCHERS[key] = (title, fn)


def sweep(img: np.ndarray, kind: str, matcher: str, threshold: float, seed: int = 0, thumbs: bool = True) -> dict:
    if kind not in ATTACKS:
        raise ImagingError("Неизвестный тип искажения.")
    if matcher not in MATCHERS:
        raise ImagingError("Неизвестный сопоставитель.")
    title, unit, values = ATTACKS[kind]
    fn = MATCHERS[matcher][1]
    pts, breakpoint_ = [], None
    for k, v in enumerate(values):
        att = apply_attack(img, kind, v, seed + k)
        s = float(np.clip(fn(img, att), 0, 1))
        if breakpoint_ is None and s < threshold:
            breakpoint_ = v
        pts.append({"value": v, "score": round(s, 4), "pass": s >= threshold,
                    "thumb": to_b64_png(att, 96) if thumbs else None})
    return {"kind": kind, "title": title, "unit": unit, "points": pts, "breakpoint": breakpoint_}
