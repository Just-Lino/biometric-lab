"""Генератор синтетических отпечатков (итеративная Gabor-фильтрация шума по полю ориентаций,
идея SFinGe). Один «палец» = seed; «оттиски» одного пальца = разные искажения одного мастер-образа.
Реальных биометрических данных людей не используется."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from app.modules.fingerprint.filters import oriented_gabor

SIZE = 256
DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "synthetic"
# параметры искажений для оттисков 1..3 (угол, сдвиг, масштаб, шум, размытие)
IMPRESSIONS = {
    1: dict(rot=3, shift=3, scale=0.01, noise=3, blur=0.0),
    2: dict(rot=10, shift=8, scale=0.03, noise=8, blur=0.5),
    3: dict(rot=15, shift=12, scale=0.04, noise=14, blur=0.9),
}


def _orientation_field(rng: np.random.RandomState) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cx, cy = SIZE / 2 + rng.uniform(-18, 18), SIZE * 0.42 + rng.uniform(-15, 15)
    dx, dy = cx + rng.uniform(-60, 60), SIZE * 0.80 + rng.uniform(-10, 10)
    phi = 0.5 * (np.arctan2(yy - cy, xx - cx) - np.arctan2(yy - dy, xx - dx))
    phi += rng.uniform(-0.4, 0.4)
    return np.mod(phi, np.pi).astype(np.float32)


@lru_cache(maxsize=64)
def master(finger: int) -> np.ndarray:
    rng = np.random.RandomState(1000 + finger)
    phi = _orientation_field(rng)
    img = rng.randn(SIZE, SIZE).astype(np.float32)
    for _ in range(6):
        img = oriented_gabor(img, phi, period=9.0 + rng.uniform(-0.5, 0.5) * 0, sigma=4.0)
        img = (img - img.mean()) / (img.std() + 1e-6)
    ridge = np.tanh(1.6 * img)
    gray = np.clip(135 - 115 * ridge, 0, 255)
    # форма пальца: эллипс с мягким краем на светлом фоне
    m = np.zeros((SIZE, SIZE), np.float32)
    cv2.ellipse(m, (SIZE // 2, SIZE // 2 + 4), (92, 116), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    out = 240 * (1 - m) + gray * m
    return out.astype(np.float32)


@lru_cache(maxsize=256)
def impression(finger: int, imp: int) -> np.ndarray:
    """Берёт готовый PNG из datasets/synthetic, а если его нет, генерирует (медленно, ~4 с на палец)."""
    f = DATASET_DIR / f"f{finger:02d}_i{imp}.png"
    if f.exists():
        return cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    return _generate_impression(finger, imp)


def _generate_impression(finger: int, imp: int) -> np.ndarray:
    cfg = IMPRESSIONS[imp]
    rng = np.random.RandomState(finger * 100 + imp)
    s = 1 + rng.uniform(-cfg["scale"], cfg["scale"])
    M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rng.uniform(-cfg["rot"], cfg["rot"]), s)
    M[:, 2] += rng.uniform(-cfg["shift"], cfg["shift"], 2)
    img = cv2.warpAffine(master(finger), M, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=240)
    if cfg["blur"] > 0:
        img = cv2.GaussianBlur(img, (0, 0), cfg["blur"])
    img = (img - 128) * rng.uniform(0.85, 1.05) + 128 + rng.uniform(-10, 10)
    img = img + rng.randn(SIZE, SIZE) * cfg["noise"]
    return np.clip(img, 0, 255).astype(np.uint8)
