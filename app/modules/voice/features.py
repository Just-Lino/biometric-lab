"""Шаблон голоса и сходство. Шаблон - вектор чисел по всей речи записи; перед сравнением он приводится к z-оценкам по «популяции»
(средние и разброс по синтетическим дикторам), чтобы признаки разного масштаба вносили сравнимый вклад. Сходство = exp(-средний квадрат разности z-векторов)."""
from functools import lru_cache

import numpy as np

from app.modules.voice import synthetic
from app.modules.voice.audio import analyze

METHODS = {
    "mfcc": "MFCC: средние и разброс спектральной огибающей (тембр)",
    "mfcc_delta": "MFCC + динамика (дельта-коэффициенты)",
    "pitch": "Высота тона (f0): медиана, диапазон, разброс",
    "combo": "MFCC + высота тона",
}
# рекомендуемые пороги (по EER на синтетических дикторах); шкалы методов разные
DEFAULT_THRESHOLDS = {"mfcc": 0.50, "mfcc_delta": 0.40, "pitch": 0.65, "combo": 0.45}


def _pitch_part(a: dict) -> np.ndarray:
    m = a["mask"]
    f = a["f0"][m]
    v = f[f > 0]
    if len(v) < 5:
        raise ValueError("В записи слишком мало озвученных кадров: высоту тона определить не удалось. Говорите протяжнее и громче.")
    lv = np.log(v)
    return np.array([np.median(lv), np.percentile(lv, 10), np.percentile(lv, 90), lv.std()])  # только высота тона: медиана, диапазон, разброс


def template(a: dict, method: str) -> np.ndarray:
    m = a["mask"]
    mf = a["mfcc"][m]
    base = np.r_[mf.mean(axis=0), mf.std(axis=0)]
    if method == "mfcc":
        return base
    if method == "mfcc_delta":
        d = a["dmfcc"][m]
        return np.r_[base, d.mean(axis=0), d.std(axis=0)]
    if method == "pitch":
        return _pitch_part(a)
    if method == "combo":
        return np.r_[base, _pitch_part(a)]
    raise ValueError(f"Неизвестный метод: {method}")


@lru_cache(maxsize=32)
def population(method: str, vad: str, vad_level: float, n_mfcc: int) -> tuple[np.ndarray, np.ndarray]:
    vs = []
    for k in range(1, synthetic.N_SPEAKERS + 1):
        for r in range(1, synthetic.N_RECORDINGS + 1):
            vs.append(template(analyze(synthetic.recording(k, r), vad, vad_level, n_mfcc), method))
    V = np.stack(vs)
    return V.mean(axis=0), np.maximum(V.std(axis=0), 1e-6)


def zvec(v: np.ndarray, method: str, vad: str, vad_level: float, n_mfcc: int) -> np.ndarray:
    mu, sd = population(method, vad, vad_level, n_mfcc)
    return (v - mu) / sd


def similarity(za: np.ndarray, zb: np.ndarray) -> float:
    """Сходство 0..1: exp(-d²), где d² - средний квадрат разности z-векторов (одинаковые шаблоны дают 1, независимые около 0,15)."""
    return float(np.exp(-np.mean((za - zb) ** 2)))
