"""Синтетические дикторы: источник импульсов гортани (высота тона f0) пропускается через резонаторы речевого тракта (форманты).
У каждого диктора свои f0 и длина речевого тракта (масштаб формант), у каждой записи своя «фраза» (последовательность гласных),
уровень шума и небольшой сдвиг высоты тона. Реальные голоса людей не используются: набор нужен, чтобы видеть весь путь от звука до решения."""
from functools import lru_cache

import numpy as np
from scipy.signal import lfilter

from app.modules.voice.audio import SR

N_SPEAKERS, N_RECORDINGS, DURATION = 12, 3, 2.6
VOWELS = [(700, 1220, 2600), (300, 2300, 3000), (350, 800, 2500), (500, 1500, 2500), (450, 1100, 2400), (600, 1800, 2700)]
NOISE = {1: 0.002, 2: 0.003, 3: 0.004}
PHRASE = [(0, 1), (2, 3), (4, 5)]  # все произносят одну и ту же фразу (три слова по две гласных): так сравнивается голос, а не содержание  # запись 1 чистая, 3 самая шумная


def speaker(k: int) -> dict:
    """Параметры диктора k (1..12): высота тона, длина речевого тракта, наклон спектра, «придыхание»."""
    g = lambda a, b: (k * a + b) % 1.0
    return {"f0": 95 + 135 * g(0.618, 0.13), "fscale": 0.86 + 0.34 * g(0.382, 0.55), "breath": 0.002 + 0.008 * g(0.754, 0.3), "tilt": 0.6 + 0.35 * g(0.43, 0.7)}


def _resonator(f: float, bw: float):
    r = np.exp(-np.pi * bw / SR)
    return [1.0], [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r]


def _word(sp: dict, f0: float, n: int, vowels: tuple, rng) -> np.ndarray:
    """Слово из двух гласных: импульсы гортани, пропущенные через три резонатора речевого тракта."""
    out = np.zeros(n)
    for half, vi in zip(np.array_split(np.arange(n), 2), vowels):
        vow = VOWELS[vi]
        e = np.zeros(len(half))
        t = 0.0
        while t < len(half):
            e[int(t)] = 1.0
            t += SR / (f0 * (1 + 0.012 * rng.standard_normal()) * (1 - 0.06 * half[int(t)] / n))  # дрожание и небольшое падение тона к концу слова
        y = e
        for F, bw in zip(vow, (80, 100, 120)):
            b, a = _resonator(F * sp["fscale"] * (1 + 0.006 * rng.standard_normal()), bw)  # каждое произнесение чуть отличается
            y = lfilter(b, a, y)
        y = y / (np.sqrt((y ** 2).mean()) + 1e-9)  # у разных гласных разная звонкость: выравниваем громкость
        out[half] = y * np.hanning(len(half)) ** 0.4
    return out * np.hanning(n) ** 0.5


@lru_cache(maxsize=128)
def recording(k: int, r: int) -> np.ndarray:
    if not (1 <= k <= N_SPEAKERS and 1 <= r <= N_RECORDINGS):
        raise KeyError(f"Неизвестный образец: v{k:02d}_r{r}")
    sp = speaker(k)
    rng = np.random.default_rng(1000 * k + r)
    n = int(SR * DURATION)
    speech = np.zeros(n)
    f0 = sp["f0"] * (1 + 0.015 * (r - 2) + 0.01 * rng.standard_normal())  # каждая запись чуть выше или ниже
    start = 0.25
    for vowels in PHRASE:  # три слова с паузами: так видно, как VAD отделяет речь от тишины
        dur = rng.uniform(0.5, 0.65)
        i0, m = int(start * SR), int(dur * SR)
        speech[i0:i0 + m] = _word(sp, f0, min(m, n - i0), vowels, rng)[:n - i0]
        start += dur + rng.uniform(0.12, 0.2)
    speech = lfilter([1 - sp["tilt"]], [1, -sp["tilt"]], speech)  # наклон спектра: тембр «ярче» или «глуше»
    active = np.abs(speech) > 1e-4
    speech = speech / (np.sqrt((speech[active] ** 2).mean()) + 1e-9) * (0.10 - 0.01 * (r - 1))
    out = speech + (sp["breath"] + NOISE[r]) * rng.standard_normal(n)
    return np.clip(out, -1, 1).astype(np.float32)
