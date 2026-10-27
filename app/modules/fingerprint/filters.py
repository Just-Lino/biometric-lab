from functools import lru_cache

import cv2
import numpy as np
from scipy import fft as sfft

K_BINS = 16


@lru_cache(maxsize=32)
def gabor_bank(period: float, sigma: float, bins: int = K_BINS, gamma: float = 0.7):
    ks = max(9, int(2 * np.ceil(2.5 * sigma) + 1))
    bank = []
    for k in range(bins):
        phi = k * np.pi / bins  # направление гребня
        kern = cv2.getGaborKernel((ks, ks), sigma, phi + np.pi / 2, period, gamma, 0, ktype=cv2.CV_32F)
        kern -= kern.mean()
        kern /= np.abs(kern).sum() + 1e-9
        bank.append(kern)
    return bank


@lru_cache(maxsize=8)
def _bank_fft(period: float, sigma: float, bins: int, nh: int, nw: int):
    """Спектры ядер банка Габора для размера (nh, nw): считаются один раз и переиспользуются для всех образцов."""
    out = []
    for k in gabor_bank(period, sigma, bins):
        kk = np.zeros((nh, nw), np.float32)
        h, w = k.shape
        kk[:h, :w] = k
        out.append(sfft.rfft2(np.roll(kk, (-(h // 2), -(w // 2)), (0, 1))))  # центр ядра в начало координат
    return np.stack(out)


def bank_responses(img: np.ndarray, period: float, sigma: float, bins: int) -> np.ndarray:
    """Отклики всех фильтров банка через FFT (в разы быстрее восьми-шестнадцати cv2.filter2D с большими ядрами).
    Ядра чётные (симметричны относительно центра), поэтому свёртка совпадает с корреляцией filter2D; край отражается как в BORDER_REFLECT."""
    h, w = img.shape
    r = gabor_bank(period, sigma, bins)[0].shape[0] // 2
    nh, nw = sfft.next_fast_len(h + 2 * r, real=True), sfft.next_fast_len(w + 2 * r, real=True)
    buf = np.zeros((nh, nw), np.float32)
    buf[:h + 2 * r, :w + 2 * r] = np.pad(img.astype(np.float32), r, mode="symmetric")
    spec = sfft.rfft2(buf)[None] * _bank_fft(float(period), float(sigma), bins, nh, nw)
    return sfft.irfft2(spec, s=(nh, nw), axes=(1, 2))[:, r:r + h, r:r + w]


def oriented_gabor(img: np.ndarray, phi: np.ndarray, period: float, sigma: float,
                   bins: int = K_BINS) -> np.ndarray:
    """Gabor-фильтрация, где ориентация фильтра в каждом пикселе берётся из поля ориентаций."""
    resp = bank_responses(img, float(period), float(sigma), bins)
    pos = phi / (np.pi / bins)
    lo = np.floor(pos).astype(int)
    w = (pos - lo).astype(np.float32)
    lo %= bins
    hi = (lo + 1) % bins
    r_lo = np.take_along_axis(resp, lo[None], axis=0)[0]
    r_hi = np.take_along_axis(resp, hi[None], axis=0)[0]
    return (1 - w) * r_lo + w * r_hi  # интерполяция между соседними ориентациями
