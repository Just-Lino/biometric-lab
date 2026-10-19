from functools import lru_cache

import cv2
import numpy as np

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


def oriented_gabor(img: np.ndarray, phi: np.ndarray, period: float, sigma: float,
                   bins: int = K_BINS) -> np.ndarray:
    """Gabor-фильтрация, где ориентация фильтра в каждом пикселе берётся из поля ориентаций."""
    img = img.astype(np.float32)
    resp = np.stack([cv2.filter2D(img, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
                     for k in gabor_bank(float(period), float(sigma), bins)])
    pos = phi / (np.pi / bins)
    lo = np.floor(pos).astype(int)
    w = (pos - lo).astype(np.float32)
    lo %= bins
    hi = (lo + 1) % bins
    r_lo = np.take_along_axis(resp, lo[None], axis=0)[0]
    r_hi = np.take_along_axis(resp, hi[None], axis=0)[0]
    return (1 - w) * r_lo + w * r_hi  # интерполяция между соседними ориентациями
