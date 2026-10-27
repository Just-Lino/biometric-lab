from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import skeletonize

from app.modules.fingerprint.filters import oriented_gabor
from app.modules.fingerprint.minutiae import Minutia, draw_minutiae, extract_minutiae
from app.modules.fingerprint.params import FingerprintParams


@dataclass
class PipelineResult:
    steps: list[dict] = field(default_factory=list)  # {id,title,description,image}
    minutiae: list[Minutia] = field(default_factory=list)
    mask: np.ndarray | None = None
    gray: np.ndarray | None = None
    enhanced: np.ndarray | None = None  # результат фильтра Габора (нужен методам сопоставления по изображению)


def segment(gray: np.ndarray, p: FingerprintParams) -> np.ndarray:
    f = gray.astype(np.float32)
    mean = cv2.blur(f, (16, 16))
    std = np.sqrt(np.maximum(cv2.blur(f * f, (16, 16)) - mean * mean, 0))
    m = (std > p.seg_threshold).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return m


def orientation_field(img: np.ndarray, smooth: float):
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    gxx, gyy, gxy = (cv2.GaussianBlur(a, (0, 0), smooth) for a in (gx * gx, gy * gy, gx * gy))
    theta_grad = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    phi = np.mod(theta_grad + np.pi / 2, np.pi)  # направление гребня
    coh = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-6)
    return phi.astype(np.float32), np.clip(coh, 0, 1).astype(np.float32)


def _draw_orientation(gray, phi, mask, step=12):
    vis = cv2.cvtColor((gray * 0.55 + 100).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for y in range(step // 2, gray.shape[0], step):
        for x in range(step // 2, gray.shape[1], step):
            if mask[y, x]:
                a = phi[y, x]
                dx, dy = np.cos(a) * step * 0.45, np.sin(a) * step * 0.45
                cv2.line(vis, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), (0, 0, 220), 1, cv2.LINE_AA)
    return vis


def _remove_small(binary: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def _norm_u8(a: np.ndarray, mask=None) -> np.ndarray:
    ref = a[mask > 0] if mask is not None and mask.any() else a
    lo, hi = np.percentile(ref, 1), np.percentile(ref, 99)
    return np.clip((a - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)


def run_pipeline(gray: np.ndarray, p: FingerprintParams, with_images: bool = True) -> PipelineResult:
    res = PipelineResult(gray=gray)
    add = (lambda *a, **k: None)
    if with_images:
        def add(i, t, d, im):
            res.steps.append(dict(id=i, title=t, description=d, image=im))

    add("original", "1. Исходное изображение (grayscale)",
        "Биометрический образ в оттенках серого. Гребни тёмные, межгребневые промежутки светлые. "
        f"Размер {gray.shape[1]}×{gray.shape[0]}, средняя яркость {gray.mean():.0f}.", gray)

    mask = segment(gray, p)
    res.mask = mask
    seg_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_vis, cnts, -1, (0, 200, 0), 2)
    add("segmentation", "2. Сегментация области отпечатка",
        "Область с выраженной текстурой гребней отделяется от фона (по локальной дисперсии). "
        "Дальнейшие признаки ищутся только внутри этой области.", seg_vis)

    clahe = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(8, 8)).apply(gray)
    sel = mask > 0
    mu, sd = clahe[sel].mean(), clahe[sel].std() + 1e-6
    norm = np.clip((clahe.astype(np.float32) - mu) / sd * 55 + 128, 0, 255)
    add("normalized", "3. Нормализация контраста (CLAHE)",
        "Выравнивание яркости и контраста уменьшает влияние освещения и давления пальца.",
        norm.astype(np.uint8))

    phi, coh = orientation_field(norm, p.orient_smooth)
    add("orientation", "4. Поле ориентаций гребней",
        "Красные штрихи показывают локальное направление папиллярных линий. "
        "Оно управляет ориентацией Gabor-фильтра в каждой точке.", _draw_orientation(norm, phi, mask))

    enh = oriented_gabor((norm - 128) / 64.0, phi, p.gabor_period, p.gabor_sigma, bins=8)
    enh_u8 = _norm_u8(enh, mask)
    res.enhanced = enh_u8
    add("enhanced", "5. Усиление гребней фильтром Габора",
        f"Ориентированный Gabor (период {p.gabor_period:g} px, σ={p.gabor_sigma:g}) подавляет шум "
        "и разрывы, усиливая периодическую структуру гребней.", enh_u8)

    if p.bin_method == "otsu":
        t, _ = cv2.threshold(enh_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ridge = enh_u8 < t
    elif p.bin_method == "fixed":
        ridge = enh_u8 < p.bin_threshold
    else:
        ridge = enh_u8.astype(np.int16) < cv2.GaussianBlur(enh_u8, (0, 0), 8).astype(np.int16) - 2
    ridge &= sel
    ridge = _remove_small(ridge, 40)            # убрать мелкие пятна
    ridge = ~_remove_small(~ridge, 25)          # заполнить мелкие дыры
    ridge &= sel
    add("binary", f"6. Бинаризация ({p.bin_method})",
        "Изображение разделяется на гребни (чёрные) и фон. Метод и порог можно менять в эксперименте.",
        np.where(ridge, 0, 255).astype(np.uint8))

    skel = skeletonize(ridge)
    add("skeleton", "7. Скелетизация (thinning)",
        "Гребни утоньшаются до линий толщиной 1 пиксель — так проще найти окончания и разветвления.",
        np.where(skel, 0, 255).astype(np.uint8))

    res.minutiae = extract_minutiae(skel, phi, coh, mask, p)
    n_e = sum(m.type == "ending" for m in res.minutiae)
    n_b = len(res.minutiae) - n_e
    vis = draw_minutiae(np.where(skel, 0, 255).astype(np.uint8), res.minutiae)
    add("minutiae", "8. Выделение minutiae",
        f"● окончание гребня (ridge ending): {n_e}; ▲ разветвление (bifurcation): {n_b}. "
        "Точки у края и слишком близкие друг к другу отбрасываются как ненадёжные.", vis)
    return res
