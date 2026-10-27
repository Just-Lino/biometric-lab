"""Локализация зрачка и радужки: три метода.

daugman - интегро-дифференциальный оператор Даугмана: ищет окружность, на которой средняя яркость меняется резче всего;
hough   - преобразование Хафа для окружностей по границам Кэнни (cv2.HoughCircles);
contour - порог по яркости для зрачка (самая тёмная область) и поиск перепада яркости вдоль лучей для границы радужки.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.modules.iris.params import IrisParams

LOCALIZERS = {
    "daugman": "Интегро-дифференциальный оператор Даугмана",
    "hough": "Преобразование Хафа (окружности)",
    "contour": "Порог + радиальный поиск границы",
}
LATERAL = np.radians(np.r_[np.arange(-40, 41, 3), np.arange(140, 221, 3)])  # боковые дуги: веки их не закрывают


@dataclass
class Circles:
    pupil: tuple[float, float, float]  # x, y, r
    iris: tuple[float, float, float]
    method: str


def preprocess(gray: np.ndarray, p: IrisParams) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), p.blur)


def _dark_seed(img: np.ndarray) -> tuple[float, float, float] | None:
    """Грубое положение зрачка: центр и примерный радиус самой тёмной крупной области ближе к центру кадра."""
    t = np.percentile(img, 4) + 6
    m = (img <= t).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    h, w = img.shape
    best, bs = None, -1e9
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 60:
            continue
        sc = st[i, cv2.CC_STAT_AREA] - 3 * np.hypot(cen[i][0] - w / 2, cen[i][1] - h / 2)
        if sc > bs:
            best, bs = (float(cen[i][0]), float(cen[i][1]), float(np.sqrt(st[i, cv2.CC_STAT_AREA] / np.pi))), sc
    return best


def _circle_means(img, cx, cy, radii, angles):
    xs = cx[:, None, None] + radii[None, :, None] * np.cos(angles)[None, None, :]
    ys = cy[:, None, None] + radii[None, :, None] * np.sin(angles)[None, None, :]
    h, w = img.shape
    xi, yi = np.clip(np.rint(xs).astype(int), 0, w - 1), np.clip(np.rint(ys).astype(int), 0, h - 1)
    return img[yi, xi].mean(axis=2)


def _daugman_search(img, centers, radii, angles, sigma=1.5):
    m = _circle_means(img, centers[:, 0], centers[:, 1], radii, angles)
    d = gaussian_filter1d(np.diff(m, axis=1), sigma, axis=1)  # рост яркости наружу (тёмное -> светлое)
    k, j = np.unravel_index(np.argmax(d), d.shape)
    return float(centers[k, 0]), float(centers[k, 1]), float((radii[j] + radii[j + 1]) / 2)


def _grid(cx, cy, half, step):
    o = np.arange(-half, half + 1, step, dtype=np.float32)
    gx, gy = np.meshgrid(cx + o, cy + o)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def daugman(img: np.ndarray, p: IrisParams) -> Circles | None:
    f = img.astype(np.float32)
    seed = _dark_seed(img)
    if seed is None:
        return None
    h, w = img.shape
    r0 = seed[2]
    pr = _daugman_search(f, _grid(seed[0], seed[1], 10, 2), np.arange(max(6, 0.6 * r0), 1.6 * r0 + 4, 1.0),
                         np.linspace(0, 2 * np.pi, 64, endpoint=False))
    rmax = min(4.5 * pr[2], 0.62 * min(h, w))
    ir = _daugman_search(f, _grid(pr[0], pr[1], 8, 2), np.arange(1.5 * pr[2], rmax, 1.0), LATERAL)
    return Circles(pr, ir, "daugman")


def hough(img: np.ndarray, p: IrisParams) -> Circles | None:
    """Вариант HOUGH_GRADIENT_ALT (OpenCV): точнее классического и даёт мало ложных окружностей."""
    h, w = img.shape
    f = img.astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    no_glint = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))  # убираем блик
    pc = cv2.HoughCircles(no_glint, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=5, param1=80, param2=p.hough_quality,
                          minRadius=10, maxRadius=int(0.3 * min(h, w)))
    dark = np.percentile(img, 5) + 40
    cands = []
    for x, y, r in (pc[0] if pc is not None else []):
        inner = _circle_means(f, np.array([x]), np.array([y]), np.array([0.7 * r]), ang)[0, 0]
        if inner > dark:  # зрачок - тёмный круг
            continue
        outer = _circle_means(f, np.array([x]), np.array([y]), np.array([1.25 * r]), ang)[0, 0]
        cands.append((outer - inner, (float(x), float(y), float(r))))
    if not cands:
        return None
    pr = max(cands)[1]
    ic = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=3, param1=40, param2=max(0.3, p.hough_quality - 0.25),
                          minRadius=int(1.5 * pr[2]), maxRadius=int(min(0.62 * min(h, w), 4.5 * pr[2])))
    near = [c for c in (ic[0] if ic is not None else []) if np.hypot(c[0] - pr[0], c[1] - pr[1]) < 0.4 * pr[2]]
    if not near:
        return None
    lat = lambda c, k: _circle_means(f, np.array([c[0]]), np.array([c[1]]), np.array([k * c[2]]), LATERAL)[0, 0]
    best = max(near, key=lambda c: lat(c, 1.08) - lat(c, 0.92))  # граница радужки: самый сильный перепад на боковых дугах
    return Circles(pr, (float(best[0]), float(best[1]), float(best[2])), "hough")


def _fit_circle(pts: np.ndarray):
    x, y = pts[:, 0], pts[:, 1]
    a = np.c_[2 * x, 2 * y, np.ones(len(x))]
    c, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    return c[0], c[1], np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 1))


def contour(img: np.ndarray, p: IrisParams) -> Circles | None:
    seed = _dark_seed(img)
    if seed is None:
        return None
    f = img.astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    sx, sy, r0 = (np.array([v]) for v in seed)
    lvl_p = _circle_means(f, sx, sy, np.array([0.4 * r0[0]]), ang)[0, 0]
    lvl_i = np.median(_circle_means(f, sx, sy, np.linspace(1.6, 2.2, 4) * r0[0], ang))
    m = (img <= (lvl_p + lvl_i) / 2).astype(np.uint8)  # порог посередине между яркостью зрачка и радужки
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    k = lab[int(seed[1]), int(seed[0])] if lab[int(seed[1]), int(seed[0])] > 0 else 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnt, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cnt, key=cv2.contourArea))  # выпуклая оболочка закрывает «дырку» от блика
    mo = cv2.moments(hull)
    if mo["m00"] < 50:
        return None
    px, py, pr = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"], np.sqrt(mo["m00"] / np.pi)
    h, w = img.shape
    rs = np.arange(1.4 * pr, min(4.5 * pr, 0.62 * min(h, w)), 0.5, dtype=np.float32)
    xs = (px + rs[None, :] * np.cos(LATERAL)[:, None]).astype(np.float32)
    ys = (py + rs[None, :] * np.sin(LATERAL)[:, None]).astype(np.float32)
    prof = cv2.remap(f, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    grad = gaussian_filter1d(prof, 2, axis=1, order=1)  # перепад яркости вдоль каждого луча
    j = grad.argmax(1)
    pts = np.c_[xs[np.arange(len(j)), j], ys[np.arange(len(j)), j]]
    cx, cy, r = _fit_circle(pts)
    res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    keep = res < max(2.0, 2.5 * np.median(res))  # отбрасываем выбросы (ресницы, блики) и уточняем
    if keep.sum() >= 8:
        cx, cy, r = _fit_circle(pts[keep])
    return Circles((float(px), float(py), float(pr)), (float(cx), float(cy), float(r)), "contour")


_FN = {"daugman": daugman, "hough": hough, "contour": contour}


def localize(img_pre: np.ndarray, p: IrisParams, method: str) -> Circles | None:
    c = _FN[method](img_pre, p)
    if c is None:
        return None
    (px, py, pr), (ix, iy, ir) = c.pupil, c.iris
    if not (ir > pr * 1.2 and np.hypot(px - ix, py - iy) < 0.6 * pr):  # проверка правдоподобия
        return None
    return c


def resolve(p: IrisParams) -> str:
    return "daugman" if p.localizer == "auto" else p.localizer
