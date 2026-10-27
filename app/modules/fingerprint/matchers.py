"""Методы сопоставления отпечатков.

hough - минуции + голосование за поворот и сдвиг (основной метод лаборатории);
local - минуции + локальные структуры соседей: не требует общего совмещения, устойчив к поворотам;
poc   - фазовая корреляция (Phase-Only Correlation) усиленных изображений с перебором поворотов;
sift  - ключевые точки SIFT на снимке с выровненным контрастом + RANSAC (универсальный метод, не специфичный для отпечатков);
cnn   - нейросеть, обученная вами на вкладке «Обучение распознаванию»: косинусное сходство embedding.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.modules.fingerprint.matching import MatchResult, _wrap_pi, draw_match, match_minutiae
from app.modules.fingerprint.minutiae import Minutia
from app.modules.fingerprint.params import FingerprintParams

MATCHERS = {
    "hough": "Минуции + голосование за поворот и сдвиг",
    "local": "Минуции + локальные структуры соседей",
    "poc": "Фазовая корреляция (POC) с перебором поворотов",
    "sift": "Ключевые точки SIFT + RANSAC",
    "cnn": "Нейросеть, обученная вами (вкладка «Обучение»)",
}
# рекомендуемые пороги (по EER на синтетическом наборе лаборатории); шкалы методов разные
DEFAULT_THRESHOLDS = {"hough": 0.50, "local": 0.55, "poc": 0.13, "sift": 0.16, "cnn": 0.80}


def recommended_threshold(name: str, source: str | None = None) -> float:
    """Для сети - порог по EER на снимках, которых она не видела; для остальных методов - заранее измеренный (на реальных наборах свой)."""
    if name == "cnn":
        from app.training import registry
        m = registry.get("fingerprint")
        if m:
            return m.threshold
    if source:
        from app.modules.fingerprint import real
        t = real.THRESHOLDS.get(source, {}).get(name)
        if t is not None:
            return t
    return DEFAULT_THRESHOLDS[name]


@dataclass
class FpMatch:
    score: float
    n_a: int = 0
    n_b: int = 0
    matched: int = 0
    rotation_deg: float = 0.0
    shift: tuple[float, float] = (0.0, 0.0)
    overlay: np.ndarray | None = None
    details: list = field(default_factory=list)


# ---------- локальные структуры минуций ----------
def _descriptors(ms: list[Minutia], k: int = 5):
    pts = np.array([[m.x, m.y] for m in ms], np.float32).reshape(-1, 2)
    th = np.array([m.theta for m in ms], np.float32)
    out = []
    for i in range(len(ms)):
        d = np.hypot(*(pts - pts[i]).T)
        d[i] = 1e9
        nb = np.argsort(d)[:k]
        ang = np.arctan2(pts[nb, 1] - pts[i, 1], pts[nb, 0] - pts[i, 0])
        out.append((d[nb], _wrap_pi(ang - th[i]), _wrap_pi(th[nb] - th[i])))  # расстояние, направление на соседа и разность ориентаций
    return out


def match_local(a: list[Minutia], b: list[Minutia], p: FingerprintParams, k: int = 5) -> MatchResult:
    if len(a) < 3 or len(b) < 3:
        return MatchResult()
    da, db = _descriptors(a, k), _descriptors(b, k)
    dt, at = p.dist_tol * 0.6, np.radians(p.angle_tol_deg)
    cand = []
    for i, (d1, a1, t1) in enumerate(da):
        for j, (d2, a2, t2) in enumerate(db):
            if a[i].type != b[j].type:
                continue
            ok = ((np.abs(d1[:, None] - d2[None, :]) < dt) & (np.abs(_wrap_pi(a1[:, None] - a2[None, :])) < at)
                  & (np.abs(_wrap_pi(t1[:, None] - t2[None, :])) < at))
            used, s = set(), 0
            for r in range(len(d1)):  # каждому соседу из A - не более одного соседа из B
                for c in np.where(ok[r])[0]:
                    if c not in used:
                        used.add(c)
                        s += 1
                        break
            if s >= 3:
                cand.append((s / k, i, j))
    cand.sort(reverse=True)
    ua, ub, pairs = set(), set(), []
    for s, i, j in cand:  # одна минуция - одна пара
        if i not in ua and j not in ub:
            ua.add(i), ub.add(j), pairs.append((i, j))
    rot = 0.0
    if pairs:
        rot = float(np.degrees(np.median([_wrap_pi(b[j].theta - a[i].theta) for i, j in pairs])))
    return MatchResult(2 * len(pairs) / (len(a) + len(b)), pairs, rot, (0.0, 0.0))


# ---------- фазовая корреляция ----------
POC_N = 128
ANGLES = np.arange(-40, 41, 4)


def _prep(enh: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = cv2.resize(enh.astype(np.float32), (POC_N, POC_N), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.float32), (POC_N, POC_N), interpolation=cv2.INTER_AREA)
    v = img[m > 0.5] if (m > 0.5).any() else img.ravel()
    img = (img - v.mean()) / (v.std() + 1e-6) * m
    return img * np.outer(np.hanning(POC_N), np.hanning(POC_N)).astype(np.float32)


_fy, _fx = np.meshgrid(np.fft.fftfreq(POC_N), np.fft.fftfreq(POC_N), indexing="ij")
_BAND = ((np.hypot(_fx, _fy) > 0.06) & (np.hypot(_fx, _fy) < 0.32)).astype(np.float32)  # полоса частот гребней


def _spectra(res) -> dict:
    """Спектры изображения при каждом повороте (считаются один раз на образец)."""
    if getattr(res, "_poc", None) is None:
        base = _prep(res.enhanced, res.mask)
        c = (POC_N / 2, POC_N / 2)
        res._poc = {int(a): np.fft.fft2(cv2.warpAffine(base, cv2.getRotationMatrix2D(c, float(a), 1.0), (POC_N, POC_N))) for a in ANGLES}
        res._poc_img = base
    return res._poc


def match_poc(ra, rb) -> tuple[float, float, tuple[float, float]]:
    fa = _spectra(ra)[0]
    best = (0.0, 0.0, (0.0, 0.0))
    for ang, fb in _spectra(rb).items():
        r = fa * np.conj(fb)
        r = r / (np.abs(r) + 1e-9) * _BAND  # оставляем только фазу в полосе частот гребней
        c = np.real(np.fft.ifft2(r))
        k = np.unravel_index(np.argmax(c), c.shape)
        if c[k] > best[0]:
            dy, dx = (k[0] if k[0] < POC_N // 2 else k[0] - POC_N), (k[1] if k[1] < POC_N // 2 else k[1] - POC_N)
            best = (float(c[k]), float(ang), (dx * 256 / POC_N, dy * 256 / POC_N))
    return best


# ---------- SIFT ----------
def _sift(res):
    if getattr(res, "_sift", None) is None:
        sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.02)
        m = cv2.erode((res.mask > 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        # на сглаженном результате фильтра Габора SIFT находит около 5 точек, поэтому берётся снимок с выровненным контрастом
        kp, des = sift.detectAndCompute(cv2.createCLAHE(3.0, (4, 4)).apply(res.gray), m)
        res._sift = (kp, des)
    return res._sift


def match_sift(ra, rb) -> tuple[float, int, int, int, np.ndarray | None, list]:
    (ka, da), (kb, db) = _sift(ra), _sift(rb)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0.0, len(ka or []), len(kb or []), 0, None, []
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 4:
        return 0.0, len(ka), len(kb), 0, None, good
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    n_in = int(inl.sum()) if inl is not None else 0
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))) if M is not None else 0.0
    return min(1.0, n_in / 40), len(ka), len(kb), n_in, (M, inl, good, rot), good


def _side_by_side(ga, gb):
    return np.hstack([cv2.cvtColor(ga, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gb, cv2.COLOR_GRAY2BGR)])


def compare(name: str, ra, rb, p: FingerprintParams) -> FpMatch:
    """ra, rb - результаты конвейера (PipelineResult). Возвращает оценку сходства 0..1 и данные для визуализации."""
    if name in ("hough", "local"):
        m = (match_minutiae(ra.minutiae, rb.minutiae, dist_tol=p.dist_tol, angle_tol_deg=p.angle_tol_deg, max_rotation_deg=p.max_rotation_deg)
             if name == "hough" else match_local(ra.minutiae, rb.minutiae, p))
        return FpMatch(round(float(m.score), 4), len(ra.minutiae), len(rb.minutiae), len(m.matched), m.rotation_deg, m.shift,
                       draw_match(ra.gray, rb.gray, ra.minutiae, rb.minutiae, m),
                       [["Minutiae A", len(ra.minutiae)], [ "Minutiae B", len(rb.minutiae)], ["Совпало", len(m.matched)],
                        ["Поворот", f"{m.rotation_deg:.1f}°"], ["Сдвиг", f"{m.shift[0]:.1f}, {m.shift[1]:.1f} px"]])
    if name == "cnn":
        from app.training import registry
        m = registry.require("fingerprint", "отпечатков пальцев", p.source)
        ea, eb = registry.embed("fingerprint", ra.gray), registry.embed("fingerprint", rb.gray)
        cos = float(ea @ eb)
        over = _side_by_side(ra.gray, rb.gray)
        return FpMatch(round(max(0.0, cos), 4), 0, 0, 0, 0.0, (0.0, 0.0), over,
                       [["Косинусное сходство embedding", f"{cos:.3f}"], ["Размерность embedding", len(ea)],
                        ["Сеть", f"{m.epochs} эпох, точность на новых снимках {m.test_acc * 100:.0f} %, {m.config.get('arch', 'small')}, {m.config.get('optimizer', 'adam')}"],
                        ["Рекомендуемый порог", f"{m.threshold:.2f} (по EER на снимках, которых сеть не видела)"]])
    if name == "poc":
        peak, ang, (dx, dy) = match_poc(ra, rb)
        c = (128.0, 128.0)
        M = cv2.getRotationMatrix2D(c, -ang, 1.0)
        M[:, 2] += (-dx, -dy)
        aligned = cv2.warpAffine(rb.gray, cv2.getRotationMatrix2D(c, ang, 1.0), (256, 256), borderValue=240)
        aligned = np.roll(aligned, (int(round(dy)), int(round(dx))), (0, 1))
        over = np.hstack([cv2.cvtColor(ra.gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR),
                          cv2.merge([ra.gray, aligned, aligned])])  # третий кадр: A красным, B после совмещения циановым
        return FpMatch(round(min(1.0, peak), 4), 0, 0, 0, ang, (dx, dy), over,
                       [["Пик корреляции", f"{peak:.3f}"], ["Лучший поворот", f"{ang:.0f}°"], ["Сдвиг пика", f"{dx:.1f}, {dy:.1f} px"],
                        ["Проверено поворотов", len(ANGLES)]])
    if name == "sift":
        score, na, nb, n_in, extra, good = match_sift(ra, rb)
        over = _side_by_side(ra.gray, rb.gray)
        rot = 0.0
        if extra is not None:
            M, inl, good, rot = extra
            ka, kb = _sift(ra)[0], _sift(rb)[0]
            for g, ok in zip(good, inl.ravel()):
                xa, ya = ka[g.queryIdx].pt
                xb, yb = kb[g.trainIdx].pt
                cv2.line(over, (int(xa), int(ya)), (int(xb) + 256, int(yb)), (0, 200, 0) if ok else (160, 160, 160), 1, cv2.LINE_AA)
        return FpMatch(round(min(1.0, score), 4), na, nb, n_in, rot, (0.0, 0.0), over,
                       [["Ключевых точек A", na], ["Ключевых точек B", nb], ["Хороших пар (тест отношения)", len(good)], ["Согласованных (RANSAC)", n_in],
                        ["Поворот", f"{rot:.1f}°"]])
    raise ValueError(name)
