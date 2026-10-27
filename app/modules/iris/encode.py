"""Нормализация Даугмана (rubber sheet) и кодирование радужки: четыре метода."""
from dataclasses import dataclass

import cv2
import numpy as np
from functools import lru_cache

from app.modules.iris.localize import Circles
from app.modules.iris.params import IrisParams

RAD, ANG = 32, 256  # размер нормализованной развёртки: 32 кольца × 256 направлений
ENCODERS = {
    "gabor": "2D-фильтры Габора → Iris Code (Даугман)",
    "loggabor": "1D лог-Габор по кольцам → Iris Code (Masek)",
    "ordinal": "Порядковые признаки (ordinal measures) → код",
    "dct": "Косинусное преобразование окон (DCT, Monro) → код",
    "haar": "Вейвлет Хаара → знаковый код",
    "cnn": "Нейросеть, обученная вами (вкладка «Обучение»)",  # не бинарный код: embedding сети и косинусное сходство
}


def normalize(gray: np.ndarray, c: Circles):
    """Rubber sheet: точка (r, θ) = (1-r)·точка зрачка + r·точка радужки по направлению θ. Возвращает развёртку и
    маску точек, попавших внутрь кадра, и координаты точек развёртки на исходном изображении."""
    th = np.linspace(0, 2 * np.pi, ANG, endpoint=False)
    r = np.linspace(0, 1, RAD + 2)[1:-1][:, None]  # без самого края зрачка и лимба
    (px, py, pr), (ix, iy, ir) = c.pupil, c.iris
    xp, yp = px + pr * np.cos(th), py + pr * np.sin(th)
    xi, yi = ix + ir * np.cos(th), iy + ir * np.sin(th)
    mx = ((1 - r) * xp + r * xi).astype(np.float32)
    my = ((1 - r) * yp + r * yi).astype(np.float32)
    strip = cv2.remap(gray.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = (strip >= 0) & (mx >= 0) & (mx < gray.shape[1]) & (my >= 0) & (my < gray.shape[0])
    return np.clip(strip, 0, 255), valid, (mx, my)


def occlusion_mask(strip: np.ndarray, valid: np.ndarray, on: bool) -> np.ndarray:
    """Маска «хороших» точек: исключаются веки (светлее радужки), ресницы (темнее) и блики.
    Статистика радужки берётся с боковых секторов, которые веки обычно не закрывают."""
    if not on:
        return valid.copy()
    b = cv2.GaussianBlur(strip, (0, 0), 2)
    lat = np.r_[np.arange(0, ANG // 10), np.arange(ANG // 2 - ANG // 10, ANG // 2 + ANG // 10), np.arange(ANG - ANG // 10, ANG)]
    ref = b[:, lat][valid[:, lat]]
    if ref.size < 50:
        return valid.copy()
    mu, sd = np.median(ref), 1.4826 * np.median(np.abs(ref - np.median(ref))) + 1e-3
    m = valid & (np.abs(b - mu) < 3.0 * sd) & (strip < 235)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return m


@dataclass
class Template:
    kind: str  # binary
    code: np.ndarray  # (планы, строки, столбцы) bool
    mask: np.ndarray  # маска надёжных битов той же формы
    width: int  # число столбцов кода (для пересчёта сдвига)
    response: np.ndarray  # для визуализации: отклик первого фильтра


def _zscore(strip, mask):
    v = strip[mask] if mask.any() else strip.ravel()
    return ((strip - v.mean()) / (v.std() + 1e-6)).astype(np.float32)


def _wrap_filter(img, kern):
    pad = kern.shape[1]
    ext = np.concatenate([img[:, -pad:], img, img[:, :pad]], 1)  # угол замкнут: развёртка циклична по θ
    return cv2.filter2D(ext, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT)[:, pad:-pad]


def _gabor_kernels(lam):
    kx = int(2 * round(lam) + 1)
    x = np.arange(kx, dtype=np.float32) - kx // 2
    y = np.arange(9, dtype=np.float32) - 4
    xx, yy = np.meshgrid(x, y)
    env = np.exp(-(xx ** 2) / (2 * (lam / 2.5) ** 2) - yy ** 2 / (2 * 2.5 ** 2))
    re, im = env * np.cos(2 * np.pi * xx / lam), env * np.sin(2 * np.pi * xx / lam)
    return (re - re.mean()).astype(np.float32), im.astype(np.float32)  # вещественная часть без постоянной составляющей


def _bits(re, im, mask, frac=0.08):
    """Квантование фазы в 2 бита (знаки вещественной и мнимой частей); слабые отклики - ненадёжные биты, в маску не идут."""
    tr, ti = frac * np.abs(re).max(), frac * np.abs(im).max()
    return np.stack([re > 0, im > 0]), np.stack([mask & (np.abs(re) > tr), mask & (np.abs(im) > ti)])


def encode(strip: np.ndarray, mask: np.ndarray, encoder: str, p: IrisParams) -> Template:
    z = _zscore(strip, mask)
    if encoder == "gabor":
        codes, masks, resp = [], [], None
        for lam in (p.wavelength, 2 * p.wavelength):
            kr, ki = _gabor_kernels(lam)
            re, im = _wrap_filter(z, kr), _wrap_filter(z, ki)
            resp = re if resp is None else resp
            c, m = _bits(re, im, mask)
            codes.append(c)
            masks.append(m)
        return Template("binary", np.concatenate(codes), np.concatenate(masks), ANG, resp)
    if encoder == "loggabor":
        f = np.fft.fftfreq(ANG)
        f0 = 1.0 / p.wavelength
        g = np.zeros(ANG)
        pos = f > 0
        g[pos] = np.exp(-(np.log(f[pos] / f0) ** 2) / (2 * np.log(0.55) ** 2))
        resp = np.fft.ifft(np.fft.fft(z, axis=1) * g[None, :], axis=1)  # аналитический сигнал по каждому кольцу
        c, m = _bits(resp.real.astype(np.float32), resp.imag.astype(np.float32), mask)
        return Template("binary", c, m, ANG, resp.real.astype(np.float32))
    if encoder == "dct":
        # окна 8×16 с шагом 4 по углу; у каждого окна - 6 низкочастотных коэффициентов DCT; бит = знак разницы одного
        # коэффициента у окна и у окна на 16 px правее (развёртка замкнута по углу)
        ph, pw, st = 8, 16, 4
        ext, mext = np.concatenate([z, z[:, :pw + st]], 1), np.concatenate([mask, mask[:, :pw + st]], 1).astype(np.float32)
        zz = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (0, 3)]
        nr, nc = RAD // ph, ANG // st
        co = np.zeros((nr, nc + pw // st, len(zz)), np.float32)
        ok = np.zeros((nr, nc + pw // st), np.float32)
        for i in range(nr):
            for j in range(nc + pw // st):
                win = ext[i * ph:(i + 1) * ph, j * st:j * st + pw]
                d = cv2.dct(np.ascontiguousarray(win))
                co[i, j] = [d[a, b] for a, b in zz]
                ok[i, j] = mext[i * ph:(i + 1) * ph, j * st:j * st + pw].mean()
        diff = co[:, pw // st:pw // st + nc, :] - co[:, :nc, :]
        good = (ok[:, pw // st:pw // st + nc] > 0.7) & (ok[:, :nc] > 0.7)
        code = np.moveaxis(diff > 0, 2, 0)
        m = np.moveaxis(np.broadcast_to(good[:, :, None], diff.shape) & (np.abs(diff) > 0.1 * np.abs(diff).max()), 2, 0)
        return Template("binary", code, m, nc, cv2.resize(diff[:, :, 0], (ANG, RAD), interpolation=cv2.INTER_NEAREST))
    if encoder == "haar":
        a = z.copy()
        mk = mask.copy()
        planes, pmask = [], []
        for _ in range(2):  # два уровня вейвлета Хаара
            ll = (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) / 4
            lh = (a[0::2, 0::2] + a[0::2, 1::2] - a[1::2, 0::2] - a[1::2, 1::2]) / 4
            hl = (a[0::2, 0::2] - a[0::2, 1::2] + a[1::2, 0::2] - a[1::2, 1::2]) / 4
            mk = mk[0::2, 0::2] & mk[0::2, 1::2] & mk[1::2, 0::2] & mk[1::2, 1::2]
            a = ll
        for d in (lh, hl):  # детали второго уровня: знак - бит, малые по модулю - ненадёжные
            planes.append(d > 0)
            pmask.append(mk & (np.abs(d) > 0.08 * np.abs(d).max()))
        return Template("binary", np.stack(planes), np.stack(pmask), planes[0].shape[1], cv2.resize(lh, (ANG, RAD), interpolation=cv2.INTER_NEAREST))
    if encoder == "ordinal":
        planes, pmask, resp = [], [], None
        for k in _ordinal_kernels():  # знак разности двух соседних областей: «где светлее»
            r = _wrap_filter(z, k)
            resp = r if resp is None else resp
            planes.append(r > 0)
            pmask.append(mask & (np.abs(r) > 0.08 * np.abs(r).max()))
        return Template("binary", np.stack(planes), np.stack(pmask), ANG, resp)
    raise ValueError(encoder)


@lru_cache(maxsize=1)
def _ordinal_kernels():
    """Двухлепестковые фильтры Сунь и Тань: разность двух гауссиан, сдвинутых по углу и по радиусу."""
    yy, xx = (np.mgrid[0:25, 0:25] - 12).astype(np.float32)
    g = lambda dx, dy: np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * 3.0 ** 2))
    return [(g(4, 0) - g(-4, 0)).astype(np.float32), (g(0, 4) - g(0, -4)).astype(np.float32),
            (g(7, 0) - g(-7, 0)).astype(np.float32)]


def match(a: Template, b: Template, max_shift: int) -> dict:
    """Сравнение Iris Code с перебором циклических сдвигов (поворот глаза): доля несовпавших битов (расстояние
    Хэмминга) по битам, надёжным в обоих кодах; сходство = 1 - HD."""
    s_max = int(round(max_shift * a.width / ANG))
    best = None
    for s in range(-s_max, s_max + 1):
        cb, mb = np.roll(b.code, s, axis=2), np.roll(b.mask, s, axis=2)
        m = a.mask & mb
        n = int(m.sum())
        if n < 50:
            continue
        hd = float(((a.code ^ cb) & m).sum() / n)
        if best is None or 1 - hd > best["score"]:
            best = {"score": 1 - hd, "hd": hd, "shift": s, "valid": n / m.size}
    if best is None:
        return {"score": 0.0, "hd": 1.0, "shift": 0, "valid": 0.0, "shift_deg": 0.0}
    best["shift_deg"] = round(best["shift"] * 360 / a.width, 1)
    return best
