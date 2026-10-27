"""Синтетические изображения глаза (как в ближнем ИК-диапазоне, оттенки серого).

У каждого глаза своя текстура радужки (радиальные волокна, крипты, коллеретка). Оттиски одного глаза различаются
расширением зрачка, положением, поворотом, закрытием веками, шумом и размытием; оттиск 3 самый сложный.
Генератор знает истинные окружности зрачка и радужки, поэтому точность локализации можно измерить.
Реальные биометрические данные людей не используются."""
from functools import lru_cache
from pathlib import Path
import json

import cv2
import numpy as np

H, W = 240, 320
N_EYES, N_IMPRESSIONS = 12, 3
DATA = Path(__file__).resolve().parents[3] / "datasets" / "synthetic_iris"
TEX_R, TEX_A = 64, 512


@lru_cache(maxsize=64)
def texture(eye: int) -> np.ndarray:
    rng = np.random.RandomState(1000 + eye)
    t = rng.randn(TEX_R, TEX_A).astype(np.float32)
    pad = 40
    t = np.concatenate([t[:, -pad:], t, t[:, :pad]], 1)  # угол замкнут по кругу
    t = cv2.GaussianBlur(t, (0, 0), sigmaX=1.4, sigmaY=4.5)[:, pad:-pad]  # волокна вытянуты по радиусу
    rr = np.linspace(0, 1, TEX_R, dtype=np.float32)[:, None]
    aa = np.arange(TEX_A, dtype=np.float32)[None, :]
    t = t / (t.std() + 1e-6)
    for _ in range(rng.randint(8, 14)):  # крипты - тёмные пятна
        r0, a0 = rng.uniform(0.25, 0.9), rng.uniform(0, TEX_A)
        da = np.minimum(np.abs(aa - a0), TEX_A - np.abs(aa - a0))
        t -= rng.uniform(1.0, 2.0) * np.exp(-((rr - r0) / 0.06) ** 2 - (da / rng.uniform(6, 14)) ** 2)
    k = rng.randint(8, 15)  # коллеретка - зубчатое кольцо около трети радиуса
    t += 1.2 * np.exp(-((rr - rng.uniform(0.25, 0.35)) / 0.04) ** 2) * (1 + 0.6 * np.sin(2 * np.pi * k * aa / TEX_A + rng.uniform(0, 6)))
    return ((t - t.mean()) / (t.std() + 1e-6)).astype(np.float32)


def _eye(eye: int) -> dict:
    r = np.random.RandomState(eye)
    return {"R": r.uniform(66, 82), "ratio": r.uniform(0.32, 0.44), "mean": r.uniform(100, 130), "contrast": r.uniform(18, 26)}


def render(eye: int, imp: int) -> tuple[np.ndarray, dict]:
    b, rng = _eye(eye), np.random.RandomState(eye * 100 + imp)
    lv = {1: 0.25, 2: 0.6, 3: 1.0}.get(imp, 0.6)
    R = b["R"] * (1 + rng.uniform(-0.04, 0.04) * lv)
    rp = R * float(np.clip(b["ratio"] + rng.uniform(-0.07, 0.07) * lv, 0.25, 0.55))  # расширение зрачка
    ci = np.array([W / 2 + rng.uniform(-16, 16) * lv, H / 2 + rng.uniform(-10, 10) * lv])
    cp = ci + rng.uniform(-3.5, 3.5, 2)  # центр зрачка не совпадает с центром радужки
    rot = rng.uniform(-10, 10) * lv  # поворот глаза, градусы
    s = 2  # рисуем в двойном разрешении и уменьшаем: гладкие края
    yy, xx = np.mgrid[0:H * s, 0:W * s].astype(np.float32) / s
    dx, dy = xx - cp[0], yy - cp[1]
    rho, th = np.hypot(dx, dy), np.arctan2(dy, dx)
    ux, uy = np.cos(th), np.sin(th)
    d = cp - ci
    bb = d[0] * ux + d[1] * uy
    t_edge = -bb + np.sqrt(np.maximum(bb * bb - (d @ d - R * R), 0))  # расстояние от центра зрачка до границы радужки по лучу
    rn = (rho - rp) / np.maximum(t_edge - rp, 1e-3)
    # каждый снимок: свой шум текстуры (освещение, фокус) и нелинейное растяжение радужки при изменении зрачка
    tex = texture(eye) + 0.45 * (0.5 + lv) * cv2.GaussianBlur(rng.randn(TEX_R, TEX_A).astype(np.float32), (0, 0), 1.2)
    warp = rng.uniform(-0.08, 0.08) * (0.5 + lv)
    rw = np.clip(rn + warp * np.sin(np.pi * np.clip(rn, 0, 1)), 0, 1)
    mx = (((th - np.radians(rot)) % (2 * np.pi)) / (2 * np.pi) * TEX_A).astype(np.float32)
    my = (rw * (TEX_R - 1)).astype(np.float32)
    iris_val = b["mean"] + b["contrast"] * cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    iris_val *= 1 - 0.22 * np.clip(rn, 0, 1) ** 6  # тёмный лимб у края
    img = 196 - 10 * (yy / H) + rng.randn(*xx.shape) * 1.5  # склера
    img = np.where((rn >= 0) & (rn <= 1), iris_val, img)
    img = np.where(rho < rp, 22, img)  # зрачок
    hl = cp + np.array([-0.35, -0.35]) * rp
    img = np.where(np.hypot(xx - hl[0], yy - hl[1]) < 4.5, 248, img)  # блик
    # веки: верхнее и нижнее (парабола); чем сложнее оттиск, тем сильнее закрыта радужка
    au = R * (rng.uniform(0.92, 1.15) - 0.4 * lv * rng.rand())  # верхнее веко: в сложных оттисках закрывает верх радужки
    al = R * (rng.uniform(1.0, 1.25) - 0.3 * lv * rng.rand())
    span = 1.9 * R
    yu = ci[1] - au + au * ((xx - ci[0]) / span) ** 2
    yl = ci[1] + al - al * ((xx - ci[0]) / span) ** 2
    skin = (yy < yu) | (yy > yl)
    img = np.where(skin, 168 + 8 * np.sin(xx / 37) + rng.randn(*xx.shape) * 2, img)
    img = cv2.resize(img.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    for _ in range(int(28 + 20 * lv)):  # ресницы вдоль верхнего века
        x0 = rng.uniform(ci[0] - 1.6 * R, ci[0] + 1.6 * R)
        y0 = ci[1] - au + au * ((x0 - ci[0]) / span) ** 2
        ang = np.radians(rng.uniform(60, 120))
        ln = rng.uniform(6, 16)
        cv2.line(img, (int(x0), int(y0)), (int(x0 + ln * np.cos(ang) * 0.5), int(y0 + ln * np.sin(ang))), 45, 1, cv2.LINE_AA)
    img = img * rng.uniform(0.9, 1.08)
    sig = 0.5 + 0.7 * lv * rng.rand()
    img = cv2.GaussianBlur(img, (0, 0), sig) + rng.randn(H, W) * (2 + 4 * lv)
    truth = {"pupil": [round(float(cp[0]), 2), round(float(cp[1]), 2), round(float(rp), 2)],
             "iris": [round(float(ci[0]), 2), round(float(ci[1]), 2), round(float(R), 2)], "rotation_deg": round(float(rot), 2)}
    return np.clip(img, 0, 255).astype(np.uint8), truth


@lru_cache(maxsize=1)
def _truth_file() -> dict:
    f = DATA / "truth.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


@lru_cache(maxsize=64)
def impression(eye: int, imp: int) -> tuple[np.ndarray, dict]:
    """Изображение из datasets/synthetic_iris, если папка есть, иначе генерируется (результат одинаковый)."""
    key = f"e{eye:02d}_i{imp}"
    f = DATA / f"{key}.png"
    if f.exists() and key in _truth_file():
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img, _truth_file()[key]
    return render(eye, imp)
