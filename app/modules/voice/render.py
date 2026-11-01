"""Картинки этапов обработки голоса (cv2 и NumPy, без matplotlib). Подписи на латинице: cv2.putText не рисует кириллицу."""
import cv2
import numpy as np

from app.modules.voice.audio import HOP, SR

W, H = 640, 200
INK, GRID = (60, 60, 60), (225, 225, 225)


def _canvas(h: int = H, w: int = W) -> np.ndarray:
    return np.full((h, w, 3), 255, np.uint8)


def _text(img, s, x, y, scale=0.4, col=INK):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def _time_axis(img, dur: float, top: int, bottom: int):
    h, w = img.shape[:2]
    step = 0.5 if dur <= 4 else 1.0
    t = 0.0
    while t <= dur:
        x = int(t / dur * (w - 1))
        cv2.line(img, (x, top), (x, bottom), GRID, 1)
        _text(img, f"{t:g}s", min(x + 2, w - 28), h - 4)
        t += step


def waveform(x: np.ndarray, mask: np.ndarray | None = None, h: int = H, w: int = W) -> np.ndarray:
    """Форма волны; зелёным подсвечены кадры, признанные речью."""
    img = _canvas(h, w)
    dur = len(x) / SR
    _time_axis(img, dur, 0, h - 14)
    if mask is not None:
        for i, m in enumerate(mask):
            if m:
                x0, x1 = int(i * HOP / len(x) * w), int((i + 1) * HOP / len(x) * w) + 1
                cv2.rectangle(img, (x0, 0), (x1, h - 15), (215, 240, 215), -1)
    mid, amp = (h - 14) // 2, (h - 14) / 2 - 4
    cols = np.array_split(x, w)
    for c, seg in enumerate(cols):
        if len(seg):
            cv2.line(img, (c, int(mid - seg.max() / (np.abs(x).max() + 1e-9) * amp)), (c, int(mid - seg.min() / (np.abs(x).max() + 1e-9) * amp)), (200, 90, 30), 1)
    cv2.line(img, (0, mid), (w - 1, mid), (150, 150, 150), 1)
    return img


def heat(M: np.ndarray, dur: float, h: int = H, w: int = W, cmap=cv2.COLORMAP_VIRIDIS, ylabels: tuple = ()) -> np.ndarray:
    """Тепловая карта (кадры × частоты) с осью времени; низкие частоты внизу."""
    lo, hi = np.percentile(M, 2), np.percentile(M, 99)
    u8 = np.clip((M - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8).T[::-1]
    body = cv2.applyColorMap(cv2.resize(u8, (w, h - 14), interpolation=cv2.INTER_LINEAR), cmap)
    img = np.vstack([body, _canvas(14, w)])
    for t in np.arange(0, dur, 0.5 if dur <= 4 else 1.0):
        x = int(t / dur * (w - 1))
        _text(img, f"{t:g}s", min(x + 2, w - 28), h - 3)
    for s, y in ylabels:
        _text(img, s, 3, int(y * (h - 14)) + 10, 0.4, (255, 255, 255))
    return img


def pitch_plot(f0: np.ndarray, mask: np.ndarray, h: int = H, w: int = W) -> np.ndarray:
    """Высота тона по кадрам (Гц); зелёная линия - медиана по речи."""
    img = _canvas(h, w)
    dur = len(f0) * HOP / SR
    _time_axis(img, dur, 0, h - 14)
    top = h - 16
    for hz in (100, 200, 300, 400):
        y = int(top - hz / 400 * (top - 8))
        cv2.line(img, (0, y), (w - 1, y), GRID, 1)
        _text(img, f"{hz}", 3, y - 2)
    pts = [(int(i / max(1, len(f0) - 1) * (w - 1)), int(top - v / 400 * (top - 8))) for i, v in enumerate(f0) if v > 0 and mask[i]]
    for p in pts:
        cv2.circle(img, p, 2, (200, 90, 30), -1, cv2.LINE_AA)
    v = f0[mask & (f0 > 0)]
    if len(v):
        y = int(top - np.median(v) / 400 * (top - 8))
        cv2.line(img, (0, y), (w - 1, y), (60, 160, 60), 1)
        _text(img, f"median {np.median(v):.0f} Hz", w - 150, max(12, y - 4), 0.45, (60, 140, 60))
    return img


def bars(vectors: list[np.ndarray], colors=((200, 90, 30), (60, 120, 220)), h: int = H, w: int = W, title: str = "") -> np.ndarray:
    """Столбики z-вектора шаблона (один или два голоса рядом): видно, какие признаки совпали."""
    img = _canvas(h, w)
    n = len(vectors[0])
    zero = h // 2
    cv2.line(img, (0, zero), (w - 1, zero), (150, 150, 150), 1)
    scale = (h // 2 - 16) / max(3.0, float(np.abs(np.concatenate(vectors)).max()))
    slot = (w - 10) / n
    for j, v in enumerate(vectors):
        bw = max(2, int(slot / (len(vectors) + 0.6)))
        for i, val in enumerate(v):
            x0 = int(5 + i * slot + j * bw)
            cv2.rectangle(img, (x0, zero), (x0 + bw - 1, int(zero - np.clip(val, -6, 6) * scale)), colors[j % len(colors)], -1)
    _text(img, title or "template z-scores", 4, 12)
    return img


def mark(img: np.ndarray, label: str, col=(0, 120, 255)) -> np.ndarray:
    out = img.copy()
    _text(out, label, 6, 16, 0.55, (0, 0, 0))
    _text(out, label, 6, 16, 0.5, col)
    return out
