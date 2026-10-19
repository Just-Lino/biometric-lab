from dataclasses import dataclass, asdict

import cv2
import numpy as np

from app.modules.fingerprint.params import FingerprintParams

ENDING, BIFURCATION = "ending", "bifurcation"
_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


@dataclass
class Minutia:
    x: int
    y: int
    theta: float  # ориентация гребня, радианы [0, pi)
    type: str
    quality: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["theta_deg"] = round(float(np.degrees(self.theta)), 1)
        d["quality"] = round(float(self.quality), 2)
        return d


def _crossing_number(sk: np.ndarray) -> np.ndarray:
    p = np.pad(sk, 1)
    h, w = sk.shape
    nb = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w].astype(np.int16) for dy, dx in _OFFS]
    cn = np.zeros((h, w), np.int16)
    for i in range(8):
        cn += np.abs(nb[i] - nb[(i + 1) % 8])
    return cn // 2


def _trace_is_spur(sk: np.ndarray, y: int, x: int, max_len: int) -> bool:
    """Идём от окончания вдоль скелета; если быстро упираемся в разветвление, это ложный «шип»."""
    h, w = sk.shape
    visited = {(y, x)}
    cy, cx = y, x
    for _ in range(max_len):
        nxt = [(cy + dy, cx + dx) for dy, dx in _OFFS
               if 0 <= cy + dy < h and 0 <= cx + dx < w and sk[cy + dy, cx + dx] and (cy + dy, cx + dx) not in visited]
        if not nxt:
            return False
        if len(nxt) >= 2:
            return True
        cy, cx = nxt[0]
        visited.add((cy, cx))
    return False


def extract_minutiae(skeleton: np.ndarray, phi: np.ndarray, coherence: np.ndarray,
                     mask: np.ndarray, p: FingerprintParams) -> list[Minutia]:
    sk = (skeleton > 0).astype(np.uint8)
    cn = _crossing_number(sk)
    valid = mask.astype(np.uint8)
    if p.border_margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.border_margin + 1,) * 2)
        valid = cv2.erode(valid, k)
    cand = []
    for typ, val in ((ENDING, 1), (BIFURCATION, 3)):
        ys, xs = np.where((sk == 1) & (cn == val) & (valid > 0))
        for y, x in zip(ys, xs):
            if typ == ENDING and p.spur_length > 0 and _trace_is_spur(sk, y, x, p.spur_length):
                continue
            cand.append(Minutia(int(x), int(y), float(phi[y, x]), typ, float(coherence[y, x])))
    # прореживание: оставляем более «качественные» точки, близкие отбрасываем
    cand.sort(key=lambda m: -m.quality)
    kept: list[Minutia] = []
    for m in cand:
        if all((m.x - k.x) ** 2 + (m.y - k.y) ** 2 >= p.min_minutiae_dist ** 2 for k in kept):
            kept.append(m)
    kept.sort(key=lambda m: (m.y, m.x))
    return kept


def draw_minutiae(gray: np.ndarray, minutiae: list[Minutia], scale: int = 1) -> np.ndarray:
    """● окончание (красный), ▲ разветвление (синий)."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for m in minutiae:
        c = (m.x * scale, m.y * scale)
        if m.type == ENDING:
            cv2.circle(vis, c, 5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            pts = np.array([[c[0], c[1] - 6], [c[0] - 6, c[1] + 5], [c[0] + 6, c[1] + 5]], np.int32)
            cv2.polylines(vis, [pts], True, (255, 80, 0), 1, cv2.LINE_AA)
    return vis
