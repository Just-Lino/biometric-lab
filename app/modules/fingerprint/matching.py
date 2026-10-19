from dataclasses import dataclass, field

import cv2
import numpy as np

from app.modules.fingerprint.minutiae import Minutia


@dataclass
class MatchResult:
    score: float = 0.0
    matched: list[tuple[int, int]] = field(default_factory=list)
    rotation_deg: float = 0.0
    shift: tuple[float, float] = (0.0, 0.0)


def _wrap_pi(a):
    """Приведение угла к [-pi/2, pi/2): ориентации гребней определены с точностью до pi."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def _arr(ms: list[Minutia]):
    return (np.array([[m.x, m.y] for m in ms], np.float32).reshape(-1, 2),
            np.array([m.theta for m in ms], np.float32),
            np.array([m.type == "ending" for m in ms]))


def _count(pa, ta, ea, pb, tb, eb, rot, t, dist_tol, ang_tol):
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]], np.float32)
    pa2 = pa @ R.T + t
    d = np.linalg.norm(pa2[:, None, :] - pb[None, :, :], axis=2)
    dth = np.abs(_wrap_pi(tb[None, :] - (ta[:, None] + rot)))
    ok = (d <= dist_tol) & (dth <= ang_tol) & (ea[:, None] == eb[None, :])
    pairs = sorted(zip(*np.where(ok)), key=lambda ij: d[ij])
    used_a, used_b, matched = set(), set(), []
    for i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j), matched.append((int(i), int(j)))
    return matched


def match_minutiae(A: list[Minutia], B: list[Minutia], center=(128.0, 128.0),
                   dist_tol=14.0, angle_tol_deg=20.0, max_rotation_deg=40.0) -> MatchResult:
    if len(A) < 3 or len(B) < 3:
        return MatchResult()
    c = np.array(center, np.float32)
    pa, ta, ea = _arr(A)
    pb, tb, eb = _arr(B)
    pa, pb = pa - c, pb - c
    ang_tol = np.radians(angle_tol_deg)
    max_rot = np.radians(max_rotation_deg)

    # голосование Хафа по (поворот, сдвиг)
    I, J = np.where(ea[:, None] == eb[None, :])
    rot = _wrap_pi(tb[J] - ta[I])
    keep = np.abs(rot) <= max_rot
    I, J, rot = I[keep], J[keep], rot[keep]
    if len(I) == 0:
        return MatchResult()
    cs, sn = np.cos(rot), np.sin(rot)
    tx = pb[J, 0] - (cs * pa[I, 0] - sn * pa[I, 1])
    ty = pb[J, 1] - (sn * pa[I, 0] + cs * pa[I, 1])
    keys = np.stack([np.rint(rot / np.radians(8)), np.rint(tx / 12), np.rint(ty / 12)], 1).astype(int)
    uniq, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()

    best = MatchResult()
    for k in np.argsort(-cnt)[:6]:
        sel = inv == k
        r, t = float(rot[sel].mean()), np.array([tx[sel].mean(), ty[sel].mean()], np.float32)
        matched = _count(pa, ta, ea, pb, tb, eb, r, t, dist_tol, ang_tol)
        score = 2 * len(matched) / (len(A) + len(B))
        if score > best.score:
            best = MatchResult(score, matched, float(np.degrees(r)), (float(t[0]), float(t[1])))
    return best


def draw_match(grayA, grayB, A, B, result: MatchResult) -> np.ndarray:
    h, w = grayA.shape
    canvas = np.hstack([cv2.cvtColor(grayA, cv2.COLOR_GRAY2BGR), cv2.cvtColor(grayB, cv2.COLOR_GRAY2BGR)])
    matched_a = {i for i, _ in result.matched}
    matched_b = {j for _, j in result.matched}

    def mark(m, ox, hit):
        col = (0, 170, 0) if hit else (150, 150, 150)
        c = (m.x + ox, m.y)
        if m.type == "ending":
            cv2.circle(canvas, c, 4, col, 1, cv2.LINE_AA)
        else:
            pts = np.array([[c[0], c[1] - 5], [c[0] - 5, c[1] + 4], [c[0] + 5, c[1] + 4]], np.int32)
            cv2.polylines(canvas, [pts], True, col, 1, cv2.LINE_AA)

    for i, m in enumerate(A):
        mark(m, 0, i in matched_a)
    for j, m in enumerate(B):
        mark(m, w, j in matched_b)
    for i, j in result.matched:
        cv2.line(canvas, (A[i].x, A[i].y), (B[j].x + w, B[j].y), (0, 200, 255), 1, cv2.LINE_AA)
    return canvas
