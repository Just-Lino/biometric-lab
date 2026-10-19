from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

import cv2
import numpy as np

from app.modules.face import models
from app.modules.face.params import FaceParams

_lock = Lock()


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: np.ndarray  # (5, 2): левый глаз, правый глаз, нос, левый и правый углы рта (слева направо в кадре)
    lm_source: str  # yunet | eyes-cascade | proportional
    detector: str  # haar | yunet


def resolve_detector(p: FaceParams) -> str:
    have_yunet = models.model_path("yunet") is not None
    if p.detector == "auto":
        return "yunet" if have_yunet else "haar"
    if p.detector == "yunet" and not have_yunet:
        raise ValueError("Модель YuNet не найдена в папке models/. Выполните: python -m scripts.download_models")
    return p.detector


@lru_cache(maxsize=4)
def _cascade(name: str) -> cv2.CascadeClassifier:
    c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    if c.empty():
        raise RuntimeError(f"Не удалось загрузить каскад {name}")
    return c


@lru_cache(maxsize=2)
def _yunet(path: str):
    return cv2.FaceDetectorYN.create(path, "", (320, 320), 0.8, 0.3, 5000)


def _proportional(x, y, w, h) -> np.ndarray:
    return np.array([[x + .30 * w, y + .37 * h], [x + .70 * w, y + .37 * h], [x + .50 * w, y + .62 * h],
                     [x + .33 * w, y + .80 * h], [x + .67 * w, y + .80 * h]], np.float32)


def _eyes_haar(gray: np.ndarray, box) -> np.ndarray | None:
    x, y, w, h = box
    roi = gray[y:y + int(h * 0.65), x:x + w]
    if roi.size == 0:
        return None
    m = max(8, w // 10)
    eyes = _cascade("haarcascade_eye.xml").detectMultiScale(roi, 1.1, 5, minSize=(m, m))
    left = [e for e in eyes if e[0] + e[2] / 2 < w / 2]
    right = [e for e in eyes if e[0] + e[2] / 2 >= w / 2]
    if not left or not right:
        return None
    el, er = max(left, key=lambda e: e[2] * e[3]), max(right, key=lambda e: e[2] * e[3])
    pl = (x + el[0] + el[2] / 2, y + el[1] + el[3] / 2)
    pr = (x + er[0] + er[2] / 2, y + er[1] + er[3] / 2)
    dist = np.hypot(pr[0] - pl[0], pr[1] - pl[1])
    tilt = abs(np.degrees(np.arctan2(pr[1] - pl[1], pr[0] - pl[0])))
    if not (0.25 * w <= dist <= 0.75 * w) or tilt > 35:
        return None
    return np.array([pl, pr], np.float32)


def _detect_haar(img: np.ndarray, p: FaceParams) -> Detection | None:
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    out = _cascade("haarcascade_frontalface_default.xml").detectMultiScale3(
        gray, scaleFactor=p.scale_factor, minNeighbors=p.min_neighbors,
        minSize=(p.min_face, p.min_face), outputRejectLevels=True)
    if len(out) != 3 or len(out[0]) == 0:
        return None
    rects, _, weights = out
    i = int(np.argmax([r[2] * r[3] for r in rects]))
    x, y, w, h = (int(v) for v in rects[i])
    conf = float(1 - np.exp(-float(np.ravel(weights)[i]) / 2.5))  # условная оценка по «весу» каскада
    lm = _proportional(x, y, w, h)
    src = "proportional"
    eyes = _eyes_haar(gray, (x, y, w, h))
    if eyes is not None:
        lm[:2] = eyes
        src = "eyes-cascade"
    return Detection(x, y, w, h, conf, lm, src, "haar")


def _detect_yunet(img: np.ndarray, p: FaceParams) -> Detection | None:
    path = models.model_path("yunet")
    h, w = img.shape[:2]
    with _lock:
        det = _yunet(str(path))
        det.setInputSize((w, h))
        det.setScoreThreshold(float(p.yunet_score))
        _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    faces = [f for f in faces if min(f[2], f[3]) >= p.min_face]
    if not faces:
        return None
    f = max(faces, key=lambda r: r[2] * r[3])
    x, y, bw, bh = (int(round(float(v))) for v in f[:4])
    lm = np.array(f[4:14], np.float32).reshape(5, 2)
    return Detection(x, y, bw, bh, float(f[14]), lm, "yunet", "yunet")


def detect_face(img: np.ndarray, p: FaceParams, detector: str) -> Detection | None:
    return _detect_yunet(img, p) if detector == "yunet" else _detect_haar(img, p)
