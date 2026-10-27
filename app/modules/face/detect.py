from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

import cv2
import numpy as np

from app.modules.face import models
from app.modules.face.params import FaceParams

_lock = Lock()

# имя -> название для интерфейса
DETECTORS = {
    "haar": "Haar: frontalface_default (классика)",
    "haar_alt2": "Haar: frontalface_alt2 (классика)",
    "haar_alt": "Haar: frontalface_alt (классика)",
    "haar_profile": "Haar: профиль лица (классика)",
    "ssd": "SSD ResNet-10 (нейросеть)",
    "ultraface": "UltraFace RFB-320 (лёгкая нейросеть)",
    "yunet": "YuNet (нейросеть, свои ключевые точки)",
}
CASCADES = {"haar": "haarcascade_frontalface_default.xml", "haar_alt2": "haarcascade_frontalface_alt2.xml",
            "haar_alt": "haarcascade_frontalface_alt.xml", "haar_profile": "haarcascade_profileface.xml"}


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: np.ndarray  # (5, 2): левый глаз, правый глаз, нос, левый и правый углы рта (слева направо в кадре)
    lm_source: str  # yunet | eyes-cascade | proportional
    detector: str  # ключ из DETECTORS


def detector_available(name: str) -> bool:
    if name == "yunet":
        return models.model_path("yunet") is not None
    if name == "ultraface":
        return models.model_path("ultraface") is not None
    if name == "ssd":
        return models.model_path("ssd_proto") is not None and models.model_path("ssd_model") is not None
    return True


def available_detectors() -> dict[str, bool]:
    return {k: detector_available(k) for k in DETECTORS}


def resolve_detector(p: FaceParams) -> str:
    if p.detector == "auto":
        return next((d for d in ("yunet", "ssd", "ultraface") if detector_available(d)), "haar")
    if not detector_available(p.detector):
        raise ValueError(f"Модель детектора «{DETECTORS[p.detector]}» не найдена в папке models/. "
                         "Выполните: python -m scripts.download_models")
    return p.detector


@lru_cache(maxsize=8)
def _cascade(name: str) -> cv2.CascadeClassifier:
    c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    if c.empty():
        raise RuntimeError(f"Не удалось загрузить каскад {name}")
    return c


@lru_cache(maxsize=2)
def _yunet(path: str):
    return cv2.FaceDetectorYN.create(path, "", (320, 320), 0.8, 0.3, 5000)


@lru_cache(maxsize=1)
def _ssd_net(proto: str, weights: str):
    return cv2.dnn.readNetFromCaffe(proto, weights)


def _proportional(x, y, w, h, tall: bool = False) -> np.ndarray:
    nose, mouth = (.58, .74) if tall else (.62, .80)  # рамка SSD выше и включает лоб
    return np.array([[x + .30 * w, y + .37 * h], [x + .70 * w, y + .37 * h], [x + .50 * w, y + nose * h],
                     [x + .33 * w, y + mouth * h], [x + .67 * w, y + mouth * h]], np.float32)


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


def _with_eyes(gray, box, lm: np.ndarray) -> tuple[np.ndarray, str]:
    eyes = _eyes_haar(gray, box)
    if eyes is None:
        return lm, "proportional"
    lm = lm.copy()
    lm[:2] = eyes
    return lm, "eyes-cascade"


def _detect_haar(img: np.ndarray, p: FaceParams, name: str) -> Detection | None:
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    casc = _cascade(CASCADES[name])

    def run(g):
        out = casc.detectMultiScale3(g, scaleFactor=p.scale_factor, minNeighbors=p.min_neighbors,
                                     minSize=(p.min_face, p.min_face), outputRejectLevels=True)
        if len(out) != 3 or len(out[0]) == 0:
            return []
        return [(r, float(wt)) for r, wt in zip(out[0], np.ravel(out[2]))]

    cands = [(int(r[0]), int(r[1]), int(r[2]), int(r[3]), wt) for r, wt in run(gray)]
    if name == "haar_profile":  # каскад ищет лицо, повёрнутое в одну сторону: проверяем и зеркальное отражение
        width = gray.shape[1]
        cands += [(width - int(r[0]) - int(r[2]), int(r[1]), int(r[2]), int(r[3]), wt) for r, wt in run(cv2.flip(gray, 1))]
    if not cands:
        return None
    x, y, w, h, wt = max(cands, key=lambda c: c[2] * c[3])
    conf = float(1 - np.exp(-wt / 2.5))  # условная оценка по «весу» каскада
    lm, src = _proportional(x, y, w, h), "proportional"
    if name != "haar_profile":  # у профиля глаза в прямой проекции не ищем
        lm, src = _with_eyes(gray, (x, y, w, h), lm)
    return Detection(x, y, w, h, conf, lm, src, name)


def _detect_ssd(img: np.ndarray, p: FaceParams) -> Detection | None:
    proto, weights = models.model_path("ssd_proto"), models.model_path("ssd_model")
    h, w = img.shape[:2]
    with _lock:
        net = _ssd_net(str(proto), str(weights))
        net.setInput(cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 177, 123)))
        det = net.forward()[0, 0]
    best = None
    for d in det[det[:, 2] >= p.yunet_score]:
        x1, y1, x2, y2 = np.clip(d[3:7] * [w, h, w, h], 0, [w, h, w, h])
        bw, bh = int(x2 - x1), int(y2 - y1)
        if min(bw, bh) >= p.min_face and (best is None or bw * bh > best[2] * best[3]):
            best = (int(x1), int(y1), bw, bh, float(d[2]))
    if best is None:
        return None
    x, y, bw, bh, conf = best
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    lm, src = _with_eyes(gray, (x, y, bw, bh), _proportional(x, y, bw, bh, tall=True))
    return Detection(x, y, bw, bh, conf, lm, src, "ssd")


@lru_cache(maxsize=1)
def _ultraface_net(path: str):
    return cv2.dnn.readNetFromONNX(path)


def _detect_ultraface(img: np.ndarray, p: FaceParams) -> Detection | None:
    """UltraFace RFB-320: вход 320×240, выходы - рамки (углы, доли кадра) и вероятности; лишние рамки убирает NMS."""
    h, w = img.shape[:2]
    with _lock:
        net = _ultraface_net(str(models.model_path("ultraface")))
        net.setInput(cv2.dnn.blobFromImage(img, 1 / 128.0, (320, 240), (127, 127, 127), swapRB=True))
        boxes, scores = net.forward(["boxes", "scores"])
    boxes, scores = boxes[0], scores[0][:, 1]
    keep = scores >= p.yunet_score
    if not keep.any():
        return None
    b = (np.clip(boxes[keep], 0, 1) * [w, h, w, h]).astype(np.float32)
    rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in b]
    idx = np.array(cv2.dnn.NMSBoxes(rects, scores[keep].tolist(), float(p.yunet_score), 0.3)).reshape(-1)
    best = None
    for i in idx:
        x, y, bw, bh = (int(round(v)) for v in rects[i])
        if min(bw, bh) >= p.min_face and (best is None or bw * bh > best[2] * best[3]):
            best = (x, y, bw, bh, float(scores[keep][i]))
    if best is None:
        return None
    x, y, bw, bh, conf = best
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    lm, src = _with_eyes(gray, (x, y, bw, bh), _proportional(x, y, bw, bh, tall=True))
    return Detection(x, y, bw, bh, conf, lm, src, "ultraface")


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
    if detector == "yunet":
        return _detect_yunet(img, p)
    if detector == "ssd":
        return _detect_ssd(img, p)
    if detector == "ultraface":
        return _detect_ultraface(img, p)
    return _detect_haar(img, p, detector)
