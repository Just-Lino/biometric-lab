from functools import lru_cache
from threading import Lock

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from app.modules.face import models
from app.modules.face.params import FaceParams

_lock = Lock()


def resolve_embedder(p: FaceParams) -> str:
    have = models.model_path("sface") is not None
    if p.embedder == "auto":
        return "sface" if have else "lbp"
    if p.embedder == "sface" and not have:
        raise ValueError("Модель SFace не найдена в папке models/. Выполните: python -m scripts.download_models")
    return p.embedder


@lru_cache(maxsize=2)
def _sface(path: str):
    return cv2.FaceRecognizerSF.create(path, "")


def sface_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    with _lock:
        rec = _sface(str(models.model_path("sface")))
        feat = rec.feature(aligned_bgr)
    return np.asarray(feat, np.float32).ravel()


def lbp_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    """Классический признак: LBP-гистограммы по сетке 8×8 (корень гистограмм, как в расстоянии Хеллингера)."""
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY))
    lbp = local_binary_pattern(g, 8, 1, "uniform").astype(np.int64)
    cells, step = 8, aligned_bgr.shape[0] // 8
    hist = [np.bincount(lbp[i * step:(i + 1) * step, j * step:(j + 1) * step].ravel(), minlength=10) / (step * step)
            for i in range(cells) for j in range(cells)]
    return np.sqrt(np.concatenate(hist)).astype(np.float32)


def embed(aligned_bgr: np.ndarray, embedder: str) -> np.ndarray:
    return sface_embedding(aligned_bgr) if embedder == "sface" else lbp_embedding(aligned_bgr)


def embedder_title(name: str) -> str:
    return "SFace (нейросеть, 128-мерный embedding)" if name == "sface" else "LBP-гистограммы (классический метод)"
