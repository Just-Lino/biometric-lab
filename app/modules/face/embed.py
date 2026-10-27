from functools import lru_cache
from threading import Lock

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from app.modules.face import models
from app.modules.face.params import FaceParams
from app.training import registry

_lock = Lock()

# имя -> название для интерфейса
EMBEDDERS = {
    "sface": "SFace (нейросеть, 128 чисел)",
    "arcface": "ArcFace MobileFaceNet (нейросеть, 512 чисел)",
    "lbp": "LBP-гистограммы (классика, 640 чисел)",
    "hog": "HOG - гистограммы градиентов (классика, 1764 числа)",
    "gabor": "Признаки Габора (классика, 576 чисел)",
    "dct": "DCT - низкие частоты (классика, 63 числа)",
    "pixels": "Нормированные пиксели (простейший эталон, 576 чисел)",
    "eigenfaces": "Eigenfaces - метод главных компонент по галерее (классика, до 40 чисел)",
    "cnn": "Нейросеть, обученная вами (вкладка «Обучение», 32 числа)",
}
# классические признаки у всех лиц похожи, поэтому из них вычитается «среднее лицо» галереи
CLASSICAL = {"lbp", "hog", "gabor", "dct", "pixels"}


def embedder_available(name: str) -> bool:
    if name in ("sface", "arcface"):
        return models.model_path(name) is not None
    if name == "cnn":
        return registry.get("face") is not None
    return True


def available_embedders() -> dict[str, bool]:
    return {k: embedder_available(k) for k in EMBEDDERS}


def resolve_embedder(p: FaceParams) -> str:
    if p.embedder == "auto":
        return "sface" if embedder_available("sface") else ("arcface" if embedder_available("arcface") else "lbp")
    if p.embedder == "cnn" and not embedder_available("cnn"):
        registry.require("face", "лиц")  # понятное сообщение: сеть ещё не обучена
    if not embedder_available(p.embedder):
        raise ValueError(f"Модель признаков «{EMBEDDERS[p.embedder]}» не найдена в папке models/. "
                         "Выполните: python -m scripts.download_models")
    return p.embedder


def _gray(aligned_bgr: np.ndarray, size: int, tiles: int = 4) -> np.ndarray:
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tiles, tiles)).apply(cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY))
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


# ---------- нейросети ----------
@lru_cache(maxsize=2)
def _sface(path: str):
    return cv2.FaceRecognizerSF.create(path, "")


def sface_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    with _lock:
        feat = _sface(str(models.model_path("sface"))).feature(aligned_bgr)
    return np.asarray(feat, np.float32).ravel()


@lru_cache(maxsize=1)
def _arcface(path: str):
    return cv2.dnn.readNetFromONNX(path)


def arcface_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    blob = cv2.dnn.blobFromImage(aligned_bgr, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    with _lock:
        net = _arcface(str(models.model_path("arcface")))
        net.setInput(blob)
        out = net.forward()
    return np.asarray(out, np.float32).ravel()


# ---------- классические признаки ----------
def lbp_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    """LBP-гистограммы по сетке 8×8 (корень гистограмм, как в расстоянии Хеллингера)."""
    g = _gray(aligned_bgr, aligned_bgr.shape[0])
    lbp = local_binary_pattern(g, 8, 1, "uniform").astype(np.int64)
    cells, step = 8, g.shape[0] // 8
    hist = [np.bincount(lbp[i * step:(i + 1) * step, j * step:(j + 1) * step].ravel(), minlength=10) / (step * step)
            for i in range(cells) for j in range(cells)]
    return np.sqrt(np.concatenate(hist)).astype(np.float32)


def hog_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    """HOG: гистограммы направлений градиентов в ячейках 8×8 пикселей, нормировка по блокам."""
    return hog(_gray(aligned_bgr, 64), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


@lru_cache(maxsize=1)
def _gabor_bank():
    bank = []
    for lam in (4, 8):  # две длины волны, восемь направлений
        for k in range(8):
            args = ((17, 17), lam * 0.56, k * np.pi / 8, lam, 1.0)
            bank.append((cv2.getGaborKernel(*args, 0, ktype=cv2.CV_32F), cv2.getGaborKernel(*args, np.pi / 2, ktype=cv2.CV_32F)))
    return bank


def gabor_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    """Амплитуды откликов 16 фильтров Габора, усреднённые по сетке 6×6."""
    g = _gray(aligned_bgr, 64).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    feats = []
    for kr, ki in _gabor_bank():
        re, im = cv2.filter2D(g, cv2.CV_32F, kr), cv2.filter2D(g, cv2.CV_32F, ki)
        feats.append(cv2.resize(np.sqrt(re * re + im * im), (6, 6), interpolation=cv2.INTER_AREA).ravel())
    return np.concatenate(feats).astype(np.float32)


def dct_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    """Дискретное косинусное преобразование: берутся 8×8 низких частот без нулевой (средняя яркость)."""
    d = cv2.dct(_gray(aligned_bgr, 32).astype(np.float32) / 255)
    return d[:8, :8].ravel()[1:].astype(np.float32)


def pixels_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    """Простейший эталон: сами пиксели уменьшенного лица 24×24, нормированные по яркости."""
    g = _gray(aligned_bgr, 24).astype(np.float32)
    return ((g - g.mean()) / (g.std() + 1e-6)).ravel()


_EMBED = {"sface": sface_embedding, "arcface": arcface_embedding, "lbp": lbp_embedding, "hog": hog_embedding,
          "gabor": gabor_embedding, "dct": dct_embedding, "pixels": pixels_embedding}


def embed(aligned_bgr: np.ndarray, embedder: str) -> np.ndarray:
    return _EMBED[embedder](aligned_bgr)


def eigen_basis(faces: list[np.ndarray], k: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Eigenfaces: среднее лицо и главные компоненты (SVD) по выровненным лицам галереи. Метки людей не нужны."""
    if len(faces) < 6:
        raise ValueError("Для Eigenfaces нужна галерея минимум из 6 фото (лучше 3+ человека по 2+ фото) в datasets/faces/.")
    x = np.stack([_gray(f, 32).astype(np.float32).ravel() for f in faces])
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)  # яркость каждого лица приводится к одной шкале
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean.astype(np.float32), vt[:min(k, len(faces) - 1)].astype(np.float32)


def eigen_embedding(aligned_bgr: np.ndarray, basis: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, comps = basis
    g = _gray(aligned_bgr, 32).astype(np.float32).ravel()
    return comps @ (((g - g.mean()) / (g.std() + 1e-6)) - mean)


def embedder_title(name: str) -> str:
    return EMBEDDERS[name]
