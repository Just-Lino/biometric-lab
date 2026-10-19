import cv2
import numpy as np

# шаблон ArcFace для 112×112: левый глаз, правый глаз, нос, левый и правый углы рта (в координатах изображения)
TEMPLATE = np.array([[38.29, 51.69], [73.53, 51.50], [56.03, 71.74], [41.55, 92.37], [70.73, 92.20]], np.float32)
SIZE = 112


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """МНК-оценка преобразования подобия (поворот + масштаб + сдвиг), метод Умеямы. Возвращает матрицу 2×3."""
    src, dst = src.astype(np.float64), dst.astype(np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var = (s0 ** 2).sum() / len(src)
    U, D, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = (D * np.diag(S)).sum() / var
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)


def align_face(img: np.ndarray, landmarks: np.ndarray, use_all: bool = True) -> np.ndarray:
    pts = landmarks if use_all else landmarks[:2]
    M = similarity_transform(pts, TEMPLATE[:len(pts)])
    return cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_face(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Простое вырезание лица по рамке (без выравнивания): квадрат с небольшим запасом."""
    x, y, w, h = box
    side = int(max(w, h) * 1.15)
    cx, cy = x + w / 2, y + h / 2
    M = np.array([[SIZE / side, 0, SIZE / 2 - cx * SIZE / side],
                  [0, SIZE / side, SIZE / 2 - cy * SIZE / side]], np.float32)
    return cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
