"""Наборы для обучения нейросети распознаванию: отпечатки (синтетика) и лица (галерея datasets/faces)."""
from dataclasses import dataclass

import cv2
import numpy as np

from app.modules.fingerprint import synthetic


@dataclass
class TrainData:
    modality: str
    names: list[str]
    size: int
    train: list[np.ndarray]
    ytrain: np.ndarray
    test: list[np.ndarray]
    ytest: np.ndarray
    border: float | None  # None: повторять край (лица); число: заливка (отпечатки)
    flip: bool
    shift_frac: float
    noise: tuple[float, float]


def prep(img: np.ndarray, size: int) -> np.ndarray:
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def augment(img: np.ndarray, rng: np.random.RandomState, strength: float, d: TrainData) -> np.ndarray:
    """Случайные искажения (поворот, сдвиг, масштаб, контраст, шум, зеркало для лиц): «новые оттиски» того же объекта."""
    s = float(strength)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-12, 12) * s, 1 + rng.uniform(-0.05, 0.05) * s)
    M[:, 2] += rng.uniform(-1, 1, 2) * d.shift_frac * min(h, w) * s
    kw = dict(borderMode=cv2.BORDER_REPLICATE) if d.border is None else dict(borderMode=cv2.BORDER_CONSTANT, borderValue=d.border)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, **kw).astype(np.float32)
    if d.flip and rng.rand() < 0.5 * s:
        out = out[:, ::-1]
    out = (out - 128) * (1 + (rng.uniform(0.85, 1.1) - 1) * s) + 128 + rng.uniform(-12, 12) * s
    out = out + rng.randn(h, w) * rng.uniform(*d.noise) * s
    return np.clip(out, 0, 255).astype(np.uint8)


def load_fingerprint(n_subjects: int) -> TrainData:
    n = max(3, min(n_subjects, 12))
    train, ytrain, test, ytest = [], [], [], []
    for f in range(1, n + 1):
        for i in (1, 2):
            train.append(synthetic.impression(f, i)); ytrain.append(f - 1)
        test.append(synthetic.impression(f, 3)); ytest.append(f - 1)  # самый «трудный» оттиск откладываем для проверки
    return TrainData("fingerprint", [f"Палец {f:02d}" for f in range(1, n + 1)], 64, train, np.array(ytrain),
                     test, np.array(ytest), 240.0, False, 0.04, (2.0, 10.0))


def load_face(n_subjects: int) -> TrainData:
    from app.modules.face.module import FaceModule
    from app.modules.face.params import FaceParams
    m = FaceModule()
    p = FaceParams()
    people: dict[str, list[np.ndarray]] = {}
    for person, lst in m._scan().items():
        imgs = []
        for sid, _ in lst[:8]:
            r = m._run(sid, p, with_images=False)
            if r.ok:
                imgs.append(cv2.cvtColor(r.aligned, cv2.COLOR_BGR2GRAY))
        if len(imgs) >= 2:
            people[person] = imgs
    names = list(people)[:max(2, min(n_subjects, 12))]
    if len(names) < 2:
        raise ValueError("Для обучения на лицах нужны минимум 2 человека, у каждого 2+ фото с найденным лицом в datasets/faces/<имя>/.")
    train, ytrain, test, ytest = [], [], [], []
    for k, nm in enumerate(names):
        imgs = people[nm]
        for im in imgs[:-1]:
            train.append(im); ytrain.append(k)
        test.append(imgs[-1]); ytest.append(k)
    return TrainData("face", names, 48, train, np.array(ytrain), test, np.array(ytest), None, True, 0.055, (1.0, 6.0))


def load_dataset(modality: str, n_subjects: int) -> TrainData:
    if modality == "fingerprint":
        return load_fingerprint(n_subjects)
    if modality == "face":
        return load_face(n_subjects)
    raise ValueError(f"Неизвестная биометрия: {modality}")
