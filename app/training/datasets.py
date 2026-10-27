"""Наборы для обучения нейросети распознаванию: отпечатки и радужка (синтетика или реальные FVC и CASIA из datasets/real)
и лица (галерея datasets/faces)."""
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
    holdout: int = 3  # номер снимка, оставленного для проверки (отпечатки и радужка)
    dataset: str = "synthetic"  # из какого набора взяты данные


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


def load_fingerprint(n_subjects: int, holdout: int = 3, dataset: str = "synthetic") -> TrainData:
    """Оттиск с номером holdout откладывается для проверки (по умолчанию 3 - самый «трудный»), остальные идут в обучение."""
    if dataset != "synthetic":
        from app.modules.fingerprint import real
        if dataset == real.SOCO or not real.is_real_source(dataset):
            raise ValueError("Для обучения подходят синтетика и наборы FVC. В SOCOFing по одному снимку на палец, различать пальцы по ним нельзя.")
        if not real.source_available(dataset):
            raise ValueError(f"Набор «{dataset}» не найден в datasets/real/. Подготовьте данные: python -m scripts.prepare_real")
        fingers = real.fvc_fingers(dataset)
        keys = sorted(k for k, v in fingers.items() if len(v) >= 3)[:max(3, min(n_subjects, 10))]
        if len(keys) < 3:
            raise ValueError("В наборе меньше трёх пальцев с тремя и более оттисками.")
        train, ytrain, test, ytest = [], [], [], []
        for cls, k in enumerate(keys):
            for i, sid in enumerate(fingers[k], 1):
                (test if i == holdout else train).append(real.load(sid))
                (ytest if i == holdout else ytrain).append(cls)
        return TrainData("fingerprint", [f"Палец {k}" for k in keys], 64, train, np.array(ytrain), test, np.array(ytest),
                         None, False, 0.04, (2.0, 8.0), holdout, dataset)
    n = max(3, min(n_subjects, 12))
    train, ytrain, test, ytest = [], [], [], []
    for f in range(1, n + 1):
        for i in (1, 2, 3):
            (test if i == holdout else train).append(synthetic.impression(f, i))
            (ytest if i == holdout else ytrain).append(f - 1)
    return TrainData("fingerprint", [f"Палец {f:02d}" for f in range(1, n + 1)], 64, train, np.array(ytrain),
                     test, np.array(ytest), 240.0, False, 0.04, (2.0, 10.0), holdout)


def load_face(n_subjects: int, dataset: str = "all") -> TrainData:
    """Берёт только нужное число людей (с двумя и более найденными лицами), а не прогоняет всю галерею: в LFW больше 5 000 человек."""
    import random

    from app.modules.face.module import FaceModule
    from app.modules.face.params import FaceParams
    m = FaceModule()
    p = FaceParams()
    try:
        scan = m._scan(dataset)
    except KeyError as e:
        raise ValueError(str(e).strip("'\""))
    candidates = [name for name, lst in scan.items() if len(lst) >= 2]
    random.Random(0).shuffle(candidates)  # фиксированная выборка: результат обучения повторяется
    want = max(2, min(n_subjects, MAX_FACES))
    people: dict[str, list[np.ndarray]] = {}
    for person in candidates:
        imgs = []
        for sid, _ in scan[person][:8]:
            r = m._run(sid, p, with_images=False)
            if r.ok:
                imgs.append(cv2.cvtColor(r.aligned, cv2.COLOR_BGR2GRAY))
        if len(imgs) >= 2:
            people[person] = imgs
        if len(people) >= want:
            break
    names = list(people)
    if len(names) < 2:
        raise ValueError("Для обучения на лицах нужны минимум 2 человека, у каждого 2+ фото с найденным лицом в datasets/faces/<имя>/.")
    train, ytrain, test, ytest = [], [], [], []
    for k, nm in enumerate(names):
        imgs = people[nm]
        for im in imgs[:-1]:
            train.append(im); ytrain.append(k)
        test.append(imgs[-1]); ytest.append(k)
    return TrainData("face", names, 48, train, np.array(ytrain), test, np.array(ytest), None, True, 0.055, (1.0, 6.0), 3, dataset)


def iris_crop(img: np.ndarray, circles) -> np.ndarray:
    """Квадрат 96×96 вокруг найденной радужки: вход сети (одинаков при обучении и при распознавании)."""
    x, y, r = circles.iris
    h = int(round(1.1 * r))
    pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_REPLICATE)
    return cv2.resize(pad[int(y):int(y) + 2 * h, int(x):int(x) + 2 * h], (96, 96), interpolation=cv2.INTER_AREA)


def load_iris(n_subjects: int, holdout: int = 3, dataset: str = "synthetic") -> TrainData:
    """Вход сети - квадрат вокруг найденной радужки (локализация оператором Даугмана): поворот при аугментации
    соответствует повороту глаза. Снимок holdout (по умолчанию 3, самый сложный) - проверка, остальные - обучение."""
    from app.modules.iris import real, synthetic as iris_syn
    from app.modules.iris.localize import localize, preprocess
    from app.modules.iris.params import IrisParams
    p = IrisParams()
    if dataset == "synthetic":
        n = max(3, min(n_subjects, iris_syn.N_EYES))
        eyes = {f"Глаз {e:02d}": [iris_syn.impression(e, i)[0] for i in (1, 2, 3)] for e in range(1, n + 1)}
    elif real.is_real_source(dataset):
        if not real.source_available():
            raise ValueError("Набор CASIA не найден в datasets/real/. Подготовьте данные: python -m scripts.prepare_real")
        eyes = {f"Глаз {e[1:]}": [real.load(sid) for sid in sids] for e, sids in real.eye_images(3, max(3, min(n_subjects, MAX_EYES))).items()}
    else:
        raise ValueError(f"Неизвестный набор данных «{dataset}».")
    names, train, ytrain, test, ytest = [], [], [], [], []
    for name, imgs in eyes.items():
        cls = len(names)
        names.append(name)
        for i, img in enumerate(imgs, 1):
            c = localize(preprocess(img, p), p, "daugman") or localize(preprocess(img, p), p, "contour")
            if c is None:
                continue
            (test if i == holdout else train).append(iris_crop(img, c))
            (ytest if i == holdout else ytrain).append(cls)
    if len(train) < 3 or len(test) < 3:
        raise ValueError("Не удалось найти радужку на достаточном числе снимков набора.")
    return TrainData("iris", names, 64, train, np.array(ytrain), test, np.array(ytest), None, False, 0.03, (2.0, 8.0), holdout, dataset)


MAX_EYES, MAX_FACES = 40, 30  # больше объектов - дольше обучение (сеть на NumPy, процессор)


def default_dataset(modality: str) -> str:
    return "all" if modality == "face" else "synthetic"


def available_datasets() -> dict[str, list[dict]]:
    """Наборы, на которых можно обучать сеть (только найденные на диске): модальность -> [{id, title, real}]."""
    from app.modules.fingerprint import real as fp_real
    from app.modules.iris import real as iris_real
    from app.modules.registry import get_module
    fp = [{"id": "synthetic", "title": "Синтетика (12 пальцев × 3 оттиска)", "real": False, "max_subjects": 12}]
    fp += [{"id": x["id"], "title": x["title"], "real": True, "max_subjects": 10} for x in fp_real.sources_info() if x["id"] != fp_real.SOCO]
    ir = [{"id": "synthetic", "title": "Синтетика (12 глаз × 3 снимка)", "real": False, "max_subjects": 12}]
    ir += [{"id": x["id"], "title": x["title"], "real": True, "max_subjects": MAX_EYES} for x in iris_real.sources_info()]
    face = [{"id": x["id"], "title": x["title"], "real": True, "max_subjects": MAX_FACES} for x in get_module("face").info()["sources"]]
    return {"fingerprint": fp, "iris": ir, "face": face}


def load_dataset(modality: str, n_subjects: int, holdout: int = 3, dataset: str | None = None) -> TrainData:
    dataset = dataset or default_dataset(modality)
    if modality == "iris":
        return load_iris(n_subjects, holdout, dataset)
    if modality == "fingerprint":
        return load_fingerprint(n_subjects, holdout, dataset)
    if modality == "face":
        return load_face(n_subjects, dataset)
    raise ValueError(f"Неизвестная биометрия: {modality}")
