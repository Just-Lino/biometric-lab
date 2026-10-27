"""Сеть, обученная на вкладке «Обучение распознаванию», как ещё один метод распознавания.

После обучения веса сохраняются в data/trained/<модальность>.npz и переживают перезапуск сервера. Вкладки отпечатков, радужки и лица
используют сеть как метод «Нейросеть, обученная вами»: embedding образца и косинусное сходство."""
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.training.cnn import CNN
from app.training.datasets import prep

_lock = threading.Lock()
_cache: dict[str, "TrainedModel"] = {}


@dataclass
class TrainedModel:
    modality: str
    filters: tuple
    size: int
    names: list
    holdout: int
    threshold: float  # порог при EER на образцах, которых сеть не видела при обучении
    eer: float
    test_acc: float
    epochs: int
    config: dict
    trained_at: str
    dataset: str = "synthetic"  # на каких данных обучена сеть (synthetic, fvc2004_db1, casia_interval, LFW ...)
    weights: dict = field(repr=False, default_factory=dict)
    _net: CNN | None = field(default=None, repr=False)

    def net(self) -> CNN:
        if self._net is None:
            net = CNN(self.size, len(self.names), 32, filters=self.filters)
            net.load_state(self.weights)
            self._net = net
        return self._net

    def info(self) -> dict:
        return {"available": True, "threshold": self.threshold, "eer": self.eer, "test_acc": self.test_acc, "epochs": self.epochs,
                "classes": len(self.names), "holdout": self.holdout, "arch": self.config.get("arch", "small"), "dataset": self.dataset,
                "optimizer": self.config.get("optimizer", "adam"), "trained_at": self.trained_at}


def _dir() -> Path:
    return Path(settings.db_path).resolve().parent / "trained"


def store(m: TrainedModel) -> None:
    with _lock:
        _cache[m.modality] = m
        d = _dir()
        d.mkdir(parents=True, exist_ok=True)
        meta = {k: getattr(m, k) for k in ("modality", "filters", "size", "names", "holdout", "threshold", "eer", "test_acc", "epochs", "config", "trained_at", "dataset")}
        np.savez_compressed(d / f"{m.modality}.npz", meta=json.dumps(meta, ensure_ascii=False), **m.weights)


def get(modality: str) -> TrainedModel | None:
    with _lock:
        if modality in _cache:
            return _cache[modality]
        f = _dir() / f"{modality}.npz"
        if not f.exists():
            return None
        try:
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            meta["filters"] = tuple(meta["filters"])
            m = TrainedModel(weights={k: z[k] for k in z.files if k != "meta"}, **meta)
        except Exception:  # повреждённый файл - как будто модели нет
            return None
        _cache[modality] = m
        return m


def reset_cache() -> None:
    with _lock:
        _cache.clear()


def require(modality: str, title: str, dataset: str | None = None) -> TrainedModel:
    """Обученная сеть. Если указан набор данных (dataset), сеть должна быть обучена на нём: сеть, обученная на синтетике, на реальных снимках не работает."""
    m = get(modality)
    if m is None:
        raise ValueError(f"Сеть для «{title}» ещё не обучена. Откройте вкладку «Обучение распознаванию», выберите эти данные и нажмите «Начать обучение».")
    if dataset and m.dataset != dataset:
        raise ValueError(f"Сеть для «{title}» обучена на наборе «{m.dataset}», а выбран «{dataset}». Обучите сеть на выбранном наборе "
                         "(вкладка «Обучение распознаванию») или выберите набор, на котором она обучена.")
    return m


def embed(modality: str, gray: np.ndarray) -> np.ndarray:
    """Единичный вектор embedding образца (серое изображение uint8; для радужки - вырезка вокруг радужки 96×96, для лица - выровненное лицо)."""
    m = require(modality, modality)
    x = prep(gray, m.size)[None, None]
    with _lock:
        e = m.net().forward(x)[1][0]
    return (e / (np.linalg.norm(e) + 1e-9)).astype(np.float32)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
