"""Единый контракт биометрического модуля.

Чтобы добавить новую биометрию (face, iris, retina), создайте папку в app/modules/,
реализуйте BiometricModule и зарегистрируйте её в registry.py.
API и интерфейс подхватят модуль без изменений остального кода."""
from abc import ABC, abstractmethod
from typing import Any


class BiometricModule(ABC):
    name: str = ""
    upload_kind: str = "image"  # что загружают в модуль: image или audio
    title: str = ""
    default_threshold: float = 0.5

    @abstractmethod
    def list_samples(self, source: str | None = None, q: str = "", limit: int = 0) -> list[dict[str, Any]]:
        """Образцы выбранного набора данных (source) плюс загрузки текущего посетителя; q - поиск по подписи, limit - не больше N образцов (0 - все)."""

    @abstractmethod
    def get_sample_preview(self, sample_id: str, params: dict) -> bytes: ...

    @abstractmethod
    def process(self, sample_id: str, params: dict) -> dict[str, Any]:
        """Пошаговая обработка образца: шаги с изображениями + шаблон."""

    @abstractmethod
    def compare(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        """Сравнение двух образцов: score, соответствия, визуализация."""

    def evaluate(self, params: dict, n_subjects: int = 10) -> dict[str, Any]:
        raise NotImplementedError

    def has_upload(self, sample_id: str) -> bool:
        """Есть ли такая загрузка у текущего посетителя (хранилище загрузок модуля лежит в `_uploads`)."""
        store = getattr(self, "_uploads", None)
        return store is not None and store.get(sample_id) is not None

    def upload(self, data: bytes) -> str:
        raise NotImplementedError

    def relief(self, sample_id: str, params: dict) -> dict[str, Any]:
        """Данные для 3D-визуализации (карта высот, текстуры, ключевые точки)."""
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        """Служебная информация для интерфейса (доступные модели, наборы данных sources, размер галереи и т.п.)."""
        return {}

    def stress(self, params: dict, n_subjects: int = 12) -> dict[str, Any]:
        """Стресс-тест устойчивости к искажениям (только там, где есть набор с искажёнными копиями)."""
        raise NotImplementedError

    def compare_detectors(self, sample_id: str, params: dict) -> dict[str, Any]:
        raise NotImplementedError

    def compare_embedders(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        raise NotImplementedError

    def evaluate_embedders(self, params: dict, n_subjects: int = 8) -> dict[str, Any]:
        raise NotImplementedError
