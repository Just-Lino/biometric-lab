"""Единый контракт биометрического модуля.

Чтобы добавить новую биометрию (face, iris, retina), создайте папку в app/modules/,
реализуйте BiometricModule и зарегистрируйте её в registry.py.
API и интерфейс подхватят модуль без изменений остального кода."""
from abc import ABC, abstractmethod
from typing import Any


class BiometricModule(ABC):
    name: str = ""
    title: str = ""

    @abstractmethod
    def list_samples(self) -> list[dict[str, Any]]: ...

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

    def upload(self, data: bytes) -> str:
        raise NotImplementedError

    def relief(self, sample_id: str, params: dict) -> dict[str, Any]:
        """Данные для 3D-визуализации (карта высот, текстуры, ключевые точки)."""
        raise NotImplementedError
