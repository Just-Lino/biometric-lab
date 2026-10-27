"""Небольшой потокобезопасный LRU-кэш для результатов конвейеров (запросы FastAPI выполняются в пуле потоков)."""
import threading
from collections import OrderedDict
from typing import Any


class LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._d: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)
