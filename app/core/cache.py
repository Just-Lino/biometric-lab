"""Небольшой потокобезопасный LRU-кэш (запросы FastAPI выполняются в пуле потоков).

Нужен для результатов конвейеров и для ответов ИИ-наставника. Необязательное время жизни записи (ttl, секунды):
по его истечении запись считается отсутствующей, поэтому устаревший ответ не живёт в памяти бесконечно.
Кэш хранится в памяти процесса: при нескольких экземплярах сервера у каждого он свой."""
import threading
import time
from collections import OrderedDict
from typing import Any, Callable


class LRUCache:
    def __init__(self, maxsize: int, ttl: float | None = None, clock: Callable[[], float] = time.monotonic):
        self.maxsize, self.ttl, self._clock = maxsize, ttl, clock
        self._d: OrderedDict[str, tuple[float | None, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            expires, value = item
            if expires is not None and self._clock() >= expires:
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        expires = None if self.ttl is None else self._clock() + self.ttl
        with self._lock:
            self._d[key] = (expires, value)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)
