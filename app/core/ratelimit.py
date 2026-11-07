"""Ограничитель частоты запросов: не больше N обращений за окно времени на один ключ (например, сессию посетителя).

ВАЖНО: счётчики хранятся в памяти процесса. Пока сервер работает одним экземпляром (или с привязкой сессии к экземпляру),
лимит точный. При нескольких экземплярах без привязки сессий у каждого будет свой счётчик и фактический лимит вырастет
во столько раз, сколько экземпляров; тогда нужно общее хранилище (например, Redis или таблица в PostgreSQL)."""
import math
import threading
import time
from collections import OrderedDict, deque
from typing import Callable


class SlidingWindowLimiter:
    def __init__(self, window: float = 3600.0, max_keys: int = 20_000, clock: Callable[[], float] = time.monotonic):
        self.window, self.max_keys, self._clock = window, max_keys, clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> tuple[bool, int]:
        """Регистрирует обращение. (True, 0): можно; (False, N): лимит исчерпан, повторить через N секунд.
        limit <= 0 отключает ограничение. Отклонённое обращение окно не продлевает, поэтому ожидание не растёт от повторных попыток."""
        if limit <= 0:
            return True, 0
        now = self._clock()
        with self._lock:
            q = self._hits.get(key)
            if q is None:
                q = self._hits[key] = deque()
            self._hits.move_to_end(key)
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, math.ceil(q[0] + self.window - now))
            q.append(now)
            if len(self._hits) > self.max_keys:
                self._prune(now)
            return True, 0

    def _prune(self, now: float) -> None:
        """Память под ключи ограничена: сначала выбрасываем ключи с истёкшим окном, затем самые давние."""
        for k in [k for k, q in self._hits.items() if not q or q[-1] <= now - self.window]:
            del self._hits[k]
        while len(self._hits) > self.max_keys:
            self._hits.popitem(last=False)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
