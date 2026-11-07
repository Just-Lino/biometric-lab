"""Загруженные посетителем изображения: только в памяти процесса, раздельно по сессиям.

Файлы на диск не пишутся и пропадают при перезапуске сервера. Чужие загрузки не видны: образец доступен только
той сессии, которая его загрузила (см. app/core/session.py)."""
import re
import threading
import uuid
from collections import OrderedDict

import numpy as np

from app.core.session import current_sid


# Идентификатор загрузки: по нему API отличает «загрузка потеряна» (перезапуск, чужая сессия) от неизвестного образца набора данных.
UPLOAD_ID = re.compile(r"up_[0-9a-f]{6}")


class UploadStore:
    def __init__(self, per_session: int = 20, total: int = 200):
        self._d: OrderedDict[str, tuple[str | None, np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()
        self.per_session, self.total = per_session, total

    def add(self, img: np.ndarray) -> str:
        sid, uid = current_sid(), "up_" + uuid.uuid4().hex[:6]
        with self._lock:
            self._d[uid] = (sid, img)
            mine = [k for k, (o, _) in self._d.items() if o == sid]
            while len(mine) > self.per_session:  # у каждой сессии не больше per_session снимков, старые вытесняются
                self._d.pop(mine.pop(0), None)
            while len(self._d) > self.total:
                self._d.popitem(last=False)
        return uid

    def get(self, uid: str) -> np.ndarray | None:
        with self._lock:
            v = self._d.get(uid)
        return v[1] if v is not None and v[0] == current_sid() else None

    def __contains__(self, uid: str) -> bool:
        return self.get(uid) is not None

    def ids(self) -> list[str]:
        sid = current_sid()
        with self._lock:
            return [k for k, (o, _) in self._d.items() if o == sid]
