"""Где лежат реальные открытые наборы (datasets/real). Папка не входит в Git; наборы кладёт пользователь
(скрипт `python -m scripts.prepare_real` копирует CASIA и SOCOFing из скачанных папок).

Путь читается при каждом обращении (realdata.REAL_DIR), поэтому тесты подменяют его на временную папку."""
from pathlib import Path

REAL_DIR = Path(__file__).resolve().parents[2] / "datasets" / "real"

IMG_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def subdir(name: str) -> Path:
    return REAL_DIR / name


def matches(query: str, *fields: str) -> bool:
    """Поиск по подстроке без учёта регистра (пустой запрос подходит всему)."""
    q = query.strip().lower()
    return not q or any(q in f.lower() for f in fields)
