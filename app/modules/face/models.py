"""Нейросетевые модели лица (OpenCV Zoo): YuNet (детектор) и SFace (embeddings).
Файлы лежат в папке models/. Если их нет, лаборатория работает на Haar + LBP (классический вариант)."""
import logging
import threading
import urllib.request
from pathlib import Path

log = logging.getLogger("uvicorn.error")
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

MODELS = {
    "yunet": ("face_detection_yunet_2023mar.onnx", 100_000,
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "sface": ("face_recognition_sface_2021dec.onnx", 10_000_000,
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
}


def model_path(key: str) -> Path | None:
    name, min_bytes, _ = MODELS[key]
    p = MODELS_DIR / name
    return p if p.exists() and p.stat().st_size >= min_bytes else None


def download(key: str, timeout: int = 120) -> bool:
    name, min_bytes, url = MODELS[key]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODELS_DIR / (name + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "biometric-lab"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if tmp.stat().st_size < min_bytes:
            raise ValueError("файл слишком мал (возможно, это не модель)")
        tmp.replace(MODELS_DIR / name)
        log.info("Модель %s скачана", name)
        return True
    except Exception as e:  # сеть недоступна и т.п.
        log.warning("Не удалось скачать %s: %s", name, e)
        tmp.unlink(missing_ok=True)
        return False


def download_missing() -> dict[str, bool]:
    return {k: (model_path(k) is not None) or download(k) for k in MODELS}


def download_missing_async() -> None:
    if any(model_path(k) is None for k in MODELS):
        threading.Thread(target=download_missing, daemon=True).start()
