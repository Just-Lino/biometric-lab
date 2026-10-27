"""Нейросетевые модели лица. Файлы лежат в папке models/. Если их нет, лаборатория работает на классических методах
(Haar, LBP, HOG и др.), которые ничего не скачивают.

  yunet    - детектор YuNet (OpenCV Zoo), даёт 5 ключевых точек
  ssd      - детектор SSD ResNet-10 (OpenCV), два файла: описание сети и веса
  sface    - признаки SFace (OpenCV Zoo), 128 чисел
  ultraface - лёгкий детектор UltraFace RFB-320 (около 1 МБ, самый быстрый)
  arcface  - признаки ArcFace MobileFaceNet (InsightFace, пакет buffalo_sc), 512 чисел; лицензия модели - только
             для исследований и обучения, не для коммерческого использования
"""
import logging
import shutil
import threading
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("uvicorn.error")
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

# ключ: (имя файла, минимальный размер, адрес, имя файла внутри zip или None)
MODELS = {
    "yunet": ("face_detection_yunet_2023mar.onnx", 100_000,
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", None),
    "sface": ("face_recognition_sface_2021dec.onnx", 10_000_000,
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx", None),
    "ssd_proto": ("ssd_res10_deploy.prototxt", 10_000,
                  "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt", None),
    "ssd_model": ("ssd_res10.caffemodel", 5_000_000,
                  "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", None),
    "ultraface": ("ultraface_rfb320.onnx", 500_000,
                  "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx", None),
    "arcface": ("arcface_mobilefacenet.onnx", 10_000_000,
                "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip", "w600k_mbf.onnx"),
}


def model_path(key: str) -> Path | None:
    name, min_bytes, _, _ = MODELS[key]
    p = MODELS_DIR / name
    return p if p.exists() and p.stat().st_size >= min_bytes else None


def download(key: str, timeout: int = 180) -> bool:
    name, min_bytes, url, member = MODELS[key]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODELS_DIR / (name + ".part")
    zip_tmp = MODELS_DIR / (name + ".zip.part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "biometric-lab"})
        target = zip_tmp if member else tmp
        with urllib.request.urlopen(req, timeout=timeout) as r, open(target, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if member:  # модель лежит внутри zip-архива
            with zipfile.ZipFile(zip_tmp) as z, z.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            zip_tmp.unlink(missing_ok=True)
        if tmp.stat().st_size < min_bytes:
            raise ValueError("файл слишком мал (возможно, это не модель)")
        tmp.replace(MODELS_DIR / name)
        log.info("Модель %s скачана", name)
        return True
    except Exception as e:  # сеть недоступна и т.п.
        log.warning("Не удалось скачать %s: %s", name, e)
        tmp.unlink(missing_ok=True)
        zip_tmp.unlink(missing_ok=True)
        return False


def download_missing() -> dict[str, bool]:
    return {k: (model_path(k) is not None) or download(k) for k in MODELS}


def download_missing_async() -> None:
    if any(model_path(k) is None for k in MODELS):
        threading.Thread(target=download_missing, daemon=True).start()
