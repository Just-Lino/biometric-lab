"""Скачать нейросетевые модели в папку models/: YuNet, SFace, SSD ResNet-10 (2 файла), UltraFace, ArcFace MobileFaceNet.
   python -m scripts.download_models   (нужен интернет, около 40 МБ; без моделей работают все классические методы)"""
from app.modules.face import models

TITLES = {"yunet": "детектор YuNet", "sface": "признаки SFace", "ssd_proto": "детектор SSD (описание сети)", "ssd_model": "детектор SSD (веса)",
          "ultraface": "детектор UltraFace", "arcface": "признаки ArcFace MobileFaceNet (лицензия: только исследования и обучение)"}

if __name__ == "__main__":
    print("Папка моделей:", models.MODELS_DIR)
    for key, ok in models.download_missing().items():
        print(f"{TITLES.get(key, key)}: {'готово' if ok else 'НЕ СКАЧАНО (проверьте интернет)'}")
