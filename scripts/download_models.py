"""Скачать модели лица (YuNet, SFace) в папку models/.   python -m scripts.download_models"""
from app.modules.face import models

if __name__ == "__main__":
    print("Папка моделей:", models.MODELS_DIR)
    for key, ok in models.download_missing().items():
        print(f"{key}: {'готово' if ok else 'НЕ СКАЧАНО (проверьте интернет)'}")
