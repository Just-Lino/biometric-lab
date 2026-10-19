"""Генерация синтетического датасета отпечатков в datasets/synthetic (запускать один раз).
    python -m scripts.generate_dataset
"""
import cv2

from app.modules.fingerprint import synthetic
from app.modules.fingerprint.module import N_FINGERS, N_IMPRESSIONS

if __name__ == "__main__":
    synthetic.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for f in range(1, N_FINGERS + 1):
        for i in range(1, N_IMPRESSIONS + 1):
            cv2.imwrite(str(synthetic.DATASET_DIR / f"f{f:02d}_i{i}.png"), synthetic._generate_impression(f, i))
        print(f"палец {f:02d} готов")
