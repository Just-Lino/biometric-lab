from typing import Literal

from pydantic import BaseModel, Field

Detector = Literal["auto", "haar", "haar_alt2", "haar_alt", "haar_profile", "ssd", "ultraface", "yunet"]
Embedder = Literal["auto", "sface", "arcface", "lbp", "hog", "gabor", "dct", "pixels", "eigenfaces", "cnn"]


class FaceParams(BaseModel):
    detector: Detector = "auto"
    embedder: Embedder = "auto"
    align: bool = True
    source: str = Field("all", max_length=40)  # источник галереи (all, LFW, gallery): оттуда берутся люди для оценки качества и «среднего лица»
    # Haar-каскады
    scale_factor: float = Field(1.1, ge=1.02, le=1.6)
    min_neighbors: int = Field(5, ge=1, le=15)
    min_face: int = Field(40, ge=20, le=400)
    # нейросетевые детекторы (YuNet, SSD): порог уверенности
    yunet_score: float = Field(0.8, ge=0.3, le=0.99)
    # эксперимент: добавочный шум (имитация плохого качества)
    extra_noise: float = Field(0.0, ge=0, le=80)
