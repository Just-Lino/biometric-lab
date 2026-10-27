from typing import Literal

from pydantic import BaseModel, Field

Localizer = Literal["auto", "daugman", "hough", "contour"]
Encoder = Literal["gabor", "loggabor", "ordinal", "dct", "haar", "cnn"]


class IrisParams(BaseModel):
    source: str = Field("synthetic", max_length=40)  # набор данных для оценки качества и проверки обученной сети: synthetic, casia_interval
    localizer: Localizer = "auto"
    encoder: Encoder = "gabor"
    blur: float = Field(2.0, ge=0.5, le=6.0)  # сглаживание перед поиском границ (σ)
    hough_quality: float = Field(0.6, ge=0.3, le=0.95)  # Хаф: порог «совершенства» окружности (меньше - больше кандидатов)
    mask_eyelids: bool = True  # исключать веки, ресницы и блики из сравнения
    wavelength: float = Field(16.0, ge=6.0, le=48.0)  # длина волны фильтров Габора / лог-Габора (пиксели развёртки)
    max_shift: int = Field(10, ge=0, le=24)  # компенсация поворота глаза: сдвиги развёртки ±, пиксели
    extra_noise: float = Field(0.0, ge=0, le=60)
