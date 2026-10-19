from typing import Literal

from pydantic import BaseModel, Field


class FingerprintParams(BaseModel):
    # предобработка
    clahe_clip: float = Field(2.0, ge=0.5, le=10)
    seg_threshold: float = Field(18.0, ge=2, le=80)
    orient_smooth: float = Field(8.0, ge=2, le=20)
    # Gabor
    gabor_period: float = Field(9.0, ge=4, le=16)
    gabor_sigma: float = Field(4.0, ge=1.5, le=10)
    # бинаризация
    bin_method: Literal["adaptive", "otsu", "fixed"] = "adaptive"
    bin_threshold: int = Field(128, ge=20, le=230)
    # minutiae
    min_minutiae_dist: int = Field(8, ge=3, le=30)
    border_margin: int = Field(14, ge=0, le=40)
    spur_length: int = Field(7, ge=0, le=20)
    # matching
    dist_tol: float = Field(14.0, ge=3, le=40)
    angle_tol_deg: float = Field(20.0, ge=3, le=60)
    max_rotation_deg: float = Field(40.0, ge=5, le=90)
    # эксперимент: добавочный шум для имитации плохого качества
    extra_noise: float = Field(0.0, ge=0, le=80)
