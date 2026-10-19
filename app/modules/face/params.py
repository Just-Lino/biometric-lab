from typing import Literal

from pydantic import BaseModel, Field


class FaceParams(BaseModel):
    detector: Literal["auto", "haar", "yunet"] = "auto"
    embedder: Literal["auto", "sface", "lbp"] = "auto"
    align: bool = True
    # Haar-каскад
    scale_factor: float = Field(1.1, ge=1.02, le=1.6)
    min_neighbors: int = Field(5, ge=1, le=15)
    min_face: int = Field(40, ge=20, le=400)
    # YuNet
    yunet_score: float = Field(0.8, ge=0.3, le=0.99)
    # эксперимент: добавочный шум (имитация плохого качества)
    extra_noise: float = Field(0.0, ge=0, le=80)
