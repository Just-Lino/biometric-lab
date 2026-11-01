from typing import Literal

from pydantic import BaseModel, Field


class VoiceParams(BaseModel):
    source: str = Field("synthetic", max_length=40)  # набор данных (пока только синтетические дикторы; свои записи загружаются отдельно)
    method: Literal["mfcc", "mfcc_delta", "pitch", "combo"] = "mfcc"
    vad: Literal["adaptive", "fixed", "zcr"] = "adaptive"  # как отделять речь от тишины
    vad_level: float = Field(30.0, ge=10, le=50)  # для «fixed»: сколько дБ ниже самого громкого кадра считать речью
    n_mfcc: int = Field(13, ge=8, le=20)  # число коэффициентов MFCC в шаблоне
    extra_noise: float = Field(0.0, ge=0, le=60)  # добавочный шум, % от громкости записи
