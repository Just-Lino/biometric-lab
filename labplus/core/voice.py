"""Голосовой шаблон: регистрация, проверка «свой/чужой». В профиле только числа; записи фраз лежат отдельными WAV, чтобы их можно было прослушать, и удаляются вместе с профилем."""
from __future__ import annotations

import json
import os
import hashlib
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .audio import FEATURE_DIM, AudioError

MIN_ENROLL = 3
MAX_ENROLL = 8
_LOCK = threading.Lock()
SCORE_SCALE = 1.5          # масштаб перевода расстояния в оценку сходства 0..1


def data_dir() -> Path:
    p = Path(os.environ.get("LABPLUS_DATA", "labplus_data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def audio_dir(name: str) -> Path:
    """Папка с записями профиля; имя папки - хэш имени, чтобы не зависеть от символов в имени."""
    return data_dir() / "voice_audio" / hashlib.sha1((name or "").strip().encode("utf-8")).hexdigest()[:16]


def save_phrases(name: str, wavs: list[bytes]) -> None:
    d = audio_dir(name)
    with _LOCK:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        for i, w in enumerate(wavs):
            (d / f"{i}.wav").write_bytes(w)


def phrase_count(name: str) -> int:
    d = audio_dir(name)
    return len(list(d.glob("*.wav"))) if d.is_dir() else 0


def phrase_path(name: str, i: int) -> Path | None:
    p = audio_dir(name) / f"{int(i)}.wav"
    return p if p.is_file() else None


def _store() -> Path:
    return data_dir() / "voice_profiles.json"


def _load() -> dict:
    f = _store()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(db: dict) -> None:
    _store().write_text(json.dumps(db, ensure_ascii=False, indent=1), encoding="utf-8")


def clean_name(name: str) -> str:
    name = (name or "").strip()
    if not re.fullmatch(r"[\w\- ]{1,32}", name, flags=re.UNICODE):
        raise AudioError("Имя профиля: 1–32 символа (буквы, цифры, пробел, дефис). Лучше псевдоним, не ФИО.")
    return name


def build_template(vectors: list[np.ndarray]) -> dict:
    """Шаблон = среднее по фразам + разброс по каждому признаку (с нижней границей)."""
    if len(vectors) < MIN_ENROLL:
        raise AudioError(f"Для шаблона нужно минимум {MIN_ENROLL} фразы, получено {len(vectors)}.")
    V = np.stack(vectors)
    mu = V.mean(axis=0)
    sd = V.std(axis=0, ddof=1)
    sigma = np.maximum.reduce([sd, 0.15 * np.abs(mu), np.full_like(mu, 0.15)])
    return {"mu": mu.tolist(), "sigma": sigma.tolist(), "dim": int(len(mu)), "n_phrases": len(vectors)}


def distance(vec: np.ndarray, tpl: dict) -> float:
    mu, sg = np.asarray(tpl["mu"]), np.asarray(tpl["sigma"])
    return float(np.sqrt((((vec - mu) / sg) ** 2).mean()))


def score_from_distance(d: float) -> float:
    """Нормированное расстояние → оценка сходства в [0,1] (1 — идеально совпало)."""
    return float(np.exp(-0.5 * (d / SCORE_SCALE) ** 2))


def enroll(name: str, vectors: list[np.ndarray]) -> dict:
    name = clean_name(name)
    if len(vectors) > MAX_ENROLL:
        raise AudioError(f"Слишком много фраз (максимум {MAX_ENROLL}).")
    tpl = build_template(vectors)
    # самопроверка: насколько стабилен голос между фразами (leave-one-out)
    loo = []
    for i in range(len(vectors)):
        rest = [v for j, v in enumerate(vectors) if j != i]
        if len(rest) >= 2:
            mu = np.mean(rest, axis=0)
            sd = np.maximum.reduce([np.std(rest, axis=0, ddof=1), 0.15 * np.abs(mu), np.full_like(mu, 0.15)])
            loo.append(score_from_distance(float(np.sqrt((((vectors[i] - mu) / sd) ** 2).mean()))))
    tpl["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tpl["self_consistency"] = float(np.mean(loo)) if loo else None
    with _LOCK:
        db = _load()
        db[name] = tpl
        _save(db)
    return {"name": name, "dim": tpl["dim"], "n_phrases": tpl["n_phrases"],
            "self_consistency": tpl["self_consistency"], "template_preview": [round(v, 2) for v in tpl["mu"][:8]]}


def get_profile(name: str) -> dict | None:
    return _load().get((name or "").strip())


def list_profiles() -> list[dict]:
    return [{"name": n, "n_phrases": p["n_phrases"], "created": p.get("created", ""),
             "dim": p["dim"], "n_audio": phrase_count(n)} for n, p in _load().items()]


def delete_profile(name: str) -> bool:
    with _LOCK:
        db = _load()
        ok = db.pop((name or "").strip(), None) is not None
        _save(db)
        shutil.rmtree(audio_dir(name), ignore_errors=True)
    return ok


def verify(name: str, vec: np.ndarray, threshold: float) -> dict:
    tpl = get_profile(name)
    if tpl is None:
        raise AudioError("Профиль не найден. Сначала зарегистрируйте голос.")
    if len(vec) != tpl["dim"] or tpl["dim"] != FEATURE_DIM:
        raise AudioError("Шаблон создан другой версией алгоритма. Зарегистрируйте голос заново.")
    d = distance(vec, tpl)
    s = score_from_distance(d)
    return {"score": round(s, 4), "distance": round(d, 3), "threshold": threshold,
            "decision": "MATCH" if s >= threshold else "NO MATCH"}
