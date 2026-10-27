"""Озвучка на сервере: русские голоса Piper (ONNX, процессор). Голоса лежат в models/tts/ (скачивает `python -m scripts.download_tts`,
в Git не попадают). Если пакет piper-tts или файлы голосов отсутствуют, озвучка недоступна, а интерфейс использует голос браузера.

Движок Piper: GPL-3.0-or-later (ставится отдельно, requirements-tts.txt). Голоса Дмитрий и Денис обучены на данных CC0 (OHF-Voice/voice-datasets)."""
import io
import logging
import threading
import wave
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("uvicorn.error")
TTS_DIR = Path(__file__).resolve().parents[2] / "models" / "tts"
VOICES = [{"id": "dmitri", "title": "Дмитрий (Piper, мужской)"}, {"id": "denis", "title": "Денис (Piper, мужской)"}]
MAX_CHARS = 600
_lock = threading.Lock()  # синтез по одной фразе: процессор делят сервер, конвейеры и обучение сети


def _paths(voice_id: str) -> tuple[Path, Path]:
    return TTS_DIR / f"ru_RU-{voice_id}-medium.onnx", TTS_DIR / f"ru_RU-{voice_id}-medium.onnx.json"


def engine_available() -> bool:
    try:
        import piper  # noqa: F401
        return True
    except Exception:
        return False


def voice_files_present() -> bool:
    """Скачаны ли файлы голосов (отдельно от того, установлен ли движок): нужно, чтобы подсказать, чего именно не хватает."""
    return any(_paths(v["id"])[0].exists() and _paths(v["id"])[1].exists() for v in VOICES)


def available_voices() -> list[dict]:
    """Голоса, которые можно использовать прямо сейчас (движок установлен и файлы голоса скачаны)."""
    if not engine_available():
        return []
    return [v for v in VOICES if _paths(v["id"])[0].exists() and _paths(v["id"])[0].stat().st_size > 10_000_000 and _paths(v["id"])[1].exists()]


@lru_cache(maxsize=2)
def _load(voice_id: str):
    from piper import PiperVoice
    model, cfg = _paths(voice_id)
    return PiperVoice.load(str(model), config_path=str(cfg))


_warming = False


def warm_up(voice_id: str) -> None:
    """Загрузить голос в фоне (около 10 секунд), чтобы первая озвучка не ждала."""
    global _warming
    if _warming:
        return
    _warming = True

    def run():
        try:
            with _lock:
                _load(voice_id)
        except Exception as e:  # noqa: BLE001
            log.warning("Не удалось загрузить голос %s: %s", voice_id, e)

    threading.Thread(target=run, daemon=True).start()


def synthesize(text: str, voice_id: str) -> bytes:
    """WAV (16 бит, моно) с озвучкой текста. ValueError - понятное сообщение для пользователя."""
    text = " ".join((text or "").split())
    if not text:
        raise ValueError("Пустой текст.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Слишком длинный текст для озвучки (не больше {MAX_CHARS} символов).")
    if voice_id not in {v["id"] for v in available_voices()}:
        raise ValueError("Голос не установлен. Выполните: pip install -r requirements-tts.txt и python -m scripts.download_tts")
    with _lock:
        voice = _load(voice_id)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            voice.synthesize_wav(text, w)
    return buf.getvalue()
