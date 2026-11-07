"""Озвучка на сервере: голоса Piper (ONNX, процессор): русские и казахские. Голоса лежат в models/tts/ (скачивает `python -m scripts.download_tts`,
в Git не попадают). Если пакет piper-tts или файлы голосов отсутствуют, озвучка недоступна, а интерфейс использует голос браузера.

Движок Piper: GPL-3.0-or-later (ставится отдельно, requirements-tts.txt). Голоса Дмитрий и Денис обучены на данных CC0 (OHF-Voice/voice-datasets).
Казахские голоса - модель kk_KZ-issai-high (корпус ISSAI KazakhTTS, Nazarbayev University; при использовании указывайте авторство и не создавайте
оскорбительный или дискриминационный контент): одна модель с несколькими дикторами, выбор по номеру диктора."""
import io
import logging
import threading
import wave
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("uvicorn.error")
TTS_DIR = Path(__file__).resolve().parents[2] / "models" / "tts"
ISSAI = "kk_KZ-issai-high"
VOICES = [{"id": "dmitri", "title": "Дмитрий (Piper, мужской)", "lang": "ru", "stem": "ru_RU-dmitri-medium"},
          {"id": "denis", "title": "Денис (Piper, мужской)", "lang": "ru", "stem": "ru_RU-denis-medium"},
          {"id": "kk_raya", "title": "Райя (Piper, женский)", "lang": "kk", "stem": ISSAI, "speaker": 3},
          {"id": "kk_iseke", "title": "Исеке (Piper, мужской)", "lang": "kk", "stem": ISSAI, "speaker": 1},
          {"id": "kk_f2", "title": "Дикторша 2 (Piper, женский)", "lang": "kk", "stem": ISSAI, "speaker": 4},
          {"id": "kk_m2", "title": "Диктор 2 (Piper, мужской)", "lang": "kk", "stem": ISSAI, "speaker": 0}]
MAX_CHARS = 600
_lock = threading.Lock()  # синтез по одной фразе: процессор делят сервер, конвейеры и обучение сети


def _voice(voice_id: str) -> dict | None:
    return next((v for v in VOICES if v["id"] == voice_id), None)


def _paths(voice_id: str) -> tuple[Path, Path]:
    stem = _voice(voice_id)["stem"]
    return TTS_DIR / f"{stem}.onnx", TTS_DIR / f"{stem}.onnx.json"


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


@lru_cache(maxsize=3)
def _load_stem(stem: str):
    from piper import PiperVoice
    return PiperVoice.load(str(TTS_DIR / f"{stem}.onnx"), config_path=str(TTS_DIR / f"{stem}.onnx.json"))


def _load(voice_id: str):
    return _load_stem(_voice(voice_id)["stem"])


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
            sp = _voice(voice_id).get("speaker")
            if sp is None:
                voice.synthesize_wav(text, w)
            else:
                from piper import SynthesisConfig
                voice.synthesize_wav(text, w, syn_config=SynthesisConfig(speaker_id=sp))
    return buf.getvalue()
