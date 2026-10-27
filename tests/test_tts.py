"""Озвучка на сервере (Piper): список голосов, синтез, ошибки и откат на голос браузера, если голосов нет."""
import io
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import tts

have_voice = bool(tts.available_voices())
need_voice = pytest.mark.skipif(not have_voice, reason="голос Piper не установлен (pip install -r requirements-tts.txt, python -m scripts.download_tts)")


def test_voices_endpoint_lists_only_installed_voices():
    with TestClient(app) as c:
        r = c.get("/api/v1/assistant/voices").json()
        assert {"voices", "engine", "python", "files"} <= set(r) and all({"id", "title"} <= set(v) for v in r["voices"])
        assert [v["id"] for v in r["voices"]] == [v["id"] for v in tts.available_voices()]


def test_without_voices_synthesis_is_refused_politely(monkeypatch):
    monkeypatch.setattr(tts, "available_voices", lambda: [])
    with TestClient(app) as c:
        assert c.get("/api/v1/assistant/voices").json()["voices"] == []
        r = c.post("/api/v1/assistant/tts", json={"text": "Привет", "voice": "dmitri"})
        assert r.status_code == 400 and "не установлен" in r.json()["detail"]


def test_input_limits():
    with TestClient(app) as c:
        assert c.post("/api/v1/assistant/tts", json={"text": "", "voice": "dmitri"}).status_code == 422
        assert c.post("/api/v1/assistant/tts", json={"text": "а" * (tts.MAX_CHARS + 1), "voice": "dmitri"}).status_code == 422
        assert c.post("/api/v1/assistant/tts", json={"text": "Привет", "voice": "../../etc/passwd"}).status_code == 400


@need_voice
def test_synthesis_returns_wav():
    voice = tts.available_voices()[0]["id"]
    with TestClient(app) as c:
        r = c.post("/api/v1/assistant/tts", json={"text": "Покажу, как изображение превращается в решение.", "voice": voice})
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
        w = wave.open(io.BytesIO(r.content))
        assert w.getnchannels() == 1 and w.getnframes() / w.getframerate() > 1.5  # реальная речь, а не тишина
