import sys

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services import assistant, tts
from app.services.events import record_event

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])


class ParseIn(BaseModel):
    text: str = Field("", max_length=500)
    context: dict = Field(default_factory=dict)  # mod: текущая вкладка
    alternatives: list[str] = Field(default_factory=list, max_length=5)  # другие варианты распознавания речи


@router.post("/parse")
def parse(body: ParseIn):
    """Фраза -> список действий для интерфейса (см. app/services/assistant.py)."""
    res = assistant.parse(body.text, body.context, [a[:500] for a in body.alternatives])
    record_event("assistant", {"text": body.text[:200], "understood": res["understood"], "actions": [a["do"] for a in res["actions"]]})
    return res


class TtsIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=tts.MAX_CHARS)
    voice: str = Field(..., max_length=20)


@router.get("/voices")
def voices():
    """Серверные голоса озвучки (Piper), если движок и файлы голосов установлены."""
    v = tts.available_voices()
    if v:
        tts.warm_up(v[0]["id"])
    return {"voices": v, "engine": tts.engine_available(), "python": sys.executable, "files": tts.voice_files_present()}


@router.post("/tts")
def speak(body: TtsIn):
    """Озвучка фразы выбранным серверным голосом: WAV."""
    try:
        wav = tts.synthesize(body.text, body.voice)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.get("/examples")
def examples():
    return {"examples": assistant.EXAMPLES, "help": assistant.HELP}
