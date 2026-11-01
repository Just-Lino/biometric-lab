"""FastAPI-роутер LAB+: страницы /labplus/* и API /api/labplus/*."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .core import audio, explain, imaging, intent, metrics, standards
from .core import voice as voice_core

router = APIRouter()
STATIC = Path(__file__).parent / "static"
PAGES = {"voice", "metrics", "robustness", "standards"}
MAX_AUDIO = 6 * 1024 * 1024
MAX_IMAGE = 8 * 1024 * 1024


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except (audio.AudioError, metrics.MetricsError, imaging.ImagingError) as e:
        raise HTTPException(status_code=400, detail=str(e))


def _read_limited(f: UploadFile, limit: int) -> bytes:
    data = f.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="Файл слишком большой.")
    if not data:
        raise HTTPException(status_code=400, detail="Пустой файл.")
    return data


# ───────────────────────── страницы ─────────────────────────
@router.get("/labplus", include_in_schema=False)
def hub():
    return RedirectResponse("/labplus/voice")


@router.get("/labplus/static/{name}", include_in_schema=False)
def static_file(name: str):
    if name not in {"labplus.css", "labplus.js"}:
        raise HTTPException(404)
    return FileResponse(STATIC / name)


@router.get("/labplus/{page}", include_in_schema=False)
def page(page: str):
    if page not in PAGES:
        raise HTTPException(404)
    return FileResponse(STATIC / f"{page}.html")


@router.get("/api/labplus/health")
def health():
    return {"ok": True, "version": "1.0.0", "feature_dim": audio.FEATURE_DIM}


# ───────────────────────── голос ─────────────────────────
def _analyze_upload(f: UploadFile):
    return _guard(audio.analyze, _guard(audio.read_wav, _read_limited(f, MAX_AUDIO)))


@router.post("/api/labplus/voice/analyze")
def voice_analyze(file: UploadFile = File(...)):
    a = _analyze_upload(file)
    return {"viz": audio.viz_payload(a)}


@router.post("/api/labplus/voice/enroll")
def voice_enroll(name: str = Form(...), files: List[UploadFile] = File(...)):
    vecs, quality, raws = [], [], []
    last = None
    for i, f in enumerate(files, 1):
        try:
            raw = _read_limited(f, MAX_AUDIO)
            a = _guard(audio.analyze, _guard(audio.read_wav, raw))
            raws.append(raw)
        except HTTPException as e:
            raise HTTPException(400, f"Фраза {i}: {e.detail}")
        vecs.append(a["vector"])
        quality.append({"speech_sec": round(a["speech_sec"], 2), "f0_hz": None if a["f0_hz"] is None else round(a["f0_hz"], 1)})
        last = a
    info = _guard(voice_core.enroll, name, vecs)
    voice_core.save_phrases(info["name"], raws)
    info["n_audio"] = len(raws)
    info["phrases"] = quality
    info["viz"] = audio.viz_payload(last)
    return info


@router.post("/api/labplus/voice/verify")
def voice_verify(name: str = Form(...), threshold: float = Form(0.5), file: UploadFile = File(...)):
    a = _analyze_upload(file)
    res = _guard(voice_core.verify, name, a["vector"], float(threshold))
    res["viz"] = audio.viz_payload(a)
    res["quality"] = {"speech_sec": round(a["speech_sec"], 2), "clipping": round(a["clipping"], 4), "f0_hz": res["viz"]["f0_hz"]}
    res["feature_vector"] = [round(v, 2) for v in a["vector"].tolist()]
    return res


@router.post("/api/labplus/voice/robustness")
def voice_robustness(name: str = Form(...), threshold: float = Form(0.5), file: UploadFile = File(...)):
    raw = _guard(audio.read_wav, _read_limited(file, MAX_AUDIO))
    base = _guard(audio.analyze, raw)
    ref = _guard(voice_core.verify, name, base["vector"], threshold)["score"]
    out = []
    for kind, (title, unit, values) in audio.AUDIO_ATTACKS.items():
        pts = []
        for k, v in enumerate(values):
            rejected = False
            try:
                a = audio.analyze(audio.apply_attack(raw, kind, v, seed=0))
                s = voice_core.verify(name, a["vector"], threshold)["score"]
            except audio.AudioError:       # запись после искажения непригодна (речь не найдена)
                s, rejected = 0.0, True
            pts.append({"value": v, "score": s, "pass": s >= threshold, "rejected": rejected})
        bp = next((p["value"] for p in pts if not p["pass"]), None)
        out.append({"kind": kind, "title": title, "unit": unit, "points": pts, "breakpoint": bp})
    return {"reference_score": ref, "threshold": threshold, "results": out}


@router.get("/api/labplus/voice/profiles")
def voice_profiles():
    return {"profiles": voice_core.list_profiles(), "min_phrases": voice_core.MIN_ENROLL, "max_phrases": voice_core.MAX_ENROLL}


@router.get("/api/labplus/voice/profiles/{name}/phrase/{i}.wav")
def voice_phrase(name: str, i: int):
    p = voice_core.phrase_path(name, i)
    if p is None:
        raise HTTPException(404, "Запись не найдена.")
    return FileResponse(p, media_type="audio/wav")


@router.delete("/api/labplus/voice/profiles/{name}")
def voice_delete(name: str):
    if not voice_core.delete_profile(name):
        raise HTTPException(404, "Профиль не найден.")
    return {"deleted": name}


@router.post("/api/labplus/voice/command")
def voice_command(text: str = Form(...), mode: str = Form("auto"), name: str = Form(""),
                  threshold: float = Form(0.5), require_speaker: bool = Form(False),
                  file: Optional[UploadFile] = File(None)):
    parsed = intent.parse(text)
    speaker = None
    allowed = True
    if require_speaker and name:
        if file is None:
            allowed, speaker = False, {"error": "Нет записи голоса для проверки говорящего."}
        else:
            try:
                a = audio.analyze(audio.read_wav(_read_limited(file, MAX_AUDIO)))
                speaker = voice_core.verify(name, a["vector"], float(threshold))
                allowed = speaker["decision"] == "MATCH"
            except audio.AudioError as e:
                allowed, speaker = False, {"error": str(e)}
    key = {"hint": "reply_hint", "confirm": "reply_confirm"}.get(mode, "reply_run")
    reply = parsed[key]
    if not allowed:
        reply = "Голос не узнан, команда не выполнена. Попробуйте ещё раз или отключите проверку говорящего."
    parsed.pop("reply_run"), parsed.pop("reply_hint"), parsed.pop("reply_confirm")
    return {"parsed": parsed, "speaker": speaker, "allowed": allowed, "reply": reply, "mode": mode}


# ───────────────────────── метрики ─────────────────────────
class ScoresIn(BaseModel):
    mode: str = Field("synthetic", pattern="^(synthetic|scores|csv)$")
    preset: Optional[str] = None
    n_genuine: int = 500
    n_impostor: int = 5000
    mu_g: float = 0.75
    sd_g: float = 0.10
    mu_i: float = 0.30
    sd_i: float = 0.12
    seed: int = 1
    genuine: Optional[List[float]] = None
    impostor: Optional[List[float]] = None
    csv: Optional[str] = None


@router.get("/api/labplus/metrics/presets")
def metrics_presets():
    return metrics.PRESETS


@router.post("/api/labplus/metrics/analyze")
def metrics_analyze(body: ScoresIn):
    if body.mode == "synthetic":
        p = dict(metrics.PRESETS.get(body.preset or "", {}))
        g, i = _guard(metrics.generate_scores, body.n_genuine, body.n_impostor,
                      p.get("mu_g", body.mu_g), p.get("sd_g", body.sd_g),
                      p.get("mu_i", body.mu_i), p.get("sd_i", body.sd_i), body.seed)
    elif body.mode == "csv":
        g, i = _guard(metrics.parse_scores_text, body.csv or "")
    else:
        g, i = np.asarray(body.genuine or []), np.asarray(body.impostor or [])
    res = _guard(metrics.analyze, g, i)
    res["source"] = body.mode
    return res


# ───────────────────────── устойчивость изображений ─────────────────────────
@router.get("/api/labplus/robust/options")
def robust_options():
    return {"attacks": {k: {"title": v[0], "unit": v[1], "values": v[2]} for k, v in imaging.ATTACKS.items()},
            "matchers": {k: v[0] for k, v in imaging.MATCHERS.items()}}


@router.post("/api/labplus/robust/image")
def robust_image(kind: str = Form("all"), matcher: str = Form("hog"), threshold: float = Form(0.5),
                 seed: int = Form(1), file: Optional[UploadFile] = File(None)):
    if file is not None and file.filename:
        img = _guard(imaging.decode_image, _read_limited(file, MAX_IMAGE))
        src = "upload"
    else:
        img, src = imaging.synthetic_fingerprint(seed), "synthetic"
    kinds = list(imaging.ATTACKS) if kind == "all" else [kind]
    results = [_guard(imaging.sweep, img, k, matcher, float(threshold), seed, kind != "all") for k in kinds]
    fn = imaging.MATCHERS[matcher][1]
    floor = float(fn(img, imaging.shuffled_reference(img, seed)))
    return {"source": src, "original": imaging.to_b64_png(img, 160), "threshold": threshold, "matcher": matcher,
            "floor": round(floor, 4), "results": results, "n_steps": len(imaging.ATTACKS[kinds[0]][2])}


# ───────────────────────── наставник и стандарты ─────────────────────────
class ExplainIn(BaseModel):
    kind: str
    payload: dict
    level: int = 2


@router.post("/api/labplus/explain")
def explain_endpoint(body: ExplainIn):
    return {"text": explain.generic(body.kind, body.payload, max(1, min(3, body.level)))}


@router.get("/api/labplus/standards")
def standards_endpoint():
    return standards.payload()
