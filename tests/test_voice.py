"""Голос как полноценная биометрия: этапы, сравнение, FAR/FRR/EER, сравнение методов, загрузка WAV, звук образцов и голосовые команды."""
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.modules.voice import audio, features, synthetic
from app.modules.voice.module import VoiceModule
from app.services.assistant import parse
from app.services.tutor import RuleBasedTutor


@pytest.fixture(scope="module")
def mod():
    return VoiceModule()


def test_samples_and_preview(mod):
    s = mod.list_samples()
    assert len(s) == 36 and s[0]["id"] == "v01_r1" and s[0]["person"] == "v01"
    assert [x["id"] for x in mod.list_samples(q="v05_r2")] == ["v05_r2"]
    png = mod.get_sample_preview("v01_r1", {})
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(KeyError):
        mod.get_sample_preview("zzz", {})
    with pytest.raises(KeyError):
        mod.list_samples("nope")


def test_audio_pipeline_finds_speech_and_pitch():
    for k in (1, 5, 9):
        a = audio.analyze(synthetic.recording(k, 1))
        assert 1.2 < a["speech_sec"] < 2.0  # три слова из 2,6 с записи
        f = a["f0"][a["mask"] & (a["f0"] > 0)]
        assert abs(np.median(f) / synthetic.speaker(k)["f0"] - 1) < 0.12  # найденная высота тона близка к заданной
    with pytest.raises(ValueError, match="пустая"):
        audio.analyze(np.zeros(16000, np.float32))
    with pytest.raises(ValueError, match="мало речи"):
        audio.analyze((np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32))


def test_process_has_all_steps(mod):
    r = mod.process("v02_r1", {})
    assert r["found"] and [s["id"] for s in r["steps"]] == ["original", "vad", "spectrum", "mel", "mfcc", "pitch", "template"]
    assert all(s["image"].startswith("data:image/png") and s["description"] == s["desc"] for s in r["steps"])
    assert any(f[0] == "Высота тона f0" for f in r["facts"])


def test_same_speaker_scores_higher_and_all_methods_work(mod):
    for method in features.METHODS:
        same = mod.compare("v02_r1", "v02_r2", {"method": method})["similarity"]
        diff = mod.compare("v02_r1", "v09_r2", {"method": method})["similarity"]
        assert same > diff, method
    c = mod.compare("v03_r1", "v03_r3", {})
    assert c["overlay"].startswith("data:image/png") and c["context"]["modality"] == "voice" and c["context"]["f0_a"]


def test_evaluate_and_method_tables(mod):
    ev = mod.evaluate({}, 12)
    assert ev["n_genuine"] == 36 and ev["n_impostor"] == 132 and ev["eer"] <= 0.12 and ev["real"] is False
    rows = {r["name"]: r for r in mod.evaluate_embedders({}, 12)["rows"]}
    assert set(rows) == set(features.METHODS) and rows["mfcc"]["eer"] < rows["pitch"]["eer"]  # тембр различает дикторов лучше одной высоты тона
    cmp_rows = mod.compare_embedders("v02_r1", "v02_r2", {})["rows"]
    assert [r["name"] for r in cmp_rows] == list(features.METHODS) and all(r["decision"] == "MATCH" for r in cmp_rows if r["name"] != "mfcc_delta")
    vad = mod.compare_detectors("v02_r1", {})
    assert [r["name"] for r in vad["rows"]] == ["adaptive", "fixed", "zcr"] and vad["overlay"].startswith("data:image/png")


def test_noise_param_changes_the_sample(mod):
    clean = mod.get_sample_audio("v01_r1", {})
    noisy = mod.get_sample_audio("v01_r1", {"extra_noise": 50})
    assert clean != noisy and clean[:4] == b"RIFF"
    assert mod.get_sample_audio("v01_r1", {"extra_noise": 50}) == noisy  # шум воспроизводим


def test_upload_wav_rules(mod):
    wav = mod.get_sample_audio("v04_r2", {})
    uid = mod.upload(wav)
    assert uid in {s["id"] for s in mod.list_samples()} and mod.compare(uid, "v04_r1", {})["similarity"] > mod.compare(uid, "v10_r1", {})["similarity"]
    with pytest.raises(ValueError, match="WAV"):
        mod.upload(b"not a wav")
    short = io.BytesIO()
    with wave.open(short, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(b"\x00\x00" * 4000)
    with pytest.raises(ValueError, match="короткая"):
        mod.upload(short.getvalue())
    stereo = io.BytesIO()  # стерео 44,1 кГц: приводится к моно 16 кГц
    t = np.arange(44100 * 2) / 44100
    pcm = (np.sin(2 * np.pi * 220 * t) * 12000).astype("<i2")
    with wave.open(stereo, "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(44100), w.writeframes(np.repeat(pcm, 2).tobytes())
    assert abs(len(audio.read_wav(stereo.getvalue())) / 16000 - 2.0) < 0.01


def test_api_flow_audio_route_and_upload_kind():
    with TestClient(app) as c:
        assert "voice" in {m["name"] for m in c.get("/api/v1/biometrics").json()}
        r = c.get("/api/v1/biometrics/voice/sample/v01_r1.wav")
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
        assert c.get("/api/v1/biometrics/fingerprint/sample/f01_i1.wav").status_code == 404  # у отпечатков звука нет
        assert c.get("/api/v1/biometrics/voice/sample/v01_r1.png").status_code == 200
        up = c.post("/api/v1/biometrics/voice/upload", files={"file": ("mine.wav", r.content, "audio/wav")})
        assert up.status_code == 200 and up.json()["sample_id"].startswith("up_")  # WAV не проходит проверку «это картинка»
        assert c.post("/api/v1/biometrics/voice/upload", files={"file": ("x.png", b"\x89PNG not audio", "image/png")}).status_code == 400
        cmp = c.post("/api/v1/biometrics/voice/compare", json={"sample_a": "v01_r1", "sample_b": "v01_r2", "threshold": 0.5}).json()
        assert cmp["decision"] == "MATCH" and cmp["similarity"] > 0.5
        assert c.post("/api/v1/biometrics/voice/process", json={"sample_id": "v01_r1", "params": {"method": "nope"}}).status_code == 422
        assert c.post("/api/v1/biometrics/voice/evaluate", json={"params": {}, "n_subjects": 5}).status_code == 200


def test_voice_commands_and_tutor():
    ctx = {"mod": "fingerprint"}
    assert [a["do"] for a in parse("открой голос и сравни записи", ctx)["actions"]] == ["open", "compare"]
    assert parse("открой голос и сравни записи", ctx)["actions"][0] == {"do": "open", "mod": "voice"}
    r = parse("выбери диктора пять запись два как образец б", {"mod": "voice"})["actions"]
    assert r == [{"do": "sample", "slot": "b", "kind": "диктор", "n": 5, "m": 2}]
    assert parse("запиши мой голос в образец а", {"mod": "voice"})["actions"] == [{"do": "record", "slot": "a"}]
    assert parse("прослушай образец б", {"mod": "voice"})["actions"] == [{"do": "play", "slot": "b"}]
    assert {"do": "method", "value": "pitch"} in parse("сравни голоса по высоте тона", {"mod": "voice"})["actions"]
    assert parse("сравни методом голосования", {"mod": "fingerprint"})["actions"][0] == {"do": "method", "value": "hough"}  # «голосование» - не вкладка «Голос»
    t = RuleBasedTutor().ask("diagnose", "", {"modality": "voice", "similarity": 0.2, "threshold": 0.5, "f0_a": 110, "f0_b": 190, "speech_a": 0.9, "speech_b": 1.6,
                                              "extra_noise": 30, "method": "pitch"})
    assert "NO MATCH" in t and "110" in t and "мало" in t and "шум" in t and "высоты" in t.lower() or "высоте тона" in t
    assert "MFCC" in RuleBasedTutor().ask("explain", "что такое MFCC", {})
