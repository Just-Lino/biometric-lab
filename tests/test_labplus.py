import io
import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from labplus import router as labplus_router
from labplus.core import audio, imaging, intent, metrics, standards

scipy = pytest.importorskip("scipy")
from tests.synth import voice  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LABPLUS_DATA", str(tmp_path))
    app = FastAPI()
    app.include_router(labplus_router)
    return TestClient(app)


def wav(x):
    return audio.write_wav(x)


SPK = {"A": (120, 1.0), "C": (210, 1.18)}


# ───────── страницы и служебное ─────────
def test_health_and_pages(client):
    assert client.get("/api/labplus/health").json()["ok"] is True
    for p in ("voice", "metrics", "robustness", "standards"):
        r = client.get(f"/labplus/{p}")
        assert r.status_code == 200 and "LAB+" in r.text
    assert client.get("/labplus/nope").status_code == 404
    assert client.get("/labplus/static/labplus.js").status_code == 200
    assert client.get("/labplus/static/secret.py").status_code == 404


# ───────── метрики ─────────
def test_probit_matches_scipy():
    from scipy.stats import norm
    p = np.array([1e-4, 0.01, 0.2, 0.5, 0.8, 0.99, 0.9999])
    assert np.allclose(metrics.probit(p), norm.ppf(p), atol=1e-6)


def test_eer_matches_analytic():
    from scipy.stats import norm
    g, i = metrics.generate_scores(20000, 20000, 0.7, 0.1, 0.4, 0.1, seed=3)
    r = metrics.analyze(g, i, ci=False)
    analytic = 1 - norm.cdf((0.55 - 0.4) / 0.1)       # пороги симметричны: t = 0.55
    assert abs(r["eer"] - analytic) < 0.01
    assert abs(r["eer_threshold"] - 0.55) < 0.02
    assert 0.5 < r["auc"] < 1.0
    assert abs(r["dprime"] - 3.0) < 0.15


def test_tradeoff_monotonic():
    g, i = metrics.generate_scores(500, 5000, 0.7, 0.12, 0.38, 0.14)
    r = metrics.analyze(g, i, ci=False)
    assert all(a >= b for a, b in zip(r["fmr"], r["fmr"][1:]))      # FMR падает с порогом
    assert all(a <= b for a, b in zip(r["fnmr"], r["fnmr"][1:]))    # FNMR растёт с порогом


def test_perfect_and_overlapping():
    r = metrics.analyze([0.9] * 10 + [0.95] * 10, [0.1] * 10 + [0.2] * 10, ci=False)
    assert r["eer"] == 0 and r["auc"] == 1.0
    same = metrics.analyze(np.random.default_rng(0).random(300), np.random.default_rng(1).random(300), ci=False)
    assert abs(same["auc"] - 0.5) < 0.08 and abs(same["eer"] - 0.5) < 0.08


def test_rule_of_three_flag():
    g, i = metrics.generate_scores(100, 200, 0.7, 0.1, 0.3, 0.1)
    ops = metrics.analyze(g, i, ci=False)["ops"]
    assert ops[0]["reliable"] is True        # FMR=0.1, 200 чужих ≥ 30
    assert ops[2]["reliable"] is False       # FMR=0.001 нужно ≥ 3000


def test_metrics_errors_and_csv(client):
    with pytest.raises(metrics.MetricsError):
        metrics.analyze([1, 2], [0, 1])
    g, i = metrics.parse_scores_text("genuine,0.9\nсвой;0.8\nimpostor 0.1\nчужой,0.2\n# c\n")
    assert list(g) == [0.9, 0.8] and list(i) == [0.1, 0.2]
    with pytest.raises(metrics.MetricsError):
        metrics.parse_scores_text("who,0.5")
    r = client.post("/api/labplus/metrics/analyze", json={"mode": "csv", "csv": "genuine,0.9\n"})
    assert r.status_code == 400
    r = client.post("/api/labplus/metrics/analyze", json={"mode": "synthetic", "preset": "good", "n_genuine": 200, "n_impostor": 2000})
    assert r.status_code == 200 and r.json()["eer"] < 0.05
    r = client.post("/api/labplus/metrics/analyze", json={"mode": "synthetic", "n_genuine": 1})
    assert r.status_code == 400


# ───────── голосовые команды ─────────
@pytest.mark.parametrize("text,intent_,target", [
    ("Покажи распознавание отпечатков пальцев по датасету FVC2004", "run", "fingerprint"),
    ("запусти распознавание лица", "run", "face"),
    ("покажи кривые ROC", "run", "metrics"),
    ("проверь устойчивость к шуму", "run", "robustness"),
    ("объясни результат", "explain", None),
    ("как работает радужка", "explain", "iris"),
    ("привет", "greet", None), ("помощь", "help", None), ("абракадабра", "unknown", None),
])
def test_intents(text, intent_, target):
    p = intent.parse(text)
    assert p["intent"] == intent_ and p["target"] == target


def test_intent_slots():
    p = intent.parse("покажи отпечатки по датасету FVC2004 для 20 человек")
    assert p["slots"] == {"dataset": "fvc2004", "people": 20}
    assert "fvc2004" in p["reply_run"]


def test_command_endpoint(client):
    r = client.post("/api/labplus/voice/command", data={"text": "покажи отпечатки пальцев", "mode": "hint"}).json()
    assert r["allowed"] and r["parsed"]["target"] == "fingerprint" and "нажмите" in r["reply"]
    r = client.post("/api/labplus/voice/command", data={"text": "покажи отпечатки", "mode": "confirm"}).json()
    assert "Запустить" in r["reply"]
    # проверка говорящего требует записи
    r = client.post("/api/labplus/voice/command", data={"text": "покажи лицо", "name": "x", "require_speaker": "true"}).json()
    assert r["allowed"] is False


# ───────── голос ─────────
def test_audio_basics():
    x = voice(*SPK["A"], dur=3, seed=1)
    a = audio.analyze(x)
    assert len(a["vector"]) == audio.FEATURE_DIM and 90 < a["f0_hz"] < 150
    assert audio.read_wav(wav(x)).shape == x.shape
    with pytest.raises(audio.AudioError):
        audio.analyze(np.zeros(16000, dtype=np.float32))
    with pytest.raises(audio.AudioError):
        audio.read_wav(b"not a wav")


def test_voice_enroll_verify_flow(client):
    files = [("files", (f"{i}.wav", wav(voice(*SPK["A"], dur=3, seed=i)), "audio/wav")) for i in range(4)]
    r = client.post("/api/labplus/voice/enroll", data={"name": "tester"}, files=files)
    assert r.status_code == 200, r.text
    assert r.json()["n_phrases"] == 4 and r.json()["dim"] == audio.FEATURE_DIM
    # в хранилище только числа, никакого аудио
    import os
    stored = open(os.path.join(os.environ["LABPLUS_DATA"], "voice_profiles.json"), encoding="utf-8").read()
    assert "RIFF" not in stored and "mu" in stored
    prof = client.get("/api/labplus/voice/profiles").json()["profiles"][0]
    assert prof["n_audio"] == 4                                        # записи фраз лежат отдельно и их можно прослушать
    wavr = client.get("/api/labplus/voice/profiles/tester/phrase/0.wav")
    assert wavr.status_code == 200 and wavr.content[:4] == b"RIFF" and wavr.headers["content-type"] == "audio/wav"
    assert client.get("/api/labplus/voice/profiles/tester/phrase/9.wav").status_code == 404

    def verify(spk, seed, th=0.5):
        return client.post("/api/labplus/voice/verify", data={"name": "tester", "threshold": th},
                           files={"file": ("v.wav", wav(voice(*SPK[spk], dur=3, seed=seed)), "audio/wav")}).json()
    own = [verify("A", 100 + k)["score"] for k in range(4)]
    other = [verify("C", 200 + k)["score"] for k in range(4)]
    assert np.mean(own) > np.mean(other) + 0.3
    assert verify("A", 101)["decision"] == "MATCH" and verify("C", 201)["decision"] == "NO MATCH"
    v = verify("A", 102)
    assert "viz" in v and len(v["viz"]["waveform"]) == 400 and len(v["viz"]["mel"]) == audio.N_MELS

    rob = client.post("/api/labplus/voice/robustness", data={"name": "tester", "threshold": 0.5},
                      files={"file": ("v.wav", wav(voice(*SPK["A"], dur=3, seed=103)), "audio/wav")}).json()
    assert {x["kind"] for x in rob["results"]} == set(audio.AUDIO_ATTACKS)
    noise = next(x for x in rob["results"] if x["kind"] == "noise")["points"]
    assert noise[0]["score"] >= noise[-1]["score"]                    # сильный шум хуже слабого

    assert client.get("/api/labplus/voice/profiles").json()["profiles"][0]["name"] == "tester"
    assert client.delete("/api/labplus/voice/profiles/tester").status_code == 200
    assert client.get("/api/labplus/voice/profiles/tester/phrase/0.wav").status_code == 404   # записи удалены вместе с профилем
    assert client.delete("/api/labplus/voice/profiles/tester").status_code == 404
    assert verify("A", 1).get("detail")                                # профиля больше нет


def test_voice_errors(client):
    files = [("files", (f"{i}.wav", wav(voice(*SPK["A"], dur=3, seed=i)), "audio/wav")) for i in range(2)]
    assert client.post("/api/labplus/voice/enroll", data={"name": "t"}, files=files).status_code == 400   # < 3 фраз
    bad = [("files", ("x.wav", b"junk", "audio/wav"))] * 3
    assert client.post("/api/labplus/voice/enroll", data={"name": "t"}, files=bad).status_code == 400
    ok = [("files", (f"{i}.wav", wav(voice(*SPK["A"], dur=3, seed=i)), "audio/wav")) for i in range(3)]
    assert client.post("/api/labplus/voice/enroll", data={"name": "../../etc"}, files=ok).status_code == 400  # имя профиля


def test_command_with_speaker_check(client):
    files = [("files", (f"{i}.wav", wav(voice(*SPK["A"], dur=3, seed=i)), "audio/wav")) for i in range(4)]
    client.post("/api/labplus/voice/enroll", data={"name": "me"}, files=files)

    def cmd(spk, seed):
        return client.post("/api/labplus/voice/command", data={"text": "покажи отпечатки", "name": "me", "require_speaker": "true", "threshold": 0.5},
                           files={"file": ("c.wav", wav(voice(*SPK[spk], dur=3, seed=seed)), "audio/wav")}).json()
    assert cmd("A", 300)["allowed"] is True
    stranger = cmd("C", 301)
    assert stranger["allowed"] is False and "не узнан" in stranger["reply"]


# ───────── устойчивость изображений ─────────
def test_imaging_sweeps():
    img = imaging.synthetic_fingerprint(1)
    assert img.shape == (192, 192) and not np.array_equal(img, imaging.synthetic_fingerprint(2))
    for key in imaging.MATCHERS:
        fn = imaging.MATCHERS[key][1]
        assert fn(img, img) > 0.99
        assert fn(img, imaging.synthetic_fingerprint(2)) < 0.5
    r = imaging.sweep(img, "blur", "hog", 0.5, thumbs=False)
    sc = [p["score"] for p in r["points"]]
    assert sc[0] > 0.99 and sc[-1] < sc[0] and r["breakpoint"] is not None
    for k in imaging.ATTACKS:                                          # все искажения работают и сохраняют размер
        assert imaging.apply_attack(img, k, imaging.ATTACKS[k][2][-1]).shape == img.shape


def test_register_custom_matcher():
    imaging.register_matcher("const", "Тестовый", lambda a, b: 0.7)
    r = imaging.sweep(imaging.synthetic_fingerprint(1), "noise", "const", 0.5, thumbs=False)
    assert all(p["score"] == 0.7 for p in r["points"]) and r["breakpoint"] is None
    del imaging.MATCHERS["const"]


def test_robust_endpoint(client):
    r = client.post("/api/labplus/robust/image", data={"kind": "all", "matcher": "ncc", "threshold": 0.5}).json()
    assert r["source"] == "synthetic" and len(r["results"]) == len(imaging.ATTACKS)
    rot = next(x for x in r["results"] if x["kind"] == "rotate")
    assert rot["breakpoint"] == 5                                      # NCC ломается уже на 5°
    r = client.post("/api/labplus/robust/image", data={"kind": "noise", "matcher": "hog"}).json()
    assert r["results"][0]["points"][0]["thumb"].startswith("data:image/png")
    import cv2
    ok, buf = cv2.imencode(".png", imaging.synthetic_fingerprint(5))
    r = client.post("/api/labplus/robust/image", data={"kind": "blur"}, files={"file": ("a.png", buf.tobytes(), "image/png")}).json()
    assert r["source"] == "upload"
    assert client.post("/api/labplus/robust/image", data={"kind": "blur"}, files={"file": ("a.png", b"junk", "image/png")}).status_code == 400
    assert client.post("/api/labplus/robust/image", data={"kind": "zzz"}).status_code == 400
    assert client.post("/api/labplus/robust/image", data={"matcher": "zzz"}).status_code == 400


# ───────── наставник и стандарты ─────────
def test_explain_and_standards(client):
    for kind, payload in [("voice", {"score": 0.3, "threshold": 0.5, "quality": {"speech_sec": 1.0}}),
                          ("voice", {"score": 0.9, "threshold": 0.5}),
                          ("metrics", {"eer": 0.05, "auc": 0.98, "dprime": 3, "ops": [{"reliable": False, "target_fmr": 0.001, "need_impostors": 3000}]}),
                          ("robust", {"threshold": 0.5, "results": [{"title": "Шум", "unit": "σ", "breakpoint": 35}]})]:
        texts = [client.post("/api/labplus/explain", json={"kind": kind, "payload": payload, "level": lv}).json()["text"] for lv in (1, 2, 3)]
        assert len(texts[0]) < len(texts[1]) < len(texts[2])
    assert "запишите голос" in client.post("/api/labplus/explain", json={"kind": "voice", "payload": {}}).json()["text"].lower()
    d = client.get("/api/labplus/standards").json()
    codes = {x["code"] for x in d["catalog"]}
    assert {"ISO/IEC 19795-1:2021", "ISO/IEC 30107-3:2023", "ISO/IEC 24745:2022"} <= codes
    assert all(x["verified"] in ("web", "project", "check") for x in d["catalog"])
    assert "не сертифицирована" in d["disclaimer"]
