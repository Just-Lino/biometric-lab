import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def test_api_flow():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/").status_code == 200
        assert c.get("/api/v1/biometrics").json()[0]["name"] == "fingerprint"
        assert len(c.get("/api/v1/biometrics/fingerprint/samples").json()) >= 36

        r = c.post("/api/v1/biometrics/fingerprint/process", json={"sample_id": "f01_i1"})
        assert r.status_code == 200 and len(r.json()["steps"]) == 8

        n0 = len(c.get("/api/v1/experiments?limit=1000").json())  # журнал общий для всей сессии тестов: считаем прирост
        r = c.post("/api/v1/biometrics/fingerprint/compare",
                   json={"sample_a": "f01_i1", "sample_b": "f01_i2", "threshold": 0.5})
        j = r.json()
        assert j["decision"] == "MATCH"
        r = c.post("/api/v1/biometrics/fingerprint/compare",
                   json={"sample_a": "f01_i1", "sample_b": "f05_i1", "threshold": 0.5})
        assert r.json()["decision"] == "NO MATCH"

        r = c.post("/api/v1/biometrics/fingerprint/relief", json={"sample_id": "f01_i1"})
        j = r.json()
        assert r.status_code == 200 and j["n"] == 160 and len(j["minutiae"]) > 10
        assert set(j["textures"]) == {"original", "enhanced", "binary", "skeleton"}

        log = c.get("/api/v1/experiments?limit=1000").json()
        assert len(log) == n0 + 2
        assert "similarity" in c.get("/api/v1/experiments/export.csv").text

        t = c.post("/api/v1/tutor/ask", json={"mode": "diagnose", "context": {
            "similarity": 0.3, "threshold": 0.5, "n_a": 30, "n_b": 30, "matched": 9}}).json()
        assert t["provider"] == "rule" and "NO MATCH" in t["answer"]

        assert c.post("/api/v1/biometrics/fingerprint/process", json={"sample_id": "zzz"}).status_code == 404
        assert c.post("/api/v1/biometrics/fingerprint/process",
                      json={"sample_id": "f01_i1", "params": {"gabor_period": 99}}).status_code == 422
        assert c.post("/api/v1/biometrics/iris/process", json={"sample_id": "x"}).status_code == 404
        assert c.post("/api/v1/biometrics/face/process", json={"sample_id": "x"}).status_code == 404
        mods = {m["name"] for m in c.get("/api/v1/biometrics").json()}
        assert mods == {"fingerprint", "face", "iris", "voice"}
        info = c.get("/api/v1/biometrics/face/info").json()
        assert info["detectors"]["haar"] is True and "persons" in info
        assert set(info["detectors"]) == {"haar", "haar_alt2", "haar_alt", "haar_profile", "ssd", "ultraface", "yunet"}
        assert set(info["embedders"]) == {"sface", "arcface", "lbp", "hog", "gabor", "dct", "pixels", "eigenfaces", "cnn"}
        import cv2, skimage.data as skd
        png = cv2.imencode(".png", cv2.cvtColor(skd.astronaut(), cv2.COLOR_RGB2BGR))[1].tobytes()
        ids = [c.post("/api/v1/biometrics/face/upload", files={"file": ("a.png", png, "image/png")}).json()["sample_id"] for _ in range(2)]
        d = c.post("/api/v1/biometrics/face/detectors", json={"sample_id": ids[0]})
        assert d.status_code == 200 and len(d.json()["rows"]) == 7
        e = c.post("/api/v1/biometrics/face/embedders", json={"sample_a": ids[0], "sample_b": ids[1]})
        assert e.status_code == 200 and all(x["similarity"] > 0.99 for x in e.json()["rows"] if x["available"])
        assert c.post("/api/v1/biometrics/fingerprint/detectors", json={"sample_id": "f01_i1"}).status_code == 200
        assert c.post("/api/v1/biometrics/unknown/detectors", json={"sample_id": "x"}).status_code == 404
