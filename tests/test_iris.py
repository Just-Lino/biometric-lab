import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.modules.iris import synthetic
from app.modules.iris.encode import ENCODERS, encode, match, normalize, occlusion_mask
from app.modules.iris.localize import LOCALIZERS, localize, preprocess
from app.modules.iris.module import IrisModule
from app.modules.iris.params import IrisParams

P = IrisParams()


@pytest.fixture(scope="module")
def mod():
    return IrisModule()


def _err(c, t):
    return (np.hypot(c.pupil[0] - t["pupil"][0], c.pupil[1] - t["pupil"][1]) + abs(c.pupil[2] - t["pupil"][2]),
            np.hypot(c.iris[0] - t["iris"][0], c.iris[1] - t["iris"][1]) + abs(c.iris[2] - t["iris"][2]))


def test_dataset_matches_generator():
    img, t = synthetic.impression(3, 2)
    img2, t2 = synthetic.render(3, 2)
    assert img.shape == (synthetic.H, synthetic.W) and t == t2
    assert np.abs(img.astype(int) - img2.astype(int)).max() <= 1  # PNG без потерь


@pytest.mark.parametrize("method,tol", [("daugman", 4.0), ("contour", 4.0)])
def test_localization_accuracy(method, tol):
    for e in range(1, synthetic.N_EYES + 1):
        for i in (1, 2, 3):
            img, t = synthetic.impression(e, i)
            c = localize(preprocess(img, P), P, method)
            assert c is not None, f"e{e}i{i}"
            ep, ei = _err(c, t)
            assert ep < tol and ei < tol, f"{method} e{e}i{i}: {ep:.1f} / {ei:.1f}"


def test_hough_mostly_works():
    found = sum(localize(preprocess(synthetic.impression(e, i)[0], P), P, "hough") is not None
                for e in range(1, 13) for i in (1, 2, 3))
    assert found >= 30


def test_rubber_sheet_and_mask():
    img, _ = synthetic.impression(4, 3)
    c = localize(preprocess(img, P), P, "daugman")
    strip, valid, (mx, my) = normalize(img, c)
    assert strip.shape == (32, 256) and valid.all()
    assert 0.5 < occlusion_mask(strip, valid, True).mean() <= 1.0
    assert occlusion_mask(strip, valid, False).all()


def test_encoders_separate_eyes():
    def tpl(e, i, enc):
        img = synthetic.impression(e, i)[0]
        c = localize(preprocess(img, P), P, "daugman")
        s, v, _ = normalize(img, c)
        return encode(s, occlusion_mask(s, v, True), enc, P)
    for enc in [e for e in ENCODERS if e != "cnn"]:  # нейросеть проверяется отдельно: ей нужно обучение
        a, b, other = tpl(1, 1, enc), tpl(1, 2, enc), tpl(2, 1, enc)
        g, im = match(a, b, P.max_shift)["score"], match(a, other, P.max_shift)["score"]
        assert g > im, enc
        assert 0.35 < im < 0.75, enc  # у чужих радужек около половины битов не совпадает


def test_process_and_compare(mod):
    r = mod.process("e01_i1", {})
    assert r["found"] and len(r["steps"]) == 8 and any("эталона" in f[0] for f in r["facts"])
    same, other = mod.compare("e01_i1", "e01_i3", {}), mod.compare("e01_i1", "e02_i1", {})
    assert same["similarity"] > 0.75 > other["similarity"]
    assert same["overlay"].startswith("data:image/png") and same["context"]["modality"] == "iris"


def test_evaluate_and_rotation_compensation(mod):
    ev = mod.evaluate({})
    assert ev["eer"] <= 0.03 and ev["n_genuine"] == 36 and ev["n_impostor"] == 132
    assert mod.evaluate({"max_shift": 0})["eer"] > ev["eer"] + 0.1  # без перебора сдвигов поворот глаза ломает сравнение


def test_method_comparisons(mod):
    d = mod.compare_detectors("e05_i3", {})
    assert [x["name"] for x in d["rows"]] == list(LOCALIZERS) and d["columns"][0][0] == "n"
    assert all("error" in x for x in d["rows"] if x["found"])
    e = mod.compare_embedders("e05_i1", "e05_i3", {})
    assert [x["name"] for x in e["rows"]] == list(ENCODERS)
    assert all(x["similarity"] > 0.6 for x in e["rows"] if x["available"]) and not e["rows"][-1]["available"]  # сеть не обучена
    ee = {x["name"]: x for x in mod.evaluate_embedders({}, 6)["rows"]}
    assert ee["gabor"]["eer"] <= ee["haar"]["eer"]


def test_iris_api():
    import cv2
    with TestClient(app) as c:
        s = c.get("/api/v1/biometrics/iris/samples").json()
        assert len(s) == 36 and s[0]["id"] == "e01_i1"
        assert c.post("/api/v1/biometrics/iris/process", json={"sample_id": "e02_i2", "params": {"localizer": "contour", "encoder": "loggabor"}}).json()["found"]
        r = c.post("/api/v1/biometrics/iris/compare", json={"sample_a": "e02_i1", "sample_b": "e02_i2", "params": {}, "threshold": 0.66}).json()
        assert r["similarity"] > 0.66
        assert c.post("/api/v1/biometrics/iris/detectors", json={"sample_id": "e02_i1"}).status_code == 200
        assert c.post("/api/v1/biometrics/iris/embedders", json={"sample_a": "e02_i1", "sample_b": "e03_i1"}).status_code == 200
        png = cv2.imencode(".png", synthetic.impression(7, 1)[0])[1].tobytes()
        up = c.post("/api/v1/biometrics/iris/upload", files={"file": ("eye.png", png, "image/png")}).json()["sample_id"]
        assert c.post("/api/v1/biometrics/iris/process", json={"sample_id": up}).json()["found"]
        assert c.get("/api/v1/biometrics/iris/info").json()["eyes"] == 12
        assert c.post("/api/v1/biometrics/iris/process", json={"sample_id": "e01_i1", "params": {"localizer": "bad"}}).status_code == 422
