"""Реальные наборы (FVC, SOCOFing, CASIA), несколько источников лиц, раздельные загрузки и набор данных у обученной сети.

Настоящие наборы в тестах не нужны: фикстура собирает в временной папке маленькие копии из синтетики лаборатории
в тех же раскладках, что и у реальных наборов."""
import cv2
import numpy as np
import pytest
import skimage.data as skd
from fastapi.testclient import TestClient

import app.modules.face.module as fm
from app.core import realdata
from app.main import app
from app.modules.fingerprint import real as fp_real
from app.modules.fingerprint import synthetic as fp_syn
from app.modules.fingerprint.module import FingerprintModule
from app.modules.iris import real as iris_real
from app.modules.iris import synthetic as iris_syn
from app.modules.iris.module import IrisModule
from app.training import registry
from app.training.cnn import CNN
from app.training.datasets import available_datasets, load_dataset


def _write(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)


@pytest.fixture()
def real_dir(tmp_path, monkeypatch):
    """datasets/real из маленьких копий: FVC2004 DB1 (3 пальца × 3 оттиска), SOCOFing (3 человека), CASIA (3 глаза × 3 снимка)."""
    root = tmp_path / "real"
    monkeypatch.setattr(realdata, "REAL_DIR", root)
    for f in (1, 2, 3):
        for i in (1, 2, 3):
            img = cv2.copyMakeBorder(fp_syn.impression(f, i), 30, 30, 20, 20, cv2.BORDER_CONSTANT, value=240)  # 316×296, как кадр FVC
            _write(root / "fvc2004" / "DB1" / f"{100 + f}_{i}.tif", img)
    for p, pid in enumerate(("100__M", "101__F", "102__M")):
        base = cv2.resize(fp_syn.impression(p + 1, 1)[40:200, 50:200], (96, 103), interpolation=cv2.INTER_AREA)
        _write(root / "socofing" / "Real" / pid / f"{pid}_Right_index_finger.BMP", base)
        _write(root / "socofing" / "Real" / pid / f"{pid}_Left_thumb_finger.BMP", base)
        for k, level in enumerate(("Easy", "Medium", "Hard")):
            for kind in ("Obl", "CR", "Zcut"):
                alt = cv2.GaussianBlur(base, (0, 0), 0.6 * (k + 1))
                _write(root / "socofing" / "Altered" / f"Altered-{level}" / pid / f"{pid}_Right_index_finger_{kind}.BMP", alt)
    for e in (1, 2, 3):
        for i in (1, 2, 3):
            _write(root / "casia_iris_interval" / f"{e:03d}" / "L" / f"S1{e:03d}L{i:02d}.jpg", iris_syn.impression(e, i)[0])
    fp_real.clear_cache()
    iris_real.clear_cache()
    yield root
    fp_real.clear_cache()
    iris_real.clear_cache()


# ---------- отпечатки ----------
def test_fvc_listing_loading_and_bad_ids(real_dir):
    m = FingerprintModule()
    ids = {s["id"] for s in m.list_samples("fvc2004_db1")}
    assert "fvc2004:DB1:101_1" in ids and len(ids) == 9
    assert {s["id"] for s in m.info()["sources"]} >= {"synthetic", "fvc2004_db1", "socofing"}
    img = fp_real.load("fvc2004:DB1:102_2")
    assert img.shape == (256, 256) and img.dtype == np.uint8
    assert [s["id"] for s in m.list_samples("fvc2004_db1", q="103_2")] == ["fvc2004:DB1:103_2"]
    with pytest.raises(KeyError):
        m.list_samples("fvc2004_db3")  # такой папки нет
    for bad in ("fvc2004:DB1:../../x", "fvc2004:DB9:101_1", "soco:..:Right_index", "fvc2004:DB1:999_9"):
        with pytest.raises(KeyError):
            fp_real.load(bad)


def test_fvc_evaluate_uses_all_independent_impressions(real_dir):
    ev = FingerprintModule().evaluate({"source": "fvc2004_db1"}, 10)
    assert ev["n_genuine"] == 3 * 3 and ev["n_impostor"] == 3 * 3  # 3 пальца: по 3 пары оттисков; 3 пары пальцев × 3 оттиска
    assert ev["real"] is True and "FVC2004" in ev["source_title"]
    assert FingerprintModule().evaluate({}, 4)["real"] is False  # по умолчанию синтетика, как раньше


def test_socofing_labels_no_eer_but_stress_test(real_dir):
    m = FingerprintModule()
    items = m.list_samples("socofing")
    ids = {s["id"] for s in items}
    assert "soco:101__F:Right_index" in ids and "soco:100__M:Left_thumb" in ids
    assert next(s for s in items if s["id"] == "soco:100__M:Right_index")["label"].endswith("правый указательный")
    assert fp_real.load("soco:100__M:Right_index:Hard:CR").shape == (256, 256)
    alt = next(s for s in items if s["id"] == "soco:100__M:Right_index:Hard:CR")  # искажённые копии выбираются в списке рядом с оригиналом
    assert alt["person"] == "soco:100__M:Right_index" and "искажение сильное, центральный поворот" in alt["label"]
    pos = [s["id"] for s in items]
    assert pos.index("soco:100__M:Right_index:Easy:Obl") == pos.index("soco:100__M:Right_index") + 1
    sim = m.compare("soco:100__M:Right_index", "soco:100__M:Right_index:Easy:Obl", {"source": "socofing"})["similarity"]
    assert sim > m.compare("soco:100__M:Right_index", "soco:101__F:Right_index", {"source": "socofing"})["similarity"]
    with pytest.raises(ValueError, match="по одному снимку"):
        m.evaluate({"source": "socofing"}, 10)
    r = m.stress({}, 12)
    assert [x["level"] for x in r["rows"]] == ["Easy", "Medium", "Hard"] and r["n_people"] == 3
    assert all(x["mean"] is not None and x["n"] == 9 for x in r["rows"]) and "искажённой копией" in r["note"]


# ---------- радужка ----------
def test_casia_listing_evaluate_and_no_truth(real_dir):
    m = IrisModule()
    items = m.list_samples("casia_interval")
    assert len(items) == 9 and items[0]["id"] == "casia:S1001L01" and items[0]["person"] == "S1001L"
    assert items[0]["label"] == "Человек 001, левый глаз, снимок 1"
    assert iris_real.load("casia:S1002L03").shape == (240, 320)
    with pytest.raises(KeyError):
        iris_real.load("casia:S9999L01")
    ev = m.evaluate({"source": "casia_interval"}, 12)
    assert ev["n_genuine"] == 9 and ev["n_impostor"] == 6 and ev["real"] is True
    rows = m.compare_detectors("casia:S1001L01", {"source": "casia_interval"})["rows"]
    assert rows and all("error" not in r for r in rows)  # истинных окружностей у реальных снимков нет
    assert "ошибка" not in " ".join(r[0] for r in m.process("casia:S1001L01", {})["facts"]).lower()


# ---------- лицо ----------
@pytest.fixture()
def nested_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "GALLERY", tmp_path)
    img = cv2.cvtColor(skd.astronaut(), cv2.COLOR_RGB2BGR)
    for src, person in (("LFW", "astro"), ("LFW", "astro2"), ("LFW", "solo"), ("", "flat_person")):
        d = tmp_path / src / person if src else tmp_path / person
        n = 1 if person == "solo" else 2
        for k in range(n):
            _write(d / f"{person}_{k}.png", cv2.convertScaleAbs(img, alpha=1 - 0.05 * k, beta=5 * k))
    return fm.FaceModule()


def test_face_gallery_with_sources(nested_gallery):
    m = nested_gallery
    srcs = {s["id"] for s in m.info()["sources"]}
    assert srcs == {"all", "LFW", "gallery"}
    lfw = m.list_samples("LFW")
    assert all(s["id"].startswith("LFW::") for s in lfw) and len(lfw) == 5
    assert lfw[-1]["person"] == "solo"  # сначала люди, у которых есть пара фото
    assert [s["id"] for s in m.list_samples("gallery")] == ["flat_person__flat_person_0", "flat_person__flat_person_1"]
    assert len(m.list_samples("all")) == 7 and [s["id"] for s in m.list_samples("all", q="solo")] == ["LFW::solo__solo_0"]
    assert len(m.list_samples("all", limit=2)) == 2
    assert m.process("LFW::astro__astro_0", {})["found"]
    with pytest.raises(KeyError):
        m.list_samples("nope")
    assert m.evaluate({"source": "LFW", "embedder": "lbp"}, 5)["n_genuine"] >= 2  # люди берутся только из выбранного источника


def test_load_face_does_not_scan_whole_gallery(nested_gallery, monkeypatch):
    calls = []
    orig = fm.FaceModule._run
    monkeypatch.setattr(fm.FaceModule, "_run", lambda self, sid, p, with_images=True: calls.append(sid) or orig(self, sid, p, with_images))
    d = load_dataset("face", 2, 3, "LFW")
    assert len(d.names) == 2 and d.dataset == "LFW"
    assert len(calls) <= 4  # два человека по 2 фото, а не вся галерея


# ---------- обучение и сеть с набором данных ----------
def _model(dataset):
    net = CNN(16, 2, 4, filters=(2, 3, 3), seed=1)
    return registry.TrainedModel(modality="fingerprint", filters=(2, 3, 3), size=16, names=["a", "b"], holdout=3, threshold=0.5, eer=0.1,
                                 test_acc=0.5, epochs=1, config={}, trained_at="x", dataset=dataset, weights=net.state())


def test_trained_net_refuses_other_dataset():
    registry.store(_model("synthetic"))
    assert registry.require("fingerprint", "отпечатков", "synthetic").dataset == "synthetic"
    assert registry.require("fingerprint", "отпечатков").dataset == "synthetic"  # без указания набора проверки нет
    with pytest.raises(ValueError, match="обучена на наборе «synthetic», а выбран «fvc2004_db1»"):
        registry.require("fingerprint", "отпечатков", "fvc2004_db1")
    registry.reset_cache()
    assert registry.get("fingerprint").dataset == "synthetic" and registry.get("fingerprint").info()["dataset"] == "synthetic"  # из файла
    with pytest.raises(ValueError, match="обучена на наборе"):
        FingerprintModule().evaluate({"matcher": "cnn", "source": "fvc2004_db1"}, 4)


def test_training_datasets_from_real_sources(real_dir):
    ds = load_dataset("fingerprint", 3, 3, "fvc2004_db1")
    assert ds.names == ["Палец 101", "Палец 102", "Палец 103"] and len(ds.train) == 6 and len(ds.test) == 3 and ds.dataset == "fvc2004_db1"
    assert load_dataset("fingerprint", 3, 1, "fvc2004_db1").holdout == 1
    with pytest.raises(ValueError, match="SOCOFing"):
        load_dataset("fingerprint", 3, 3, "socofing")
    ir = load_dataset("iris", 3, 3, "casia_interval")
    assert len(ir.names) == 3 and len(ir.train) == 6 and len(ir.test) == 3 and ir.train[0].shape == (96, 96)
    listed = available_datasets()
    assert {d["id"] for d in listed["fingerprint"]} == {"synthetic", "fvc2004_db1"} and {d["id"] for d in listed["iris"]} == {"synthetic", "casia_interval"}
    with TestClient(app) as c:
        assert c.get("/api/v1/training/datasets").json()["default"]["face"] == "all"
        r = c.post("/api/v1/training/start", json={"modality": "iris", "dataset": "fvc2004_db1"})
        assert r.status_code == 400 and "недоступен" in r.json()["detail"]


# ---------- загрузки по сессиям ----------
def test_uploads_are_private_per_session_and_limited():
    png = cv2.imencode(".png", cv2.cvtColor(skd.astronaut(), cv2.COLOR_RGB2BGR))[1].tobytes()
    with TestClient(app) as a, TestClient(app) as b:
        sid = a.post("/api/v1/biometrics/face/upload", files={"file": ("a.png", png, "image/png")}).json()["sample_id"]
        assert "lab_sid" in a.cookies
        assert sid in {s["id"] for s in a.get("/api/v1/biometrics/face/samples?source=all").json()}
        assert sid not in {s["id"] for s in b.get("/api/v1/biometrics/face/samples?source=all").json()}  # другой посетитель не видит
        assert b.get(f"/api/v1/biometrics/face/sample/{sid}.png").status_code == 404
        assert a.get(f"/api/v1/biometrics/face/sample/{sid}.png").status_code == 200
        big = cv2.imencode(".png", np.zeros((4000, 5000), np.uint8))[1].tobytes()  # 20 мегапикселей, но файл маленький
        assert len(big) < 1_000_000
        assert a.post("/api/v1/biometrics/face/upload", files={"file": ("b.png", big, "image/png")}).status_code == 413
        assert a.post("/api/v1/biometrics/face/upload", files={"file": ("c.bin", b"0" * 5_000_001, "image/png")}).status_code == 413
        assert a.post("/api/v1/biometrics/face/upload", files={"file": ("d.txt", b"not an image", "text/plain")}).status_code == 400


def test_upload_store_keeps_newest_per_session():
    from app.modules.uploads import UploadStore
    st = UploadStore(per_session=3, total=10)
    ids = [st.add(np.zeros((2, 2), np.uint8)) for _ in range(5)]
    assert st.ids() == ids[2:] and ids[0] not in st


def test_real_thresholds_and_training_limits(real_dir, monkeypatch):
    monkeypatch.setitem(fp_real.THRESHOLDS, "fvc2004_db1", {"hough": 0.19})
    info = {s["id"]: s for s in FingerprintModule().info()["sources"]}
    assert info["fvc2004_db1"]["thresholds"] == {"hough": 0.19} and info["synthetic"].get("thresholds") is None
    from app.modules.fingerprint import matchers
    assert matchers.recommended_threshold("hough", "fvc2004_db1") == 0.19 and matchers.recommended_threshold("hough") == 0.50
    rows = FingerprintModule().compare_embedders("fvc2004:DB1:101_1", "fvc2004:DB1:101_2", {"source": "fvc2004_db1"})["rows"]
    assert next(r for r in rows if r["name"] == "hough")["threshold"] == 0.19
    mx = {d["id"]: d["max_subjects"] for k in ("fingerprint", "iris") for d in available_datasets()[k]}
    assert mx["synthetic"] == 12 and mx["fvc2004_db1"] == 10 and mx["casia_interval"] == 40
    from pydantic import ValidationError
    from app.api.v1.training import TrainIn
    assert TrainIn(n_subjects=40).n_subjects == 40
    with pytest.raises(ValidationError):
        TrainIn(n_subjects=41)


def test_lru_cache_is_thread_safe():
    import threading
    from app.core.cache import LRUCache
    c, errors = LRUCache(20), []

    def work(k):
        try:
            for i in range(2000):
                c.put(f"{k}-{i % 50}", i)
                c.get(f"{(k + 1) % 4}-{i % 50}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and len(c) <= 20


def test_openai_tutor_without_package_falls_back_to_rules(monkeypatch):
    import sys
    from app.core.config import settings
    from app.services import tutor
    monkeypatch.setitem(sys.modules, "openai", None)  # import openai -> ImportError
    monkeypatch.setattr(settings, "tutor_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(tutor, "_tutor", None)
    assert tutor.get_tutor().name == "rule"
