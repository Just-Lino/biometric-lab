import cv2
import numpy as np
import pytest
import skimage.data as skd

import app.modules.face.module as fm
from app.modules.face.align import SIZE, TEMPLATE, align_face, similarity_transform


@pytest.fixture()
def mod(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "GALLERY", tmp_path)
    img = cv2.cvtColor(skd.astronaut(), cv2.COLOR_RGB2BGR)
    d = tmp_path / "astro"
    d.mkdir()
    cv2.imwrite(str(d / "a.png"), img)
    cv2.imwrite(str(d / "b.png"), cv2.convertScaleAbs(img, alpha=0.8, beta=20))
    return fm.FaceModule()


def test_similarity_transform_recovers_known_motion():
    th, s, t = np.radians(17), 1.4, np.array([5.0, -3.0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    src = TEMPLATE
    dst = (s * (R @ src.T)).T + t
    M = similarity_transform(src, dst)
    assert np.allclose(M[:, :2], s * R, atol=1e-3) and np.allclose(M[:, 2], t, atol=1e-2)


def test_align_output_size():
    img = np.random.RandomState(0).randint(0, 255, (300, 300, 3), np.uint8)
    pts = TEMPLATE * 2 + 40
    assert align_face(img, pts).shape == (SIZE, SIZE, 3)


def test_process_has_all_steps(mod):
    r = mod.process("astro__a", {})
    assert r["found"] and [s["id"] for s in r["steps"]] == [
        "original", "detection", "landmarks", "aligned", "normalized", "embedding"]
    assert any(f[0] == "Размерность embedding" for f in r["facts"])


def test_compare_same_person_and_threshold_fields(mod):
    r = mod.compare("astro__a", "astro__b", {})
    assert r["similarity"] > 0.8 and r["overlay"].startswith("data:image/png")
    assert r["context"]["modality"] == "face"


def test_no_face_raises(mod):
    noise = np.random.RandomState(1).randint(0, 255, (200, 200, 3), np.uint8)
    ok, buf = cv2.imencode(".png", noise)
    uid = mod.upload(buf.tobytes())
    with pytest.raises(ValueError, match="лицо не найдено"):
        mod.compare("astro__a", uid, {})


def test_evaluate_needs_two_people(mod):
    with pytest.raises(ValueError, match="минимум 2 человека"):
        mod.evaluate({}, 5)


def test_unknown_sample(mod):
    with pytest.raises(KeyError):
        mod.process("nobody__x", {})


# ---------- несколько детекторов и признаков ----------
from app.modules.face import models as _models  # noqa: E402
from app.modules.face.embed import CLASSICAL, EMBEDDERS, embed  # noqa: E402

_DIMS = {"lbp": 640, "hog": 1764, "gabor": 576, "dct": 63, "pixels": 576}
_need_arcface = pytest.mark.skipif(_models.model_path("arcface") is None, reason="модель ArcFace не скачана")
_need_ssd = pytest.mark.skipif(_models.model_path("ssd_model") is None or _models.model_path("ssd_proto") is None,
                               reason="модель SSD не скачана")


def test_classical_embedders_shape_and_determinism():
    aligned = cv2.resize(cv2.cvtColor(skd.astronaut(), cv2.COLOR_RGB2BGR)[40:200, 120:300], (112, 112))
    for name, dim in _DIMS.items():
        a, b = embed(aligned, name), embed(aligned, name)
        assert a.shape == (dim,) and np.isfinite(a).all() and np.array_equal(a, b)
        assert abs(float(a @ a / (np.linalg.norm(a) ** 2)) - 1) < 1e-6


def test_compare_detectors_rows(mod):
    r = mod.compare_detectors("astro__a", {})
    names = [x["name"] for x in r["rows"]]
    assert names == ["haar", "haar_alt2", "haar_alt", "haar_profile", "ssd", "ultraface", "yunet"]
    found = {x["name"]: x.get("found") for x in r["rows"]}
    assert found["haar"] and found["haar_alt2"] and found["haar_alt"]  # фронтальные каскады находят лицо
    assert r["overlay"].startswith("data:image/png")
    assert all("time_ms" in x for x in r["rows"] if x["available"])


def test_compare_embedders_rows(mod):
    r = mod.compare_embedders("astro__a", "astro__b", {})
    assert [x["name"] for x in r["rows"]] == list(EMBEDDERS)
    for x in r["rows"]:
        if x["available"]:
            assert x["similarity"] > 0.8 and x["dim"] > 0  # одно и то же лицо с другой яркостью
        else:
            assert x["note"]


def test_evaluate_embedders_needs_gallery(mod):
    with pytest.raises(ValueError, match="минимум 2 человека"):
        mod.evaluate_embedders({}, 5)


def test_unknown_detector_and_missing_model_message(mod, monkeypatch):
    monkeypatch.setattr("app.modules.face.detect.detector_available", lambda n: False if n in ("ssd", "yunet") else True)
    with pytest.raises(ValueError, match="не найдена"):
        mod.process("astro__a", {"detector": "ssd"})


@_need_ssd
def test_ssd_detector_and_pipeline(mod):
    r = mod.process("astro__a", {"detector": "ssd"})
    assert r["found"] and any(f[1].startswith("SSD") for f in r["facts"])


@_need_arcface
def test_arcface_same_face_high(mod):
    r = mod.compare("astro__a", "astro__b", {"embedder": "arcface"})
    assert r["similarity"] > 0.9


_need_ultraface = pytest.mark.skipif(_models.model_path("ultraface") is None, reason="модель UltraFace не скачана")


@_need_ultraface
def test_ultraface_detector(mod):
    r = mod.process("astro__a", {"detector": "ultraface"})
    assert r["found"] and any(f[1].startswith("UltraFace") for f in r["facts"])


def test_eigenfaces_basis_and_small_gallery(mod):
    from app.modules.face.embed import eigen_basis, eigen_embedding
    base = cv2.resize(cv2.cvtColor(skd.astronaut(), cv2.COLOR_RGB2BGR)[40:200, 120:300], (112, 112))
    faces = [cv2.convertScaleAbs(base, alpha=0.7 + 0.1 * k, beta=5 * k) for k in range(8)]
    faces = [np.roll(f, k, axis=1) for k, f in enumerate(faces)]
    mean, comps = eigen_basis(faces)
    assert comps.shape == (7, 1024) and mean.shape == (1024,)
    e = eigen_embedding(faces[0], (mean, comps))
    assert e.shape == (7,) and np.isfinite(e).all()
    with pytest.raises(ValueError, match="минимум из 6"):
        eigen_basis(faces[:3])
    rows = {x["name"]: x for x in mod.compare_embedders("astro__a", "astro__b", {})["rows"]}
    assert not rows["eigenfaces"]["available"] and "6 фото" in rows["eigenfaces"]["note"]  # в тестовой галерее всего 2 фото
