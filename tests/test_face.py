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
