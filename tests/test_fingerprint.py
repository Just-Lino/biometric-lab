import numpy as np
import pytest

from app.modules.fingerprint import synthetic
from app.modules.fingerprint.matching import match_minutiae
from app.modules.fingerprint.module import FingerprintModule
from app.modules.fingerprint.params import FingerprintParams
from app.modules.fingerprint.pipeline import run_pipeline


@pytest.fixture(scope="module")
def mod():
    return FingerprintModule()


def test_synthetic_is_deterministic():
    a, b = synthetic.impression(1, 1), synthetic.impression(1, 1)
    assert a.shape == (256, 256) and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_pipeline_steps_and_minutiae():
    res = run_pipeline(synthetic.impression(1, 1), FingerprintParams())
    assert [s["id"] for s in res.steps] == ["original", "segmentation", "normalized", "orientation",
                                            "enhanced", "binary", "skeleton", "minutiae"]
    assert 15 <= len(res.minutiae) <= 90
    assert {m.type for m in res.minutiae} == {"ending", "bifurcation"}


def test_genuine_scores_higher_than_impostor(mod):
    p = FingerprintParams()
    m = lambda s: mod._run(s, p, False).minutiae
    genuine = match_minutiae(m("f01_i1"), m("f01_i2")).score
    impostor = match_minutiae(m("f01_i1"), m("f02_i1")).score
    assert genuine > 0.6 > 0.4 > impostor


def test_compare_response(mod):
    r = mod.compare("f03_i1", "f03_i2", {})
    assert r["similarity"] > 0.5 and r["overlay"].startswith("data:image/png")


def test_noise_reduces_similarity(mod):
    clean = mod.compare("f04_i1", "f04_i2", {})["similarity"]
    noisy = mod.compare("f04_i1", "f04_i2", {"extra_noise": 50})["similarity"]
    assert noisy < clean


def test_unknown_sample(mod):
    with pytest.raises(KeyError):
        mod.process("nope", {})
