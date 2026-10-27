import itertools

import numpy as np
import pytest

from app.modules.fingerprint import synthetic
from app.modules.fingerprint.matchers import DEFAULT_THRESHOLDS, MATCHERS, compare
from app.modules.fingerprint.module import FingerprintModule
from app.modules.fingerprint.params import FingerprintParams
from app.modules.fingerprint.pipeline import run_pipeline

P = FingerprintParams()


@pytest.fixture(scope="module")
def runs():
    return {(f, i): run_pipeline(synthetic.impression(f, i).copy(), P, False) for f in range(1, 7) for i in (1, 2, 3)}


CLASSIC = [m for m in MATCHERS if m != "cnn"]  # нейросеть проверяется отдельно: ей нужно обучение


@pytest.mark.parametrize("name", CLASSIC)
def test_matchers_separate_fingers(runs, name):
    g = [compare(name, runs[(f, a)], runs[(f, b)], P).score for f in range(1, 7) for a, b in itertools.combinations((1, 2, 3), 2)]
    im = [compare(name, runs[(f, i)], runs[(h, i)], P).score for f, h in itertools.combinations(range(1, 7), 2) for i in (1, 2, 3)]
    assert 0 <= min(g + im) and max(g + im) <= 1
    assert np.mean(g) > 2 * np.mean(im)  # свои заметно выше чужих
    if name != "sift":  # SIFT - неспециализированный метод, на отпечатках он заметно слабее
        assert min(g) > DEFAULT_THRESHOLDS[name] > max(im)


def test_compare_with_every_matcher_and_overlay():
    mod = FingerprintModule()
    for name in CLASSIC:
        r = mod.compare("f01_i1", "f01_i2", {"matcher": name})
        assert r["similarity"] > DEFAULT_THRESHOLDS[name] * 0.9 and r["overlay"].startswith("data:image/png")
        assert r["details"][0] == ["Метод сопоставления", MATCHERS[name]] and r["context"]["matcher"] == name


def test_default_matcher_is_unchanged():
    mod = FingerprintModule()
    assert mod.compare("f01_i1", "f01_i2", {})["similarity"] == mod.compare("f01_i1", "f01_i2", {"matcher": "hough"})["similarity"]


def test_method_comparison_cards():
    mod = FingerprintModule()
    d = mod.compare_detectors("f03_i2", {})
    assert [x["name"] for x in d["rows"]] == ["adaptive", "otsu", "fixed"] and all(x["time_ms"] > 0 for x in d["rows"])
    assert d["overlay"].startswith("data:image/png")
    e = mod.compare_embedders("f01_i1", "f01_i3", {})
    assert [x["name"] for x in e["rows"]] == list(MATCHERS) and e["rows"][0]["decision"] == "MATCH"
    assert not e["rows"][-1]["available"] and "не обучена" in e["rows"][-1]["note"]
    ev = {x["name"]: x for x in mod.evaluate_embedders({}, 6)["rows"]}
    assert ev["hough"]["eer"] <= 0.03 and ev["poc"]["eer"] <= 0.03 and ev["sift"]["eer"] > ev["hough"]["eer"]
