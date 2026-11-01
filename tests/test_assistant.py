"""Помощник лаборатории: фраза -> действия для интерфейса (разбор по правилам, без внешних сервисов)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.assistant import EXAMPLES, parse


def acts(text, mod="fingerprint"):
    return [(a["do"], a.get("mod") or a.get("value") or a.get("url")) for a in parse(text, {"mod": mod})["actions"]]


@pytest.mark.parametrize("text, mod, expected", [
    ("Покажи, как изображение превращается в решение", "fingerprint", [("tour", None)]),
    ("от изображения до решения для радужки", "fingerprint", [("open", "iris"), ("tour", None)]),
    ("Открой отпечатки, набор FVC2004 DB3, сравни образцы", "train", [("open", "fingerprint"), ("source", "fvc2004_db3"), ("compare", None)]),
    ("оцени качество на фвц 2002 база 2", "fingerprint", [("source", "fvc2002_db2"), ("evaluate", None)]),
    ("Радужка CASIA, покажи этапы", "fingerprint", [("open", "iris"), ("source", "casia_interval"), ("stages", None)]),
    ("покажи три д конвейер и облёт", "face", [("pipeline3d", None), ("orbit", None)]),
    ("покажи 3D рельеф", "fingerprint", [("relief3d", None)]),
    ("Сравни методы", "iris", [("methods", None)]),
    ("оцени методы по EER", "fingerprint", [("methods_eer", None)]),
    ("сравни методом SIFT порог 0,2", "fingerprint", [("method", "sift"), ("threshold", 0.2), ("compare", None)]),
    ("сравни методом лог габор", "iris", [("method", "loggabor"), ("compare", None)]),
    ("сравни арк фейс", "face", [("method", "arcface"), ("compare", None)]),
    ("стресс-тест SOCOFing", "iris", [("open", "fingerprint"), ("source", "socofing"), ("stress", None)]),
    ("объясни результат", "face", [("explain", None)]),
    ("начни обучение на CASIA", "fingerprint", [("open", "train"), ("train_mod", "iris"), ("train_data", "casia_interval"), ("train_start", None)]),
    ("останови обучение", "train", [("open", "train"), ("train_stop", None)]),
    ("покажи этапы", "train", [("open", "fingerprint"), ("stages", None)]),
    ("покажи кривые ROC", "fingerprint", [("page", "/labplus/metrics")]),
    ("стандарты ISO", "face", [("page", "/labplus/standards")]),
    ("проверь устойчивость к шуму", "fingerprint", [("page", "/labplus/robustness")]),
    ("открой голосовую биометрию", "fingerprint", [("open", "voice")]),
    ("открой страницу голоса LAB+", "fingerprint", [("page", "/labplus/voice")]),
])
def test_phrases_become_actions(text, mod, expected):
    assert acts(text, mod) == expected


def test_unknown_phrase_gives_hints_not_guesses():
    r = parse("абракадабра", {"mod": "fingerprint"})
    assert r["understood"] is False and r["actions"] == [] and r["examples"] == EXAMPLES and "не распознана" in r["reply"]
    assert parse("", {})["understood"] is False


def test_help_lists_capabilities():
    r = parse("что ты умеешь", {})
    assert r["actions"] == [{"do": "help"}] and "экскурсию" in r["reply"] and "3D" in r["reply"]


def test_threshold_is_clamped_and_reply_is_russian():
    r = parse("сравни порог 7", {"mod": "fingerprint"})
    assert {"do": "threshold", "value": 0.95} in r["actions"]
    assert parse("покажи этапы", {"mod": "fingerprint"})["reply"] == "Показываю этапы обработки."


def test_assistant_api_and_limits():
    with TestClient(app) as c:
        r = c.post("/api/v1/assistant/parse", json={"text": "оцени качество на FVC2004", "context": {"mod": "face"}}).json()
        assert r["understood"] and [a["do"] for a in r["actions"]] == ["open", "source", "evaluate"]
        assert c.post("/api/v1/assistant/parse", json={"text": "я" * 501}).status_code == 422
        assert c.get("/api/v1/assistant/examples").json()["examples"] == EXAMPLES
        assert 'id="as-fab"' not in c.get("/").text  # кнопка создаётся скриптом, а не вшита в разметку
        assert "/api/v1/assistant/parse" in c.get("/").text and "labplus.js" not in c.get("/").text


# ---------- улучшенное голосовое управление ----------
@pytest.mark.parametrize("text, mod, expected", [
    ("порог ноль пять", "fingerprint", [("threshold", 0.5)]),
    ("сравни порог ноль целых две десятых", "fingerprint", [("threshold", 0.2), ("compare", None)]),
    ("порог пятьдесят процентов", "fingerprint", [("threshold", 0.5)]),
    ("увеличь порог", "fingerprint", [("threshold", None)]),
    ("повтори", "fingerprint", [("repeat", None)]),
    ("стоп", "fingerprint", [("cancel", None)]),
    ("следующий этап", "fingerprint", [("step", None)]),
    ("перейди на этап четыре", "fingerprint", [("step", None)]),
    ("первый этап", "fingerprint", [("step", None)]),
    ("покажи этапы", "nn", [("open", "fingerprint"), ("stages", None)]),
    ("открой практикум", "fingerprint", [("open", "code")]),
    ("задание бинаризация", "code", [("open", "code"), ("task", "binarize")]),
    ("задание три", "code", [("open", "code"), ("task", "crossing_number")]),
    ("запусти код на примере", "code", [("open", "code"), ("code_run", None)]),
    ("дай подсказку два", "code", [("open", "code"), ("hint", None)]),
    ("проверь решение", "code", [("code_submit", None)]),
    ("объясни результат", "code", [("code_explain", None)]),
    ("запусти обучение", "nn", [("nn_play", None)]),
    ("открой площадку нейросети и запусти обучение", "fingerprint", [("open", "nn"), ("nn_play", None)]),
    ("начни обучение на CASIA", "nn", [("open", "train"), ("train_mod", "iris"), ("train_data", "casia_interval"), ("train_start", None)]),
])
def test_voice_commands(text, mod, expected):
    got = [(a["do"], a.get("mod") or a.get("value") or a.get("url") or a.get("id")) for a in parse(text, {"mod": mod})["actions"]]
    assert [(d, v) for d, v in got] == [(d, v if v is not None else g) for (d, v), (_, g) in zip(expected, got)]


def test_sample_selection_by_voice():
    r = parse("выбери палец сто три оттиск два как образец б", {"mod": "fingerprint"})["actions"]
    assert r == [{"do": "sample", "slot": "b", "kind": "палец", "n": 103, "m": 2}]
    r = parse("сравни палец 101 оттиск 1 и палец 102 оттиск 3", {"mod": "fingerprint"})["actions"]
    assert [(a["do"], a.get("slot"), a.get("n"), a.get("m")) for a in r] == [("sample", "a", 101, 1), ("sample", "b", 102, 3), ("compare", None, None, None)]
    assert parse("выбери глаз пять снимок три", {"mod": "iris"})["actions"][0]["kind"] == "глаз"


def test_relative_and_absolute_threshold():
    assert parse("увеличь порог", {"mod": "face"})["actions"] == [{"do": "threshold", "delta": 0.05}]
    assert parse("уменьши порог на 0,1", {"mod": "face"})["actions"] == [{"do": "threshold", "delta": -0.1}]
    assert parse("порог на 0,3", {"mod": "face"})["actions"] == [{"do": "threshold", "value": 0.3}]


def test_recognition_alternatives_pick_the_phrase_that_is_understood_better():
    r = parse("сравни порок", {"mod": "fingerprint"}, ["сравни порог ноль три", "сравни пароль"])
    assert r["heard"] == "сравни порог ноль три" and [a["do"] for a in r["actions"]] == ["threshold", "compare"]
    r = parse("абракадабра", {"mod": "fingerprint"}, ["тоже ерунда"])
    assert r["understood"] is False and r["heard"] == "абракадабра"


def test_close_phrases_are_suggested_instead_of_guessing():
    r = parse("покажи рельеф", {"mod": "fingerprint"})
    assert [a["do"] for a in r["actions"]] == ["relief3d"]
    r = parse("покажи тройное конвейер", {"mod": "fingerprint"})
    assert r["understood"] is False and "Возможно, вы имели в виду" in r["reply"] and r["examples"]


def test_api_accepts_alternatives():
    with TestClient(app) as c:
        r = c.post("/api/v1/assistant/parse", json={"text": "сравни порок", "alternatives": ["сравни порог ноль пять"], "context": {"mod": "fingerprint"}}).json()
        assert r["heard"] == "сравни порог ноль пять" and r["actions"][0] == {"do": "threshold", "value": 0.5}
        assert c.post("/api/v1/assistant/parse", json={"text": "a", "alternatives": ["x"] * 6}).status_code == 422
