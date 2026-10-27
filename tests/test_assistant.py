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
    ("открой голосовую биометрию", "fingerprint", [("page", "/labplus/voice")]),
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
