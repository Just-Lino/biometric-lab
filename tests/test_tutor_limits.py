"""Лимит запросов к наставнику на сессию, кэш одинаковых вопросов и границы входных данных."""
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.cache import LRUCache  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.ratelimit import SlidingWindowLimiter  # noqa: E402
from app.main import app  # noqa: E402
from app.services import tutor as tutor_service  # noqa: E402


class FakeModel(tutor_service.TutorProvider):
    """Поставщик, похожий на OpenAI (не «rule»): считает обращения к «модели». Ответы по очереди берёт из replies."""
    name = "openai"

    def __init__(self, replies=None):
        self.calls = 0
        self.replies = replies

    def ask_detailed(self, mode, question, context):
        self.calls += 1
        if self.replies:
            return self.replies.pop(0)
        return f"ответ {self.calls}: {question}", True

    def ask(self, mode, question, context):
        return self.ask_detailed(mode, question, context)[0]


@pytest.fixture()
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(tutor_service, "_tutor", fake)
    monkeypatch.setattr(tutor_service, "_answers", LRUCache(500, ttl=3600))
    monkeypatch.setattr(tutor_service, "_limiter", SlidingWindowLimiter(3600.0))
    monkeypatch.setattr(settings, "tutor_rate_limit_per_hour", 3)
    return fake


def ask(client, question="что такое EER", mode="explain", context=None):
    return client.post("/api/v1/tutor/ask", json={"mode": mode, "question": question, "context": context or {}})


def test_limit_per_session_with_clear_russian_message(model):
    with TestClient(app) as c:
        for i in range(3):
            assert ask(c, f"вопрос {i}").status_code == 200
        r = ask(c, "ещё один")
        assert r.status_code == 429
        assert "лимит" in r.json()["detail"] and "3 в час" in r.json()["detail"] and "мин" in r.json()["detail"]
        assert 1 <= int(r.headers["Retry-After"]) <= 3600
        assert model.calls == 3                          # к модели отклонённый запрос не дошёл


def test_limit_is_per_session(model):
    with TestClient(app) as a, TestClient(app) as b:      # у каждого клиента своя cookie lab_sid
        for i in range(3):
            assert ask(a, f"вопрос {i}").status_code == 200
        assert ask(a, "четвёртый").status_code == 429
        assert ask(b, "вопрос другого студента").status_code == 200


def test_identical_question_is_cached_and_does_not_spend_the_limit(model, monkeypatch):
    monkeypatch.setattr(settings, "tutor_rate_limit_per_hour", 1)
    with TestClient(app) as c:
        first = ask(c, "Что такое EER?").json()
        again = ask(c, "  что   ТАКОЕ eer?  ").json()      # регистр и пробелы вопроса значения не имеют
        assert again["answer"] == first["answer"] and model.calls == 1
        assert ask(c, "другой вопрос").status_code == 429  # единственный платный запрос уже потрачен


def test_cache_is_shared_between_sessions_but_context_matters(model):
    ctx = {"similarity": 0.9, "threshold": 0.5}
    with TestClient(app) as a, TestClient(app) as b:
        ask(a, "разбери", "diagnose", ctx)
        ask(b, "разбери", "diagnose", ctx)
        assert model.calls == 1                            # тот же вопрос и те же числа: ответ общий, студент б не тратит лимит
        ask(b, "разбери", "diagnose", {"similarity": 0.1, "threshold": 0.5})
        assert model.calls == 2                            # другие числа: другой ответ


def test_fallback_answers_are_not_cached(model):
    model.replies = [("ответ по правилам после сбоя модели", False), ("настоящий ответ модели", True)]
    with TestClient(app) as c:
        assert ask(c, "вопрос").json()["answer"].startswith("ответ по правилам")
        assert ask(c, "вопрос").json()["answer"] == "настоящий ответ модели"   # сбой не «залип» в кэше на час
        assert ask(c, "вопрос").json()["answer"] == "настоящий ответ модели" and model.calls == 2


def test_zero_limit_disables_limiting(model, monkeypatch):
    monkeypatch.setattr(settings, "tutor_rate_limit_per_hour", 0)
    with TestClient(app) as c:
        assert all(ask(c, f"вопрос {i}").status_code == 200 for i in range(12))


def test_rule_based_tutor_is_not_limited(monkeypatch):
    monkeypatch.setattr(tutor_service, "_tutor", tutor_service.RuleBasedTutor())
    monkeypatch.setattr(tutor_service, "_limiter", SlidingWindowLimiter(3600.0))
    monkeypatch.setattr(settings, "tutor_rate_limit_per_hour", 1)
    with TestClient(app) as c:
        for i in range(5):
            r = ask(c, f"что такое FAR {i}")
            assert r.status_code == 200 and r.json()["provider"] == "rule"


def test_limited_requests_are_logged_as_events(model):
    from app.core.db import connect
    with TestClient(app) as c:
        for i in range(4):
            ask(c, f"вопрос {i}")
    with connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE type='tutor_limited'").fetchone()[0] >= 1


def test_oversized_input_is_rejected(model):
    with TestClient(app) as c:
        assert ask(c, "а" * 2001).status_code == 422
        assert ask(c, "ок", context={"x": "я" * 25_000}).status_code == 422
        assert model.calls == 0


# ───────────────────────── ограничитель и кэш отдельно ─────────────────────────
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_sliding_window_expires_and_rejected_hits_do_not_extend_it():
    clock = Clock()
    lim = SlidingWindowLimiter(window=3600.0, clock=clock)
    assert [lim.hit("s", 3)[0] for _ in range(3)] == [True, True, True]
    clock.t += 600
    ok, wait = lim.hit("s", 3)
    assert not ok and wait == 3000                          # окно началось с первого обращения: 3600 - 600
    clock.t += 1000
    ok, wait = lim.hit("s", 3)
    assert not ok and wait == 2000                          # повторные попытки ожидание не увеличивают
    clock.t += 2000
    assert lim.hit("s", 3)[0] is True                       # первые три обращения вышли из окна
    assert lim.hit("other", 3)[0] is True                   # другой ключ не затронут


def test_limiter_bounds_memory():
    clock = Clock()
    lim = SlidingWindowLimiter(window=100.0, max_keys=50, clock=clock)
    for i in range(500):
        clock.t += 1
        lim.hit(f"session{i}", 5)
    assert len(lim._hits) <= 50


def test_cache_ttl_expiry_and_plain_lru_unchanged():
    clock = Clock()
    c = LRUCache(2, ttl=60, clock=clock)
    c.put("a", 1)
    clock.t += 59
    assert c.get("a") == 1
    clock.t += 2
    assert c.get("a") is None and len(c) == 0
    plain = LRUCache(2)                                     # без ttl поведение прежнее
    plain.put("a", 1); plain.put("b", 2); plain.get("a"); plain.put("c", 3)
    assert plain.get("b") is None and plain.get("a") == 1 and plain.get("c") == 3
