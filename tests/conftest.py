import pytest

from app.core.config import settings


@pytest.fixture(autouse=True, scope="session")
def _temp_db(tmp_path_factory):
    """Тесты работают на временной базе и не трогают рабочую data/lab.db студента."""
    old, auto = settings.db_path, settings.auto_download_models
    settings.db_path = str(tmp_path_factory.mktemp("db") / "test.db")
    settings.auto_download_models = False  # тесты не скачивают модели
    yield
    settings.db_path, settings.auto_download_models = old, auto


@pytest.fixture(autouse=True)
def _no_trained_models():
    """Обученные сети хранятся глобально (и на диске рядом с базой): перед каждым тестом их нет, чтобы тесты не влияли друг на друга."""
    import shutil
    from pathlib import Path

    from app.training import registry
    registry.reset_cache()
    shutil.rmtree(Path(settings.db_path).resolve().parent / "trained", ignore_errors=True)
    yield
    registry.reset_cache()


@pytest.fixture(autouse=True)
def _rule_tutor(monkeypatch):
    """Тесты не ходят во внешние API: наставник по правилам, даже если в .env включён OpenAI (и ключ не тратится)."""
    from app.services import tutor
    monkeypatch.setattr(settings, "tutor_provider", "rule")
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(tutor, "_tutor", None)
    yield
    tutor._tutor = None
