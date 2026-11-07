"""Настройки: значения по умолчанию, переопределение из окружения, границы и адрес базы данных."""
import pytest
from pydantic import ValidationError

from app.core.config import Settings

KEYS = ("DATABASE_URL", "DB_PATH", "TUTOR_RATE_LIMIT_PER_HOUR", "SANDBOX_MAX_CONCURRENT", "COOKIE_SECURE", "PORT",
        "TUTOR_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "AUTO_DOWNLOAD_MODELS")


@pytest.fixture()
def clean_env(monkeypatch):
    """Чистое окружение: на результат не влияют ни переменные компьютера, ни файл .env разработчика."""
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def make(**env) -> Settings:
    return Settings(_env_file=None, **env)


def test_defaults_keep_current_behaviour(clean_env):
    s = make()
    assert s.db_path == "data/lab.db"
    assert s.database_url is None
    assert s.tutor_provider == "rule" and s.openai_api_key is None
    assert s.sandbox_max_concurrent == 2          # столько запусков было и раньше
    assert s.tutor_rate_limit_per_hour == 30
    assert s.cookie_secure is False
    assert s.port == 8000
    assert s.db_pool_max_size == 10


def test_default_database_url_is_sqlite_from_db_path(clean_env):
    assert make().effective_database_url == "sqlite:///data/lab.db"
    assert make(db_path="C:\\Users\\x\\lab.db").effective_database_url == "sqlite:///C:/Users/x/lab.db"  # слэши единообразны
    assert make(db_path="/var/lab/lab.db").effective_database_url == "sqlite:////var/lab/lab.db"
    assert not make().uses_postgres


def test_effective_url_follows_db_path_changes(clean_env):
    """Тесты подменяют db_path после создания настроек: адрес вычисляется лениво и не устаревает."""
    s = make()
    s.db_path = "tmp/other.db"
    assert s.effective_database_url == "sqlite:///tmp/other.db"


def test_explicit_database_url_wins_and_empty_means_unset(clean_env):
    url = "postgresql://user:pass@db.example:5432/lab"
    s = make(database_url=url, db_path="ignored.db")
    assert s.effective_database_url == url and s.uses_postgres
    assert make(database_url="   ").effective_database_url == "sqlite:///data/lab.db"   # пустая строка в .env не ломает запуск


def test_values_come_from_environment(clean_env):
    clean_env.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    clean_env.setenv("TUTOR_RATE_LIMIT_PER_HOUR", "5")
    clean_env.setenv("SANDBOX_MAX_CONCURRENT", "6")
    clean_env.setenv("COOKIE_SECURE", "true")
    clean_env.setenv("PORT", "8080")
    s = make()
    assert s.uses_postgres and s.tutor_rate_limit_per_hour == 5 and s.sandbox_max_concurrent == 6
    assert s.cookie_secure is True and s.port == 8080


def test_zero_rate_limit_is_allowed_and_means_unlimited(clean_env):
    assert make(tutor_rate_limit_per_hour=0).tutor_rate_limit_per_hour == 0


@pytest.mark.parametrize("field, bad", [("sandbox_max_concurrent", 0), ("sandbox_max_concurrent", 1000), ("tutor_rate_limit_per_hour", -1),
                                        ("port", 0), ("port", 70000), ("db_pool_max_size", 0), ("db_pool_max_size", 500)])
def test_out_of_range_values_are_rejected_at_startup(clean_env, field, bad):
    with pytest.raises(ValidationError):
        make(**{field: bad})
