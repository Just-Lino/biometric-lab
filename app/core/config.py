from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки из переменных окружения и файла .env (имена переменных: поле в верхнем регистре, например DATABASE_URL)."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Biometric Virtual Lab"

    # --- база данных ---
    db_path: str = "data/lab.db"  # файл SQLite: локальная работа и тесты
    # Адрес базы данных. Не задан: используется SQLite из DB_PATH. Задан (например, postgresql://...): используется он.
    # Значение читайте через effective_database_url: db_path меняется в тестах уже после создания настроек.
    database_url: str | None = None
    # Максимум одновременных соединений с PostgreSQL в одном экземпляре сервера (для SQLite не используется).
    # Держите его меньше лимита соединений самой базы, разделённого на число экземпляров.
    db_pool_max_size: int = Field(10, ge=1, le=100)

    # --- ИИ-наставник ---
    tutor_provider: str = "rule"  # rule | openai
    openai_api_key: str | None = None  # только из переменной окружения или .env, в код и в Git не попадает
    openai_model: str = "gpt-4o-mini"
    # Сколько запросов к наставнику в час допускается с одной сессии (cookie lab_sid); 0 отключает ограничение.
    tutor_rate_limit_per_hour: int = Field(30, ge=0, le=100_000)

    # --- песочница практикума ---
    sandbox_max_concurrent: int = Field(2, ge=1, le=64)  # сколько запусков кода студентов идёт одновременно

    # --- сервер и безопасность ---
    port: int = Field(8000, ge=1, le=65535)  # порт, на котором слушает контейнер (в Cloud Run задаётся платформой)
    subject_header: str | None = None  # имя заголовка с ID пользователя от сайта-хозяина (например X-User-Id); не задан: пользователи анонимны
    cookie_secure: bool = False  # true: cookie сессии с флагом Secure (только по HTTPS); локально по http оставьте false

    auto_download_models: bool = True  # скачать YuNet/SFace для модуля лица при старте

    @property
    def effective_database_url(self) -> str:
        """Рабочий адрес базы: DATABASE_URL, а если он не задан или пуст, то SQLite-файл из DB_PATH."""
        url = (self.database_url or "").strip()
        return url or f"sqlite:///{Path(self.db_path).as_posix()}"

    @property
    def uses_postgres(self) -> bool:
        return self.effective_database_url.split(":", 1)[0].lower().startswith("postgres")


settings = Settings()
