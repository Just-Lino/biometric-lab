from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Biometric Virtual Lab"
    db_path: str = "data/lab.db"
    auto_download_models: bool = True  # скачать YuNet/SFace для модуля лица при старте
    tutor_provider: str = "rule"  # rule | openai
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"


settings = Settings()
