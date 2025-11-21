from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    bybit_api_key: Optional[str] = None
    bybit_api_secret: Optional[str] = None
    llm_model_name: str = "gpt-4o"
    llm_provider: str = "openai"
    llm_system_prompt: Optional[str] = None
    llm_system_prompt_file: Optional[str] = None
    confidence_threshold: float = 70.0
    max_sl_adjustment_factor: float = 1.5  # bounds for adjusted SL vs input
    min_sl_adjustment_factor: float = 0.5
    min_position_size_pct: float = 50.0
    max_position_size_pct: float = 150.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
