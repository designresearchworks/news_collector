from __future__ import annotations

"""
config.py — Application settings loaded from a .env file.

Copy .env.example to .env and fill in your values before running.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Required: your OpenRouter API key
    openrouter_api_key: str

    # The model to use via OpenRouter (any OpenRouter model string works)
    # Defaults to Claude 3.5 Haiku — fast, cheap, great for conversational tasks
    model: str = "anthropic/claude-3-5-haiku-20241022"

    # Path to the SQLite database file
    db_path: str = "news.db"

    # Display name shown in the browser tab and page headers
    app_title: str = "News Collector"

    # Server bind address and port
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
