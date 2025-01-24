from functools import lru_cache
from typing import final

from pydantic import AmqpDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE_NAME = ".env"


@final
class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_NAME,
        extra="allow",
    )

    # Bot
    TELEGRAM_BOT_TOKEN: str

    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # RabbitMQ
    RABBITMQ_DSN: AmqpDsn

    QUEUE_PROCESS_IMAGE: str
    QUEUE_RESULT: str

    UPLOAD_DIR: str = "uploads"
    RESULT_DIR: str = "results"


@lru_cache
def get_config() -> Config:
    return Config()
