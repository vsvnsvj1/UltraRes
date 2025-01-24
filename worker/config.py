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

    LOG_LEVEL: str = "INFO"

    # RabbitMQ
    RABBITMQ_DSN: AmqpDsn

    QUEUE_PROCESS_IMAGE: str
    QUEUE_RESULT: str


@lru_cache
def get_config() -> Config:
    return Config()
