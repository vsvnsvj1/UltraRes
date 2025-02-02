import os
from functools import lru_cache
from typing import final

from pydantic import AmqpDsn, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE_NAME = os.getenv("ENV_FILE_NAME", ".env")


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
    # RabbitMQ
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672

    # RabbitMQ queues
    QUEUE_PROCESS_IMAGE: str = Field(default="process_image_queue")
    QUEUE_RESULT: str = Field(default="result_queue")

    UPLOAD_DIR: str = "uploads"
    RESULT_DIR: str = "results"

    @property
    def RABBITMQ_DSN(self) -> AmqpDsn:
        return AmqpDsn(f"amqp://"
                       f"{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@"
                       f"{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/")


@lru_cache
def get_config() -> Config:
    return Config()
