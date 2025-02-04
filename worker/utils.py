import logging
import sys
from worker.config import Config


def setup_logging(config: Config) -> None:
    logging_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    # Настройка базового логирования
    logging.basicConfig(
        level=logging_level,
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
