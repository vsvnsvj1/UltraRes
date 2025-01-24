import asyncio
import logging

from worker.config import get_config
from worker.utils import setup_logging
from worker.worker import main

config = get_config()
setup_logging(config)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Сервис обработки изображений остановлен.")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
