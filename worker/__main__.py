import asyncio
import logging
import argparse

from worker.config import get_config
from worker.utils import setup_logging
from worker.main import main

config = get_config()
setup_logging(config)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сервис обработки изображений")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство для использования (например, 'cuda:0'). Если не указано, используется значение по умолчанию."
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(device=args.device))
    except KeyboardInterrupt:
        logger.info("Сервис обработки изображений остановлен.")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
