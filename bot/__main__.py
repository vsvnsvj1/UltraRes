import asyncio
import logging

from bot.config import get_config
from bot.main import start_pooling
from bot.utils import setup_logging

config = get_config()
setup_logging(config)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting bot pooling")

    asyncio.run(start_pooling())
