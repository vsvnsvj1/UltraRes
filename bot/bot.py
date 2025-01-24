import logging

from aiogram import Dispatcher, Router
from aiogram_dialog import setup_dialogs

from bot.config import get_config
from bot.misc import bot, dp
from bot.handlers import all_routers
from bot.utils import setup_logging

config = get_config()
setup_logging(config)
logger = logging.getLogger(__name__)

main_router = Router()


def register_dialogs(router: Router):
    """
    Register all dialogs in the router
    """
    for handler_router in all_routers:
        router.include_router(handler_router)


async def setup_dispatcher(dp: Dispatcher):
    dp.include_router(main_router)
    register_dialogs(dp)
    setup_dialogs(dp)


async def start_pooling():
    await setup_dispatcher(dp)
    await dp.start_polling(bot, skip_updates=True)
