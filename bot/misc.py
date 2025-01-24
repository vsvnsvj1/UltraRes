from aiogram import Bot, Dispatcher

from bot.config import get_config

config = get_config()

# TODO add storage
dp = Dispatcher()

# Bot
bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
