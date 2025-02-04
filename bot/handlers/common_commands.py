import logging

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

logger = logging.getLogger(__name__)

commands_router = Router()


@commands_router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """Обработчик команды /start"""
    logger.info(
        "Command /start from user with id %s",
        message.from_user.id,
    )
    await message.reply(
        "Привет! Я бот для обработки изображений. 👋\n"
        "Отправь мне изображение, и я обработаю его с помощью нейросети. 🖼",
    )


@commands_router.message(Command("help"))
async def command_help_handler(message: Message) -> None:
    """Обработчик команды /help"""
    await message.reply(
        "📝 Инструкция по использованию бота:\n\n"
        "1. Отправьте изображение как фото (не как файл)\n"
        "2. Дождитесь сообщения о начале обработки\n"
        "3. Получите обработанное изображение\n\n"
        "Команды:\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать эту справку",
    )
