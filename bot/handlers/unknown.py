from aiogram import Router
from aiogram.types import Message

unknown_router = Router()


@unknown_router.message()
async def unknown_handler(message: Message):
    """Обработчик всех неизвестных сообщений"""
    await message.reply(
        "🤔 Я понимаю только команды и изображения.\n"
        "Отправьте /help для получения справки.",
    )
