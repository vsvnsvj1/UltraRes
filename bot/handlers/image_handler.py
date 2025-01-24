import logging

from aiogram import F
from aiogram import Router, Bot
from aiogram.types import Message, ContentType
from bot.misc import rabbit_manager
from bot.scripts.message_scripts import create_json_from_message

from bot.config import get_config


config = get_config()

logger = logging.getLogger(__name__)

image_router = Router()


@image_router.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: Message, bot: Bot):
    """Обработчик входящих фотографий"""
    try:
        # Отправляем сообщение о начале обработки
        processing_msg = await message.reply(
            "📥 Получил ваше изображение. Начинаю обработку...",
        )

        photo_bytes = await bot.download(message.photo[-1])
        photo_bytes = photo_bytes.read()

        await rabbit_manager.send_json_to_queue(
            create_json_from_message(
                message.from_user.id,
                photo_bytes,
                ),
            )

        await processing_msg.edit_text(
            "🔄 Изображение отправлено на обработку.\n"
            "⏳ Я пришлю результат, как только он будет готов.",
        )

    except Exception as e:
        error_msg = f"❌ Произошла ошибка при обработке изображения: {str(e)}"
        logger.error(error_msg)
        await message.reply(error_msg)
