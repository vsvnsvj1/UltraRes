import asyncio
import logging

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import ContentType
from config import LOG_LEVEL, TELEGRAM_BOT_TOKEN

from .producer import ImageProducer

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
# Иницилизация продюсера
producer = ImageProducer(bot)


@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    """Обработчик команды /start"""
    await message.reply(
        "Привет! Я бот для обработки изображений. 👋\n"
        "Отправь мне изображение, и я обработаю его с помощью нейросети. 🖼",
    )


@dp.message(Command("help"))
async def send_help(message: types.Message):
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


@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: types.Message):
    """Обработчик входящих фотографий"""
    try:
        # Отправляем сообщение о начале обработки
        processing_msg = await message.reply(
            "📥 Получил ваше изображение. Начинаю обработку...",
        )

        photo = message.photo[-1]
        image_bytes = await bot.download(photo.file_id)
        image_bytes = image_bytes.read()

        await producer.send_image(image_bytes, message.from_user.id)

        await processing_msg.edit_text(
            "🔄 Изображение отправлено на обработку.\n"
            "⏳ Я пришлю результат, как только он будет готов.",
        )

    except Exception as e:
        error_msg = f"❌ Произошла ошибка при обработке изображения: {str(e)}"
        logger.error(error_msg)
        await message.reply(error_msg)


@dp.message()
async def handle_unknown(message: types.Message):
    """Обработчик всех остальных сообщений"""
    await message.reply(
        "🤔 Я понимаю только команды и изображения.\n" "Отправьте /help для получения справки.",
    )


async def main():

    await producer.connect()
    # Запускаем подписку на результаты
    asyncio.create_task(producer.process_result())
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        producer.close()
        logger.info("Получен сигнал завершения работы.")
    except KeyboardInterrupt:
        producer.close()
        logger.info("Получен KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        logger.info("Программа завершена")
