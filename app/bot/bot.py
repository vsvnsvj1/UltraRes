from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import ContentType
import os
from config import TELEGRAM_BOT_TOKEN, UPLOAD_DIR
from .producer import ImageProducer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)
producer = ImageProducer()

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply(
        "Привет! Отправь мне изображение, и я обработаю его с помощью нейросети."
    )

@dp.message_handler(content_types=[ContentType.PHOTO])
async def handle_photo(message: types.Message):
    try:
        # Отправляем сообщение о начале обработки
        await message.reply("Получил изображение. Начинаю обработку...")

        # Получаем информацию о фото
        photo = message.photo[-1]  # Берем максимальное разрешение
        
        # Создаем уникальное имя файла
        file_name = f"{message.from_user.id}_{photo.file_id}.jpg"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        # Скачиваем фото
        await photo.download(destination_file=file_path)
        
        # Отправляем в очередь
        await producer.send_image(file_path, message.from_user.id)
        
        await message.reply(
            "Изображение отправлено на обработку. "
            "Я пришлю результат, как только он будет готов."
        )

    except Exception as e:
        error_msg = f"Произошла ошибка при обработке изображения: {str(e)}"
        logger.error(error_msg)
        await message.reply(error_msg)

async def on_startup(dp):
    """Действия при запуске бота"""
    await producer.connect()

async def on_shutdown(dp):
    """Действия при остановке бота"""
    await producer.close()

if __name__ == '__main__':
    executor.start_polling(
        dp,
        skip_updates=True,
        on_startup=on_startup,
        on_shutdown=on_shutdown
    )