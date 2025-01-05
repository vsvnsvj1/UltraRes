from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ContentType
from aiogram.filters import Command
import asyncio
import signal
import os
from config import TELEGRAM_BOT_TOKEN, UPLOAD_DIR
from .producer import ImageProducer
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
producer = ImageProducer(bot)

@dp.message(Command('start'))
async def send_welcome(message: types.Message):
    """Обработчик команды /start"""
    await message.reply(
        "Привет! Я бот для обработки изображений. 👋\n"
        "Отправь мне изображение, и я обработаю его с помощью нейросети. 🖼"
    )

@dp.message(Command('help'))
async def send_help(message: types.Message):
    """Обработчик команды /help"""
    await message.reply(
        "📝 Инструкция по использованию бота:\n\n"
        "1. Отправьте изображение как фото (не как файл)\n"
        "2. Дождитесь сообщения о начале обработки\n"
        "3. Получите обработанное изображение\n\n"
        "Команды:\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать эту справку"
    )

@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: types.Message):
    """Обработчик входящих фотографий"""
    try:
        # Отправляем сообщение о начале обработки
        processing_msg = await message.reply(
            "📥 Получил ваше изображение. Начинаю обработку..."
        )

        # Получаем информацию о фото (берем максимальное разрешение)
        photo = message.photo[-1]
        
        # Создаем уникальное имя файла
        file_name = f"{message.from_user.id}_{photo.file_id}.jpg"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        # Скачиваем фото
        await bot.download(
            photo.file_id,
            destination=file_path
        )
        
        # Отправляем в очередь
        await producer.send_image(file_path, message.from_user.id)
        
        # Обновляем сообщение о статусе
        await processing_msg.edit_text(
            "🔄 Изображение отправлено на обработку.\n"
            "⏳ Я пришлю результат, как только он будет готов."
        )

    except Exception as e:
        error_msg = f"❌ Произошла ошибка при обработке изображения: {str(e)}"
        logger.error(error_msg)
        await message.reply(error_msg)

@dp.message()
async def handle_unknown(message: types.Message):
    """Обработчик всех остальных сообщений"""
    await message.reply(
        "🤔 Я понимаю только команды и изображения.\n"
        "Отправьте /help для получения справки."
    )

async def main():
    # Создаем директорию для загрузок, если её нет
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Создаем объект для отслеживания завершения
    stop = asyncio.Event()
    
    async def shutdown():
        """Корректное завершение всех соединений"""
        logger.info("Начинаю процесс завершения работы...")
        
        # Останавливаем поллинг бота
        stop.set()
        
        try:
            # Закрываем соединение с RabbitMQ
            await producer.close()
        except Exception as e:
            logger.error(f"Ошибка при закрытии RabbitMQ: {e}")
        
        try:
            # Закрываем сессию бота
            await bot.session.close()
        except Exception as e:
            logger.error(f"Ошибка при закрытии сессии бота: {e}")
        
        logger.info("Бот успешно остановлен")

    def signal_handler(signame):
        """Обработчик сигналов"""
        logger.info(f"Получен сигнал завершения: {signame}")
        # Запускаем завершение в event loop
        asyncio.create_task(shutdown())

    # Регистрируем обработчики сигналов
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(
            sig,
            lambda s=sig: signal_handler(s)
        )

    try:
        # Подключаемся к RabbitMQ
        await producer.connect()
        
        logger.info("Бот запущен")
        
        # Запускаем поллинг с таймаутом
        await dp.start_polling(
            bot,
            stop_event=stop,
            polling_timeout=30  # Добавляем таймаут
        )
    except Exception as e:
        logger.error(f"Ошибка при работе бота: {e}")
        raise
    finally:
        await shutdown()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Получен KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        logger.info("Программа завершена")