import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiogram.types import Message, PhotoSize
from app.bot.bot import send_welcome, send_help, handle_photo, handle_unknown
from app.bot.producer import ImageProducer

# Test bot.py functionality
@pytest.mark.asyncio
async def test_start_command():
    message = AsyncMock()
    message.reply = AsyncMock()

    await send_welcome(message)  # Directly call the handler for /start
    message.reply.assert_called_with(
        "Привет! Я бот для обработки изображений. 👋\n"
        "Отправь мне изображение, и я обработаю его с помощью нейросети. 🖼"
    )

@pytest.mark.asyncio
async def test_help_command():
    message = AsyncMock()
    message.reply = AsyncMock()

    await send_help(message)  # Directly call the handler for /help
    message.reply.assert_called_with(
        "📝 Инструкция по использованию бота:\n\n"
        "1. Отправьте изображение как фото (не как файл)\n"
        "2. Дождитесь сообщения о начале обработки\n"
        "3. Получите обработанное изображение\n\n"
        "Команды:\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать эту справку"
    )

@pytest.mark.asyncio
async def test_handle_photo():
    message = AsyncMock()
    message.reply = AsyncMock()
    message.photo = [PhotoSize(file_id="file_123", file_unique_id="unique_id_123", width=1280, height=720)]

    with patch("app.bot.producer.ImageProducer.send_image", new_callable=AsyncMock) as mock_send_image, \
         patch("app.bot.bot.bot.download", new_callable=AsyncMock) as mock_download:
        mock_download.return_value = None  # Мокаем успешное выполнение скачивания
        await handle_photo(message)  # Directly call the handler for photo messages
        message.reply.assert_called_with("📥 Получил ваше изображение. Начинаю обработку...")
        mock_send_image.assert_awaited_once()

@pytest.mark.asyncio
async def test_handle_unknown():
    message = AsyncMock()
    message.reply = AsyncMock()

    await handle_unknown(message)  # Directly call the handler for unknown messages
    message.reply.assert_called_with(
        "🤔 Я понимаю только команды и изображения.\n"
        "Отправьте /help для получения справки."
    )