import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.bot.producer import ImageProducer

@pytest.mark.asyncio
async def test_image_producer_connect():
    producer = ImageProducer(bot=MagicMock())

    with patch("aio_pika.connect_robust", new_callable=AsyncMock) as mock_connect:
        producer.channel = AsyncMock()

        # Мокируем очередь и её метод iterator
        producer.queue_result = AsyncMock()
        producer.queue_result.iterator = AsyncMock()

        # Настраиваем iterator как асинхронный контекстный менеджер
        mock_iterator = AsyncMock()
        mock_iterator.__aenter__.return_value = AsyncMock()  # Это будет возвращать объект, который можно итерировать
        mock_iterator.__aexit__.return_value = None
        producer.queue_result.iterator.return_value = mock_iterator

        await producer.connect()
        mock_connect.assert_called_once()

@pytest.mark.asyncio
async def test_image_producer_send_image():
    bot_mock = MagicMock()
    producer = ImageProducer(bot=bot_mock)
    producer.channel = AsyncMock()
    producer.channel.default_exchange = AsyncMock()

    await producer.send_image("/path/to/image.jpg", 12345)
    producer.channel.default_exchange.publish.assert_awaited_once()

@pytest.mark.asyncio
async def test_image_producer_process_result():
    bot_mock = AsyncMock()
    producer = ImageProducer(bot=bot_mock)

    message_mock = AsyncMock()
    message_mock.body = b'{"user_id": 12345, "result_image_path": "/path/to/result.jpg"}'

    # Настраиваем process как асинхронный контекстный менеджер
    mock_context_manager = AsyncMock()
    message_mock.process.return_value = mock_context_manager

    with patch("builtins.open", MagicMock()) as mock_open:
        await producer.process_result(message_mock)

    bot_mock.send_photo.assert_awaited_with(
        chat_id=12345,
        photo=mock_open.return_value,
        caption="Вот ваше обработанное изображение!"
    )



@pytest.mark.asyncio
async def test_image_producer_close():
    # Настраиваем моки
    producer = ImageProducer(bot=MagicMock())
    producer.connection = AsyncMock()
    producer.channel = AsyncMock()

    # Настраиваем is_closed для корректного поведения
    producer.connection.is_closed = False
    producer.channel.is_closed = False

    # Вызываем метод close
    await producer.close()

    # Проверяем, что оба метода были вызваны
    producer.connection.close.assert_awaited_once()
    producer.channel.close.assert_awaited_once()
