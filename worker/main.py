import asyncio
import json
import logging

import aio_pika
import cv2
import numpy as np
from model import RESRGANinf, RRDBNet

from worker.config import get_config
from worker.utils import setup_logging

config = get_config()
setup_logging(config)
logger = logging.getLogger(__name__)


SEMAPHORE_LIMIT = 2
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


async def load_model(device=None):
    """
    Загружает модель для обработки изображений.
    """
    logger.info("Загрузка модели Real-ESRGAN...")
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    upsmpl = RESRGANinf(
        scale=4,
        model=model,
        model_path="model/RealESRGAN_x4plus.pth",
        device=device,
        calc_tiles=True,
        tile_pad=10,
        pad=10,
    )
    logger.info("Модель успешно загружена.")
    return upsmpl


async def process_image(image_bytes, model):
    """
    Обрабатывает изображение, увеличивая его разрешение.

    Args:
        image_bytes (bytes): Байтовые данные изображения.
        model (RESRGANinf): Объект модели для обработки.

    Returns:
        bytes: Обработанное изображение в байтах.
    """
    logger.info("Начало обработки изображения.")

    np_image = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if img is None:
        logger.error("Ошибка: не удалось декодировать изображение.")
        raise ValueError("Не удалось декодировать изображение.")

    # Обработка изображения моделью
    logger.info("Обработка изображения с помощью модели...")
    processed_image, _ = model.upgrade_resolution(img)

    # Кодируем обратно в JPEG
    _, encoded_image = cv2.imencode(".jpg", processed_image)
    logger.info("Обработка изображения завершена.")
    return encoded_image.tobytes()


async def publish_with_retry(publisher_channel, message, routing_key, retries=3):
    """
    Публикует сообщение в очередь с повторными попытками.
    """
    for attempt in range(retries):
        try:
            await publisher_channel.default_exchange.publish(
                message,
                routing_key=routing_key,
            )
            logger.info("Сообщение успешно опубликовано.")
            return
        except Exception as e:
            logger.error(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2**attempt)  # Экспоненциальная задержка
            else:
                raise


async def handle_message(
    message: aio_pika.IncomingMessage,
    model,
    publisher_channel,
    output_queue_name,
):
    """
    Обрабатывает сообщение из очереди RabbitMQ.

    Args:
        message (aio_pika.IncomingMessage): Входящее сообщение из RabbitMQ.
        model: Объект модели для обработки изображений.
        publisher_channel: Постоянный канал для публикации результата.
        output_queue_name (str): Имя очереди для отправки результата.
    """
    async with semaphore:
        try:
            logger.info("Получено сообщение из очереди.")
            # Декодируем сообщение
            msg = json.loads(message.body)
            image_hex = msg.get("image_data")

            if not image_hex:
                logger.error("Ошибка: получены пустые данные изображения.")
                raise ValueError("Получены пустые данные изображения.")

            # Декодируем байты изображения
            image_bytes = bytes.fromhex(image_hex)

            # Обработка изображения
            logger.info("Начинается обработка изображения...")
            processed_image = await process_image(image_bytes, model)

            # Создаём сообщение
            message_to_publish = aio_pika.Message(
                body=processed_image,
                headers={"chat_id": msg["chat_id"]},
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            # Публикация с повторными попытками
            await publish_with_retry(
                publisher_channel,
                message_to_publish,
                output_queue_name,
            )

            await message.ack()
            logger.info("Исходное сообщение подтверждено (ack).")
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}", exc_info=True)
            await message.reject(
                requeue=False,
            )


async def main(device: str = None):
    """
    Основная функция, запускающая обработку изображений через очередь.
    """
    logger.info("Инициализация сервиса обработки изображений.")

    if device:
        logger.warning(f"Используется устройство: {device}")
    else:
        logger.warning("Устройство не указано, используется значение по умолчанию.")

    model = await load_model(device)

    logger.info("Подключение к RabbitMQ...")

    rabbitmq_url = str(config.RABBITMQ_DSN)

    client_props = {
        "connection_timeout": 300,  # Время ожидания соединения
        "heartbeat": 600,  # Интервал heartbeat
    }

    connection = await aio_pika.connect_robust(
        rabbitmq_url,
        client_properties=client_props,
    )

    try:
        async with connection:
            logger.info("Подключение к RabbitMQ успешно установлено.")
            channel = await connection.channel(publisher_confirms=True)
            publisher_channel = await connection.channel(publisher_confirms=True)

            # Объявление очередей
            input_queue = await channel.declare_queue(
                config.QUEUE_PROCESS_IMAGE, durable=True,
            )
            output_queue_name = config.QUEUE_RESULT

            # Устанавливаем prefetch_count
            await channel.set_qos(prefetch_count=SEMAPHORE_LIMIT)

            # Привязываем обработчик сообщений
            await input_queue.consume(
                lambda msg: handle_message(
                    msg,
                    model,
                    publisher_channel,
                    output_queue_name,
                ),
            )

            logger.info("Сервис обработки изображений запущен и ожидает сообщений.")
            await asyncio.Future()  # Бесконечное ожидаение сообщений
    except asyncio.CancelledError:
        logger.info("Получен сигнал завершения работы.")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
    finally:
        # Корректное закрытие соединения
        if not connection.is_closed:
            await connection.close()
            logger.info("Соединение с RabbitMQ закрыто.")
