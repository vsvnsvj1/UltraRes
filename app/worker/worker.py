import asyncio
import json
import logging

import aio_pika
import cv2
import numpy as np

from app.model.model import RRDBNet
from app.model.real_esrgan_inference import RESRGANinf
from config import (
    LOG_LEVEL,
    QUEUE_PROCESS_IMAGE,
    QUEUE_RESULT,
    RABBITMQ_HOST,
    RABBITMQ_PASSWORD,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_VHOST,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SEMAPHORE_LIMIT = 2  # Максимальное количество параллельных задач
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


async def load_model(device=None):
    """
    Загружает модель для обработки изображений.
    """
    logger.info("Загрузка модели Real-ESRGAN...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsmpl = RESRGANinf(
        scale=4,
        model=model,
        model_path="app/model/RealESRGAN_x4plus.pth",
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


async def publish_with_retry(connection, message, routing_key, retries=3):
    """
    Публикует сообщение в очередь с повторными попытками.
    """
    for attempt in range(retries):
        try:
            async with connection.channel() as channel:
                await channel.default_exchange.publish(
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


async def handle_message(message: aio_pika.IncomingMessage, model, connection, output_queue_name):
    """
    Обрабатывает сообщение из очереди RabbitMQ.

    Args:
        message (aio_pika.IncomingMessage): Входящее сообщение из RabbitMQ.
        model: Объект модели для обработки изображений.
        connection: Соединение RabbitMQ для публикации сообщений.
        output_queue_name (str): Имя очереди для отправки результата.
    """
    async with semaphore:
        async with message.process():
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
                await publish_with_retry(connection, message_to_publish, output_queue_name)
            except Exception as e:
                logger.error(f"Ошибка при обработке сообщения: {e}", exc_info=True)
                await message.reject(requeue=False)  # Отклоняем сообщение без повторной отправки


async def main():
    """
    Основная функция, запускающая обработку изображений через очередь.
    """
    logger.info("Инициализация сервиса обработки изображений.")
    model = await load_model()

    logger.info("Подключение к RabbitMQ...")

    rabbitmq_url = (
        f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}"
        f"@{RABBITMQ_HOST}:{RABBITMQ_PORT}/{RABBITMQ_VHOST}"
    )

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

            # Объявление очередей
            input_queue = await channel.declare_queue(QUEUE_PROCESS_IMAGE, durable=True)
            output_queue_name = QUEUE_RESULT

            # Устанавливаем prefetch_count
            await channel.set_qos(prefetch_count=SEMAPHORE_LIMIT)

            # Привязываем обработчик сообщений
            await input_queue.consume(
                lambda msg: handle_message(
                    msg,
                    model,
                    connection,
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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Сервис обработки изображений остановлен.")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
