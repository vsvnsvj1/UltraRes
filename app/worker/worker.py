import asyncio
import aio_pika
import cv2
import json
import logging
import numpy as np
from app.model.model import RRDBNet
from app.model.real_esrgan_inference import RESRGANinf
from config import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_VHOST,
    QUEUE_PROCESS_IMAGE,
    QUEUE_RESULT,
    LOG_LEVEL
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
'''
# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
'''


async def load_model(device=None):
    """
    Загружает модель для обработки изображений.
    """
    MODEL_PATH = 'app/model/RealESRGAN_x4plus.pth'
    logger.info("Загрузка модели Real-ESRGAN...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsmpl = RESRGANinf(scale=4, model=model, model_path=MODEL_PATH, device=device,calc_tiles=True, tile_pad=10, pad=10)
    logger.info("Модель успешно загружена.")
    return upsmpl

async def process_image(image_bytes, model):
    """
    Обрабатывает изображение, увеличивая его разрешение.

    Args:
        image_bytes (bytes): Байтовые данные изображения.
        model (RESRGANinf): Объект модели для обработки.
    
    Returns:
        bytes: Обработанное изображение в формате JPEG.
    """
    logger.info("Начало обработки изображения.")
    # Декодируем изображение из байтов
    np_image = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    import os
    save_path = os.path.join('results/', f'{'imgname'}.{'jpg'}')
    cv2.imwrite(save_path, img)
    
    
    if img is None:
        logger.error("Ошибка: не удалось декодировать изображение.")
        raise ValueError("Не удалось декодировать изображение.")

    # Обработка изображения моделью
    logger.info("Обработка изображения с помощью модели...")
    processed_image, _ = model.upgrade_resolution(img)

    # Кодируем обратно в JPEG
    _, encoded_image = cv2.imencode('.jpg', processed_image)
    logger.info("Обработка изображения завершена.")
    return encoded_image.tobytes()

async def main():
    """
    Основная функция, запускающая обработку изображений через очередь.
    """
    logger.info("Инициализация сервиса обработки изображений.")
    model = await load_model()
    logger.info("Подключение к RabbitMQ...")
    connection = await aio_pika.connect_robust(
        f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/{RABBITMQ_VHOST}",
        client_properties={
            "connection_timeout": 300,  # Увеличенный тайм-аут соединения
            "heartbeat": 120,           # Увеличенный heartbeat
        },
    )
    async with connection:
        logger.info("Подключение к RabbitMQ успешно установлено.")
        channel = await connection.channel(publisher_confirms=True)

        # Объявление очередей
        input_queue = await channel.declare_queue(QUEUE_PROCESS_IMAGE, durable=True)
        output_queue_name = QUEUE_RESULT

        async def on_message(message: aio_pika.IncomingMessage):
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
                   
                    # Публикация обработанного изображения в другую очередь
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=processed_image,
                            headers={"chat_id": msg["chat_id"]},
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                        ),
                        routing_key=output_queue_name,
                    )
                    logger.info("Изображение успешно обработано и отправлено в выходную очередь.")
                except Exception as e:
                    logger.error(f"Ошибка при обработке сообщения: {e}",exc_info=True)

        await input_queue.consume(on_message)
        logger.info("Сервис обработки изображений запущен и ожидает сообщений.")
        await asyncio.Future()  # Для ожидания новых сообщений

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Сервис обработки изображений остановлен.")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
