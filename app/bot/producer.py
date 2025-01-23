import json
import logging
import os
from datetime import datetime
from typing import Optional

import aio_pika
from aiogram import Bot
from aiogram.types import BufferedInputFile
from config import (
    DEBUG,
    QUEUE_PROCESS_IMAGE,
    QUEUE_RESULT,
    RABBITMQ_HOST,
    RABBITMQ_PASSWORD,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_VHOST,
    RESULT_DIR,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)


class ImageProducer:
    def __init__(self, bot: Bot):
        self.connection: Optional[aio_pika.Connection] = None
        self.channel = None
        self.queue_process = QUEUE_PROCESS_IMAGE
        self.queue_result = QUEUE_RESULT
        self.bot = bot
        self._consuming = False
        self._consume_task = None

    @staticmethod
    async def _save_image_to_dir(
        image_bytes: bytes,
        chat_id: int,
        dir_name: str,
    ) -> str:
        """
        Сохраняет изображение в директорию dir_name, если DEBUG включен.

        Args:
            image_bytes (bytes): Байты изображения.
            chat_id (int): ID пользователя.
            dir_name (str): Название директории

        Returns:
            str: Путь к сохраненному файлу.
        """
        if not DEBUG:
            return None

        os.makedirs(dir_name, exist_ok=True)
        file_name = f"{chat_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg"
        file_path = os.path.join(dir_name, file_name)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        logger.debug(f"Изображение сохранено в {file_path}")
        return file_path

    async def connect(self) -> None:
        """
        Устанавливает соединение с RabbitMQ.
        """
        try:
            self.connection = await aio_pika.connect_robust(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                login=RABBITMQ_USER,
                password=RABBITMQ_PASSWORD,
                virtualhost=RABBITMQ_VHOST,
            )
            self.channel = await self.connection.channel()
            logger.info("Подключение к RabbitMQ установлено")
        except Exception as e:
            logger.error(f"Ошибка при подключении к RabbitMQ: {e}")
            raise e

    async def close(self) -> None:
        """
        Корректное закрытие соединения с RabbitMQ
        """
        try:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            logger.info("Соединение с RabbitMQ закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения с RabbitMQ: {e}")

    async def send_image(self, image_bytes: str, chat_id: int) -> None:
        """
        Отправляет байтовое изображение в очередь RabbitMQ.

        Args:
            image_bytes (bytes): Байты изображения.
            chat_id (int): ID чата для отправки результата.
        """

        await self._save_image_to_dir(
            image_bytes,
            chat_id,
            UPLOAD_DIR,
        )

        try:
            if not self.channel:
                raise ConnectionError("Канал RabbitMQ не установлен.")

            message = {
                "chat_id": chat_id,
                "image_data": image_bytes.hex(),
            }

            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=self.queue_process,
            )

            logger.info(f"Изображение отправлено в очередь для чата {chat_id}")
        except Exception as e:
            logger.error(f"Ошибка отправки изображения в очередь: {e}")
            raise e

    async def _process_message(self, message) -> None:
        """
        Обрабатывает одно сообщение из очереди.
        """
        try:
            chat_id = self._extract_chat_id(message)
            processed_image = message.body

            await self._save_image_to_dir(
                processed_image,
                chat_id,
                RESULT_DIR,
            )
            await self._send_image_to_chat(chat_id, processed_image)

            logger.info(f"Изображение успешно отправлено в чат {chat_id}")
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")

    def _extract_chat_id(self, message) -> str:
        """
        Извлекает chat_id из заголовков сообщения.
        """
        chat_id = message.headers.get("chat_id")
        if not chat_id:
            raise ValueError("Отсутствует chat_id в заголовках.")
        return chat_id

    async def _send_image_to_chat(self, chat_id: str, image: bytes) -> None:
        """
        Отправляет изображение в чат.
        """
        image_file = BufferedInputFile(image, filename="processed_image.jpg")
        await self.bot.send_photo(
            chat_id=chat_id,
            photo=image_file,
            caption="Вот ваше обработанное изображение!",
        )

    async def process_result(self) -> None:
        """
        Подписывается на очередь и обрабатывает результаты обработки изображений.
        """
        if not self.channel:
            raise ConnectionError("Канал RabbitMQ не установлен.")

        queue = await self.channel.declare_queue(self.queue_result, durable=True)
        logger.info("Подписан на очередь результатов")

        try:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        await self._process_message(message)
        except Exception as e:
            logger.error(f"Ошибка при обработке очереди: {e}")

    async def start_consuming(self):
        """
        Функция подписки на очередь результатов и запуска обработки сообщений.
        """
        try:
            await self.process_result()
        except Exception as e:
            logger.error(f"Ошибка при подписке на очередь: {e}")
        finally:
            await self.close()
