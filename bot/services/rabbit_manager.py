import json
import logging
import os
from datetime import datetime

import aio_pika
from aio_pika import Channel, Connection, connect_robust
from aiogram import Bot
from aiogram.types import BufferedInputFile

from bot.config import get_config
from bot.scripts.message_scripts import extract_chat_id

config = get_config()

logger = logging.getLogger(__name__)


class RabbitManager:
    def __init__(self, rabbitmq_dsn: str, bot: Bot):
        self.rabbitmq_dsn = rabbitmq_dsn
        self.connection: Connection | None = None
        self.channel: Channel | None = None
        self.bot = bot

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
        if not config.DEBUG:
            return None

        os.makedirs(dir_name, exist_ok=True)
        file_name = f"{chat_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg"
        file_path = os.path.join(dir_name, file_name)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        logger.debug(f"Изображение сохранено в {file_path}")
        return file_path

    async def connect(self):
        self.connection = await connect_robust(self.rabbitmq_dsn)
        self.channel = await self.connection.channel()

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

    async def send_json_to_queue(self, json_message):
        try:
            if not self.channel:
                raise ConnectionError("Канал RabbitMQ не установлен.")

            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(json_message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=config.QUEUE_PROCESS_IMAGE,
            )

            logger.info("Изображение отправлено в очередь")
        except Exception as e:
            logger.error(f"Ошибка отправки изображения в очередь: {e}")
            raise e

    async def send_image_to_chat(self, chat_id: str, image: bytes) -> None:
        """
        Отправляет изображение в чат.
        """
        image_file = BufferedInputFile(image, filename="processed_image.jpg")
        await self.bot.send_photo(
            chat_id=chat_id,
            photo=image_file,
            caption="Вот ваше обработанное изображение!",
        )

    async def _process_message(self, message) -> None:
        """
        Обрабатывает одно сообщение из очереди.
        """
        try:
            chat_id = extract_chat_id(message)
            processed_image = message.body

            await self._save_image_to_dir(
                processed_image,
                chat_id,
                config.RESULT_DIR,
            )
            await self.send_image_to_chat(chat_id, processed_image)

            logger.info(f"Изображение успешно отправлено в чат {chat_id}")
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")

    async def process_result(self) -> None:
        """
        Подписывается на очередь и обрабатывает результаты обработки изображений.
        """
        if not self.channel:
            raise ConnectionError("Канал RabbitMQ не установлен.")

        queue = await self.channel.declare_queue(config.QUEUE_RESULT, durable=True)
        logger.info("Подписан на очередь результатов")

        try:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        await self._process_message(message)
        except Exception as e:
            logger.error(f"Ошибка при обработке очереди: {e}")
