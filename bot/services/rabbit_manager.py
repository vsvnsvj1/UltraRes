from aio_pika import connect_robust, Connection, Channel
import aio_pika
import json
import os
import datetime
import logging
from bot.config import get_config

config = get_config()

logger = logging.getLogger(__name__)


class RabbitManager:
    def __init__(self, rabbitmq_dsn: str):
        self.rabbitmq_dsn = rabbitmq_dsn
        self.connection: Connection | None = None
        self.channel: Channel | None = None
        self.response_futures = {}

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

        response_queue = await self.channel.declare_queue(
            "response_queue", durable=True,
        )
        await response_queue.consume(self._on_response)

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
                routing_key=self.queue_process,
            )

            logger.info("Изображение отправлено в очередь")
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
                config.RESULT_DIR,
            )
            await self._send_image_to_chat(chat_id, processed_image)

            logger.info(f"Изображение успешно отправлено в чат {chat_id}")
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")

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
