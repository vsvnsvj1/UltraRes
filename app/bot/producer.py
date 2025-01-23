import aio_pika
import json
from typing import Optional
import logging
import os
from datetime import datetime
from aiogram import Bot
import asyncio
from config import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_VHOST,
    QUEUE_PROCESS_IMAGE,
    QUEUE_RESULT,
    UPLOAD_DIR,
    RESULT_DIR,
    DEBUG
)
from aiogram.types import BufferedInputFile

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
    async def save_image_to_dir( image_bytes: bytes, chat_id: int, dir_name: str, file_name: str = '') -> str:
        """
        Сохраняет изображение в директорию dir_name, если DEBUG включен.

        Args:
            image_bytes (bytes): Байты изображения.
            chat_id (int): ID пользователя.
            dir_name (str): Название директории
            file_name (str): Имя файла.

        Returns:
            str: Путь к сохраненному файлу.
        """
        if not DEBUG:
            return None

        os.makedirs(dir_name, exist_ok=True)
        file_name = f"{chat_id}_{file_name}.jpg"
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
                virtualhost=RABBITMQ_VHOST
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
        
        await self.save_image_to_dir(image_bytes,
                                    chat_id,
                                    UPLOAD_DIR,
                                    datetime.now().strftime("%Y%m%d_%H%M%S")
                                    )
        
        try:
            if not self.channel:
                raise ConnectionError("Канал RabbitMQ не установлен.")
               
            message = {
                "chat_id": chat_id,
                "image_data": image_bytes.hex()
            }
            
            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=self.queue_process
            )
            
            logger.info(f"Изображение отправлено в очередь для чата {chat_id}")
        except Exception as e:
            logger.error(f"Ошибка отправки изображения в очередь: {e}")
            raise e

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
                        try:
                            chat_id = message.headers.get("chat_id")
                            if not chat_id:
                                raise ValueError("Отсутствует chat_id в заголовках.")
                            
                            processed_image = message.body
                            
                            await self.save_image_to_dir(processed_image,
                                                         chat_id,
                                                         RESULT_DIR,
                                                         datetime.now().strftime("%Y%m%d_%H%M%S")
                            )
                            
                            image_file = BufferedInputFile(processed_image, filename="processed_image.jpg")
                            await self.bot.send_photo(chat_id=chat_id, photo=image_file, caption="Вот ваше обработанное изображение!")
                            logger.info(f"Изображение успешно отправлено в чат {chat_id}")
                        except Exception as e:
                            logger.error(f"Ошибка при обработке результата: {e}")
                            
        except asyncio.CancelledError:
            logger.info("Обработка очереди остановлена из-за завершения работы.")
        except Exception as e:
            if isinstance(e, aio_pika.exceptions.ChannelInvalidStateError):
                logger.warning("Канал был закрыт до завершения обработки.")
            else:
                logger.error(f"Ошибка при обработке результата: {e}")
        finally:
            logger.info("Завершение обработки результатов.")

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
    
    