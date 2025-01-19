import aio_pika
import json
from typing import Optional
import logging
from aiogram import Bot
import asyncio
from config import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_VHOST,
    QUEUE_PROCESS_IMAGE,
    QUEUE_RESULT
)
from io import BytesIO
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

            
        '''
        try:
            # Останавливаем получение сообщений
            self._consuming = False
            
            if self._consume_task:
                try:
                    await asyncio.wait_for(self._consume_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self._consume_task.cancel()
                    
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
                
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                
            logger.info("Соединение с RabbitMQ корректно закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения с RabbitMQ: {e}")
        '''
    
    async def send_image(self, image_bytes: str, chat_id: int) -> None:
        """
        Отправляет байтовое изображение в очередь RabbitMQ.

        Args:
            image_bytes (bytes): Байты изображения.
            chat_id (int): ID чата для отправки результата.
        """
        
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
        async for message in queue:
            async with message.process():
                try:
                    chat_id = message.headers.get("chat_id")
                    if not chat_id:
                        raise ValueError("Отсутствует chat_id в заголовках.")
                    
                    processed_image = message.body
                    image_file = BufferedInputFile(processed_image, filename="processed_image.jpg")
                    await self.bot.send_photo(chat_id=chat_id, photo=image_file, caption="Вот ваше обработанное изображение!")
                    logger.info(f"Изображение успешно отправлено в чат {chat_id}")
                except Exception as e:
                    logger.error(f"Ошибка при обработке результата: {e}")


    async def start_consuming(self):
        """
        Функция подписки на очередь результатов и запуска обработки сообщений.
        """
        try:
            await self.process_result()
        except Exception as e:
            logger.error(f"Ошибка при подписке на очередь: {e}")
            