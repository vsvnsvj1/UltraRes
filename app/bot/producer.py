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
        try:
            # Подключаемся к RabbitMQ
            self.connection = await aio_pika.connect_robust(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                login=RABBITMQ_USER,
                password=RABBITMQ_PASSWORD,
                virtualhost=RABBITMQ_VHOST
            )
            self.channel = await self.connection.channel()
            
            # Объявляем очереди
            self.queue_process = await self.channel.declare_queue(
                self.queue_process,
                durable=True
            )
            self.queue_result = await self.channel.declare_queue(
                self.queue_result,
                durable=True
            )

            # Запускаем получение сообщений в отдельной задаче
            self._consuming = True
            self._consume_task = asyncio.create_task(self.start_consuming())
            
            logger.info("Подключение к RabbitMQ установлено")
        except Exception as e:
            logger.error(f"Ошибка при подключении к RabbitMQ: {e}")
            raise e

    async def start_consuming(self):
        try:
            while self._consuming:
                try:
                    async with self.queue_result.iterator() as queue_iter:
                        async for message in queue_iter:
                            if not self._consuming:
                                break
                            await self.process_result(message)
                except aio_pika.exceptions.ChannelClosed:
                    if self._consuming:
                        await asyncio.sleep(1)  # Пауза перед повторным подключением
                        continue
                    break
        except Exception as e:
            logger.error(f"Ошибка в цикле получения сообщений: {e}")
            
    async def close(self) -> None:
        """Корректное закрытие соединения с RabbitMQ"""
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

    
    async def send_image(self, image_path: str, user_id: int) -> None:
        try:
            if not self.channel:
                await self.connect()
                
            message = {
                "image_path": image_path,
                "user_id": user_id
            }
            
            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=self.queue_process
            )
            
            logger.info(f"Изображение {image_path} отправлено в очередь для пользователя {user_id}")
            
        except Exception as e:
            logger.error(f"Ошибка отправки изображения в очередь: {e}")
            raise e

    async def process_result(self, message: aio_pika.IncomingMessage):
        async with await message.process():
            try:
                body = json.loads(message.body.decode())
                user_id = body['user_id']
                result_image_path = body['result_image_path']

                # Отправляем обработанное изображение пользователю
                await self.bot.send_photo(
                    chat_id=user_id,
                    photo=open(result_image_path, 'rb'),
                    caption="Вот ваше обработанное изображение!"
                )
                logger.info(f"Обработанное изображение отправлено пользователю {user_id}")

            except Exception as e:
                logger.error(f"Ошибка при обработке результата: {e}")
                if 'user_id' in locals():
                    await self.bot.send_message(
                        user_id,
                        "Извините, произошла ошибка при отправке обработанного изображения."
                    )

    async def start_consuming(self):
        try:
            # Настраиваем получение сообщений из очереди результатов
            async with self.queue_result.iterator() as queue_iter:
                async for message in queue_iter:
                    await self.process_result(message)
        except Exception as e:
            logger.error(f"Ошибка при получении результатов: {e}")
            