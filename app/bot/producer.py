import aio_pika
from typing import Optional
import logging
from config import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_VHOST,
    QUEUE_PROCESS_IMAGE
)

logger = logging.getLogger(__name__)

class ImageProducer:
    def __init__(self):
        self.connection: Optional[aio_pika.Connection] = None
        self.channel = None
        self.queue = QUEUE_PROCESS_IMAGE

    async def connect(self) -> None:
        try:
            self.connection = await aio_pika.connect_robust(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                login=RABBITMQ_USER,
                password=RABBITMQ_PASSWORD,
                virtualhost=RABBITMQ_VHOST
            )
            self.channel = await self.connection.channel()
            self.queue = await self.channel.declare_queue(self.queue)
        except Exception as e:
            logger.error(f"Ошибка при подключении к RabbitMQ: {e}")
            raise e
    
    
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
                    body=str(message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT  # Сообщения сохраняются при перезапуске
                ),
                routing_key=QUEUE_PROCESS_IMAGE
            )
            
            logger.info(f"Изображение {image_path} отправлено в очередь для пользователя {user_id}")
            
        except Exception as e:
            logger.error(f"Ошибка отправки изображения в очередь: {e}")
            raise e
        
    async def close(self) -> None:
        """Закрытие соединения с RabbitMQ"""
        try:
            if self.connection:
                await self.connection.close()
                logger.info("Соединение с RabbitMQ закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения с RabbitMQ: {e}")

