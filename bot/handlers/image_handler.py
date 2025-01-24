import logging

from aiogram import F
from aiogram import Router
from aiogram.types import Message, ContentType

logger = logging.getLogger(__name__)

image_router = Router()


@image_router.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: Message):
    """Обработчик входящих фотографий"""
    pass
