from aiogram.types import BufferedInputFile


def create_json_from_message(chat_id: int, photo_bytes: bytes) -> dict:
    message = {
        "chat_id": chat_id,
        "image_data": photo_bytes.hex(),
    }
    return message


def extract_chat_id(self, message) -> str:
    """
    Извлекает chat_id из заголовков сообщения.
    """
    chat_id = message.headers.get("chat_id")
    if not chat_id:
        raise ValueError("Отсутствует chat_id в заголовках.")
    return chat_id


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
