def create_json_from_message(chat_id: int, photo_bytes: bytes) -> dict:
    message = {
        "chat_id": chat_id,
        "image_data": photo_bytes.hex(),
    }
    return message


def extract_chat_id(message) -> str:
    """
    Извлекает chat_id из заголовков сообщения.
    """
    chat_id = message.headers.get("chat_id")
    if not chat_id:
        raise ValueError("Отсутствует chat_id в заголовках.")
    return chat_id
