from dotenv import load_dotenv
import os

# Загружаем переменные окружения из файла .env
load_dotenv()

# Получаем значение токена из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Проверка наличия токена
if TELEGRAM_BOT_TOKEN is None:
    raise ValueError("Не задан TELEGRAM_BOT_TOKEN в файле .env")

# RabbitMQ
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")

# Очереди
QUEUE_PROCESS_IMAGE = os.getenv("QUEUE_PROCESS_IMAGE", "process_image_queue")
QUEUE_RESULT = os.getenv("QUEUE_RESULT", "result_queue")

# Настройки приложения
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Пути
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
RESULT_DIR = os.getenv("RESULT_DIR", "results")

# Проверка обязательных переменных
if TELEGRAM_BOT_TOKEN is None:
    raise ValueError("Не задан TELEGRAM_BOT_TOKEN в файле .env")

# Создаем директории если их нет
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)