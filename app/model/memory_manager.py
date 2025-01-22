import torch
import math
import psutil


class MemoryManager:
    def __init__(self, pixel_cost_kb, device):
        """
        Инициализация менеджера памяти.

        :param pixel_cost_kb: Стоимость одного пикселя в килобайтах.
        :param device: Устройство (например, MPS, CUDA, CPU).
        """
        self.pixel_cost_kb = pixel_cost_kb
        self.device = device
        self.memory_limit_kb = self.__get_memory_limit()  # Кэшируем лимит памяти

    def __get_memory_limit(self):
        """
        Получить рекомендуемое ограничение памяти для устройства.

        :return: Рекомендуемое ограничение памяти в килобайтах.
        """
        if self.device == torch.device("mps"):
            if not hasattr(torch.mps, "recommended_max_memory"):
                raise RuntimeError("MPS is not supported or PyTorch is not properly configured.")
            return torch.mps.recommended_max_memory() / 1024  # Перевод в KB

        elif self.device == torch.device("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available.")
            device_index = torch.cuda.current_device()
            free_memory, _ = torch.cuda.mem_get_info(device_index)
            return free_memory / 1024  # Перевод в KB

        elif self.device == torch.device("cpu"):
            memory_info_available_kb = psutil.virtual_memory().available / 1024
            return memory_info_available_kb

        else:
            raise ValueError(f"Device {self.device} not supported for memory management.")

    @staticmethod
    def __calculate_pixel_count(batch, channel, height, width):
        """
        Рассчитать общее количество пикселей для заданных параметров изображения.

        :param batch: Размер батча (количество изображений).
        :param channel: Количество каналов (например, 3 для RGB, 1 для Grayscale).
        :param height: Высота изображения в пикселях.
        :param width: Ширина изображения в пикселях.
        :return: Общее количество пикселей.
        """
        return batch * channel * height * width

    def calculate_tile_count(self, batch, channel, width, height):
        """
        Рассчитать количество тайлов, необходимых для обработки изображения в рамках ограничений памяти.

        :param batch: Размер батча (количество изображений).
        :param channel: Количество каналов.
        :param width: Ширина изображения в пикселях.
        :param height: Высота изображения в пикселях.
        :return: Количество тайлов (int).
        """
        pixel_count = self.__calculate_pixel_count(batch, channel, width, height)
        required_memory_kb = pixel_count * self.pixel_cost_kb
        return math.ceil(required_memory_kb / self.memory_limit_kb)
