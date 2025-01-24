from .common_commands import commands_router
from .image_handler import image_router
from .unknown import unknown_router

all_routers = [
    commands_router,
    image_router,
    unknown_router,
]
