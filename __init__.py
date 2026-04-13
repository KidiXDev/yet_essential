from .src.node import YetEssentialExtension
from .src import routes

WEB_DIRECTORY = "./web"
print(f"### YE Essential: Loading web directory from {WEB_DIRECTORY}")

async def comfy_entrypoint() -> YetEssentialExtension:
    return YetEssentialExtension()

__all__ = [
    "comfy_entrypoint",
    "WEB_DIRECTORY",
]
