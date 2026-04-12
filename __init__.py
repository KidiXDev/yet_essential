from .src.node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .src import routes

WEB_DIRECTORY = "./web"
print(f"### YE Essential: Loading web directory from {WEB_DIRECTORY}")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
