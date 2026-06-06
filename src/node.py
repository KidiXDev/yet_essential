from __future__ import annotations

from comfy_api.latest import ComfyExtension

from .registry import NODE_LIST


class YetEssentialExtension(ComfyExtension):
    async def get_node_list(self):
        return NODE_LIST
