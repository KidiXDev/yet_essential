from __future__ import annotations

from typing import Any

from aiohttp import web
from server import PromptServer

from .core import AUTOCOMPLETE_CSV_PATH, SETTINGS_PATH, Settings, TagAutocompleteIndex

TAG_INDEX = TagAutocompleteIndex(AUTOCOMPLETE_CSV_PATH)
SETTINGS = Settings(SETTINGS_PATH)


@PromptServer.instance.routes.get("/yet_essential/autocomplete/search")
async def search_autocomplete(request: web.Request) -> web.Response:
    query = request.query.get("q", "")
    try:
        requested_limit = int(request.query.get("limit", str(SETTINGS.limit)))
    except ValueError:
        requested_limit = SETTINGS.limit

    limit = min(requested_limit, SETTINGS.limit)
    return web.json_response(
        {
            "query": query,
            "settings": {
                "show_category_id": SETTINGS.show_category_id,
                "show_post_count": SETTINGS.show_post_count,
                "spacing_mode": SETTINGS.spacing_mode,
                "insertion_suffix": SETTINGS.insertion_suffix,
                "escape_parentheses": SETTINGS.escape_parentheses,
            },
            "items": TAG_INDEX.search(
                query=query, 
                limit=limit, 
                algorithm=SETTINGS.algorithm,
                sort_mode=SETTINGS.sort_mode
            ),
        }
    )


class YEPromptAutocomplete:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str, dict[str, Any]]]]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "yet_essential.autocomplete": True,
                        "default": "",
                        "placeholder": "Type a prompt. Autocomplete comes from config/autocomplete.csv",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "output_prompt"
    CATEGORY = "yet_essential/prompt"

    def output_prompt(self, prompt: str) -> tuple[str]:
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "YEPromptAutocomplete": YEPromptAutocomplete,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YEPromptAutocomplete": "YE Prompt Autocomplete",
}
