from __future__ import annotations

from pathlib import Path

from .services import (
    ModelPreviewManager,
    Settings,
    TagAutocompleteIndex,
    fetch_model_catalog,
    WildcardIndex,
    prompt_has_wildcards,
    slerp_noise,
)
from .services.wildcards import expand_prompt_wildcards as _expand_prompt_wildcards


BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = BASE_DIR / "config" / "setting.cfg"

SETTINGS = Settings(SETTINGS_PATH)
TAG_INDEX = TagAutocompleteIndex(BASE_DIR / "config" / "tag" / SETTINGS.csv_file)
MODEL_PREVIEW_MANAGER = ModelPreviewManager(BASE_DIR)
WILDCARD_INDEX = WildcardIndex(BASE_DIR / "config" / "wildcards")


def expand_prompt_wildcards(prompt: str) -> str:
    return _expand_prompt_wildcards(WILDCARD_INDEX, prompt)


__all__ = [
    "BASE_DIR",
    "SETTINGS_PATH",
    "SETTINGS",
    "TAG_INDEX",
    "MODEL_PREVIEW_MANAGER",
    "WILDCARD_INDEX",
    "expand_prompt_wildcards",
    "fetch_model_catalog",
    "prompt_has_wildcards",
    "slerp_noise",
]
