from __future__ import annotations

import re
from typing import Any

from comfy_api.latest import io


YEPostFXPipe = io.Custom("YE_POSTFX_PIPE")
YEPromptValue = io.Custom("YE_PROMPT_VALUE")


def prompt_input() -> io.String.Input:
    return io.String.Input(
        "prompt",
        multiline=True,
        dynamic_prompts=True,
        default="",
        extra_dict={"yet_essential.autocomplete": True},
    )


def format_prompt_text(prompt: str) -> str:
    return ", ".join([part.strip() for part in prompt.split(",") if part.strip()]).strip()


def normalize_prompt_part(prompt_part: str) -> str:
    return " ".join(prompt_part.lower().split())


def remove_negative_overlap(positive_prompt: str, negative_prompt: str) -> str:
    positive_parts = [part.strip() for part in positive_prompt.split(",") if part.strip()]
    negative_parts = [part.strip() for part in negative_prompt.split(",") if part.strip()]
    positive_keys = {normalize_prompt_part(part) for part in positive_parts}
    filtered_negative_parts = [
        part for part in negative_parts if normalize_prompt_part(part) not in positive_keys
    ]
    return ", ".join(filtered_negative_parts)


def make_prompt_value(prompt: str) -> dict[str, str]:
    return {"text": prompt}


def read_prompt_value(prompt_value: Any, node_name: str, input_name: str) -> str:
    if isinstance(prompt_value, dict):
        text = prompt_value.get("text")
        if isinstance(text, str):
            return text
    raise RuntimeError(
        f"{node_name}: invalid '{input_name}' input. Connect it from YE Prompt output."
    )


def read_dynamic_node_inputs(kwargs: dict[str, Any]) -> dict[str, Any]:
    prompt = kwargs.get("prompt", {})
    node_id = kwargs.get("unique_id", None)
    if prompt and node_id is not None:
        prompt_key = str(node_id)
        if prompt_key in prompt:
            inputs = prompt[prompt_key].get("inputs", {})
            if isinstance(inputs, dict):
                return inputs
        if node_id in prompt:
            inputs = prompt[node_id].get("inputs", {})
            if isinstance(inputs, dict):
                return inputs
    return {}


def collect_lora_slot_indexes(inputs: dict[str, Any]) -> list[int]:
    indexes: set[int] = set()
    for key in inputs.keys():
        match = re.match(r"^lora_name_(\d+)$", str(key))
        if match:
            indexes.add(int(match.group(1)))
    return sorted(indexes) if indexes else [1]
