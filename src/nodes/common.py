from __future__ import annotations

import re
from typing import Any

from comfy_api.latest import io


YEPostFXPipe = io.Custom("YE_POSTFX_PIPE")
YEPromptValue = io.Custom("YE_PROMPT_VALUE")
YELLMProviderValue = io.Custom("YE_LLM_PROVIDER")
YELLMPipe = io.Custom("YE_LLM_PIPE")
YELLMConfigValue = io.Custom("YE_LLM_CONFIG")
YELLMMessageValue = io.Custom("YE_LLM_MESSAGE")


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


def make_llm_provider_value(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "timeout": int(timeout),
    }


def make_llm_pipe_value(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    config: dict[str, Any] | None = None,
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return make_llm_provider_value(provider, base_url, api_key, model, timeout) | {
        "config": config,
        "messages": messages or [],
    }


def make_llm_config_value(
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    json_mode: bool,
    stop: list[str],
    presence_penalty: float,
    frequency_penalty: float,
) -> dict[str, Any]:
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        "json_mode": bool(json_mode),
        "stop": stop,
        "presence_penalty": float(presence_penalty),
        "frequency_penalty": float(frequency_penalty),
    }


def make_llm_message_value(
    role: str | None = None,
    content: str | None = None,
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if messages is not None:
        return {"messages": messages}
    text = str(content or "").strip()
    if not text:
        return {"messages": []}
    return {"messages": [{"role": str(role or "user").strip(), "content": text}]}


def _read_provider_core(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    provider = str(value.get("provider", "")).strip()
    base_url = str(value.get("base_url", "")).strip()
    api_key = str(value.get("api_key", "")).strip()
    model = str(value.get("model", "")).strip()
    timeout = int(value.get("timeout", 60) or 60)
    if provider and base_url:
        return {
            "provider": provider,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "timeout": timeout,
        }
    return None


def read_llm_provider_value(provider_value: Any, node_name: str, input_name: str) -> dict[str, Any]:
    value = _read_provider_core(provider_value)
    if value is not None:
        return value
    raise RuntimeError(
        f"{node_name}: invalid '{input_name}' input. Connect it from YELLMProvider output."
    )


def read_llm_pipe_value(pipe_value: Any, node_name: str, input_name: str) -> dict[str, Any]:
    value = _read_provider_core(pipe_value)
    if value is not None:
        return value | {
            "config": pipe_value.get("config"),
            "messages": pipe_value.get("messages", []),
        }
    raise RuntimeError(
        f"{node_name}: invalid '{input_name}' input. Connect it from YELLMPipeline output."
    )


def read_llm_config_value(config_value: Any, node_name: str, input_name: str) -> dict[str, Any]:
    if config_value is None:
        return make_llm_config_value(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_tokens=0,
            seed=-1,
            json_mode=False,
            stop=[],
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
    if isinstance(config_value, dict):
        return make_llm_config_value(
            temperature=float(config_value.get("temperature", 1.0) or 1.0),
            top_p=float(config_value.get("top_p", 1.0) or 1.0),
            top_k=int(config_value.get("top_k", 0) or 0),
            max_tokens=int(config_value.get("max_tokens", 0) or 0),
            seed=int(config_value.get("seed", -1) or -1),
            json_mode=bool(config_value.get("json_mode", False)),
            stop=[
                str(item).strip()
                for item in config_value.get("stop", [])
                if str(item).strip()
            ],
            presence_penalty=float(config_value.get("presence_penalty", 0.0) or 0.0),
            frequency_penalty=float(config_value.get("frequency_penalty", 0.0) or 0.0),
        )
    raise RuntimeError(
        f"{node_name}: invalid '{input_name}' input. Connect it from YELLMConfig output."
    )


def read_llm_message_value(message_value: Any, node_name: str, input_name: str) -> dict[str, Any]:
    if message_value is None:
        return {"messages": []}
    if isinstance(message_value, dict):
        messages = message_value.get("messages")
        if isinstance(messages, list):
            normalized: list[dict[str, str]] = []
            for item in messages:
                if not isinstance(item, dict):
                    break
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if not role or not content:
                    break
                normalized.append({"role": role, "content": content})
            else:
                return {"messages": normalized}
    raise RuntimeError(
        f"{node_name}: invalid '{input_name}' input. Connect it from YELLMMessage, YELLMChatMessage, or YELLMCombineMessages."
    )


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
