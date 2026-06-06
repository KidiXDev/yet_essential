from __future__ import annotations

from comfy_api.latest import io

from ..services.llm import LLMClient, make_provider_config
from .common import (
    YELLMConfigValue,
    YELLMMessageValue,
    YELLMProviderValue,
    YELLMPipe,
    make_llm_config_value,
    make_llm_message_value,
    make_llm_pipe_value,
    make_llm_provider_value,
    read_llm_config_value,
    read_llm_message_value,
    read_llm_pipe_value,
    read_llm_provider_value,
)


def _normalize_stop_sequences(stop: str) -> list[str]:
    return [line.strip() for line in str(stop or "").splitlines() if line.strip()]


def _looks_like_url(value: str) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")


def _extract_messages(message_value: dict[str, object] | None) -> list[dict[str, str]]:
    if not isinstance(message_value, dict):
        return []
    messages = message_value.get("messages", [])
    return messages if isinstance(messages, list) else []


def _execute_chat(
    provider_config: dict[str, object],
    messages: list[dict[str, str]],
    generation_config: dict[str, object],
) -> str:
    client = LLMClient(provider_config)
    response = client.chat_completions(
        messages=messages,
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        top_k=generation_config["top_k"],
        max_tokens=generation_config["max_tokens"],
        seed=generation_config["seed"],
        json_mode=generation_config["json_mode"],
        stop=generation_config["stop"] or None,
        presence_penalty=generation_config["presence_penalty"],
        frequency_penalty=generation_config["frequency_penalty"],
    )
    return response["text"]


class YELLMProvider(io.ComfyNode):
    PROVIDERS = ["OpenAI Compatible", "OpenRouter", "NanoGPT"]
    MODEL_PLACEHOLDERS = {
        "",
        "Select a model",
        "Loading models...",
        "No models found",
        "Failed to load models",
        "Enter NanoGPT API key",
    }

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELLMProvider",
            display_name="YE LLM Provider",
            category="yet_essential/llm",
            inputs=[
                io.Combo.Input("provider", options=cls.PROVIDERS, default="OpenAI Compatible"),
                io.String.Input("base_url", default="", multiline=False, optional=True),
                io.String.Input("api_key", default="", multiline=False),
                io.Int.Input("timeout", default=60, min=1, max=600),
                io.String.Input("model", default="", multiline=False, optional=True),
                io.String.Input("custom_model", default="", multiline=False, optional=True),
            ],
            outputs=[YELLMProviderValue.Output(display_name="provider")],
        )

    @classmethod
    def execute(
        cls,
        provider: str = "OpenAI Compatible",
        base_url: str = "",
        api_key: str = "",
        timeout: int = 60,
        model: str = "",
        custom_model: str = "",
    ) -> io.NodeOutput:
        provider_key = str(provider or "").strip().lower()
        resolved_base_url = str(base_url or "").strip()
        resolved_model = str(model or "").strip()
        if resolved_model in cls.MODEL_PLACEHOLDERS:
            resolved_model = ""
        if provider_key == "openai compatible":
            resolved_model = str(custom_model or "").strip()
        elif not resolved_model and resolved_base_url and not _looks_like_url(resolved_base_url):
            # Older workflows saved with hidden widgets removed can shift the remote
            # model value into base_url. Recover that layout transparently.
            resolved_model = resolved_base_url
            resolved_base_url = ""
        config = make_provider_config(
            provider=provider,
            base_url=resolved_base_url,
            api_key=api_key,
            model=resolved_model,
            timeout=timeout,
        )
        return io.NodeOutput(make_llm_provider_value(**config))


class YELLMConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELLMConfig",
            display_name="YE LLM Config",
            category="yet_essential/llm",
            inputs=[
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05, round=0.01),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01),
                io.Int.Input("top_k", default=0, min=0, max=1000),
                io.Int.Input("max_tokens", default=0, min=0, max=262144),
                io.Int.Input("seed", default=-1, min=-1, max=0x7FFFFFFFFFFFFFFF),
                io.Float.Input("presence_penalty", default=0.0, min=-2.0, max=2.0, step=0.05, round=0.01),
                io.Float.Input("frequency_penalty", default=0.0, min=-2.0, max=2.0, step=0.05, round=0.01),
                io.Boolean.Input("json_mode", default=False),
                io.String.Input("stop", default="", multiline=True),
            ],
            outputs=[YELLMConfigValue.Output(display_name="config")],
        )

    @classmethod
    def execute(
        cls,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        seed: int,
        presence_penalty: float,
        frequency_penalty: float,
        json_mode: bool,
        stop: str,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            make_llm_config_value(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                seed=seed,
                json_mode=bool(json_mode),
                stop=_normalize_stop_sequences(stop),
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
        )


class YELLMPipeline(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELLMPipeline",
            display_name="YE LLM Pipeline",
            category="yet_essential/llm",
            inputs=[
                YELLMProviderValue.Input("provider"),
                YELLMConfigValue.Input("config", optional=True),
                YELLMMessageValue.Input("llm_message", optional=True),
            ],
            outputs=[YELLMPipe.Output(display_name="llm_pipe")],
        )

    @classmethod
    def execute(
        cls,
        provider: dict,
        config: dict | None = None,
        llm_message: dict | None = None,
    ) -> io.NodeOutput:
        provider_config = read_llm_provider_value(provider, "YELLMPipeline", "provider")
        generation_config = read_llm_config_value(config, "YELLMPipeline", "config")
        message_value = read_llm_message_value(llm_message, "YELLMPipeline", "llm_message")
        return io.NodeOutput(
            make_llm_pipe_value(
                provider=provider_config["provider"],
                base_url=provider_config["base_url"],
                api_key=provider_config["api_key"],
                model=provider_config["model"],
                timeout=provider_config["timeout"],
                config=generation_config,
                messages=_extract_messages(message_value),
            )
        )


class YELLMMessage(io.ComfyNode):
    ROLES = ["system", "user", "assistant"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELLMMessage",
            display_name="YE LLM Message",
            category="yet_essential/llm",
            inputs=[
                io.Combo.Input("role", options=cls.ROLES, default="user"),
                io.String.Input("content", default="", multiline=True, dynamic_prompts=True),
            ],
            outputs=[YELLMMessageValue.Output(display_name="llm_message")],
        )

    @classmethod
    def execute(cls, role: str, content: str) -> io.NodeOutput:
        message = make_llm_message_value(role=role, content=content)
        if not _extract_messages(message):
            raise RuntimeError("YELLMMessage: content cannot be empty.")
        return io.NodeOutput(message)


class YELLMCombineMessages(io.ComfyNode):
    MESSAGE_TEMPLATE = io.Autogrow.TemplatePrefix(
        input=YELLMMessageValue.Input("llm_message", optional=True),
        prefix="llm_message_",
        min=1,
        max=32,
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELLMCombineMessages",
            display_name="YE LLM Combine Messages",
            category="yet_essential/llm",
            inputs=[io.Autogrow.Input("llm_messages", template=cls.MESSAGE_TEMPLATE)],
            outputs=[YELLMMessageValue.Output(display_name="llm_message")],
        )

    @classmethod
    def execute(cls, llm_messages: dict | None = None) -> io.NodeOutput:
        combined: list[dict[str, str]] = []
        for key in sorted((llm_messages or {}).keys()):
            message_value = read_llm_message_value(
                llm_messages.get(key),
                "YELLMCombineMessages",
                key,
            )
            combined.extend(_extract_messages(message_value))
        return io.NodeOutput(make_llm_message_value(messages=combined))


class YELLMChatMessage(io.ComfyNode):
    ROLES = ["system", "user", "assistant"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELLMChatMessage",
            display_name="YE LLM Chat Message",
            category="yet_essential/llm",
            inputs=[
                io.Combo.Input("role", options=cls.ROLES, default="user"),
                io.String.Input("content", default="", multiline=True, dynamic_prompts=True),
            ],
            outputs=[YELLMMessageValue.Output(display_name="llm_message")],
        )

    @classmethod
    def execute(cls, role: str, content: str) -> io.NodeOutput:
        message = make_llm_message_value(role=role, content=content)
        if not _extract_messages(message):
            raise RuntimeError("YELLMChatMessage: content cannot be empty.")
        return io.NodeOutput(message)


class YECallLLM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YECallLLM",
            display_name="YE Call LLM",
            category="yet_essential/llm",
            inputs=[
                YELLMProviderValue.Input("provider"),
                YELLMConfigValue.Input("config", optional=True),
                io.String.Input("system_prompt", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(
        cls,
        provider: dict,
        config: dict | None = None,
        system_prompt: str = "",
        prompt: str = "",
    ) -> io.NodeOutput:
        provider_config = read_llm_provider_value(provider, "YECallLLM", "provider")
        generation_config = read_llm_config_value(config, "YECallLLM", "config")
        messages: list[dict[str, str]] = []
        system_text = str(system_prompt or "").strip()
        prompt_text = str(prompt or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        if prompt_text:
            messages.append({"role": "user", "content": prompt_text})
        if not messages:
            raise RuntimeError("YECallLLM: provide a system prompt or prompt.")
        return io.NodeOutput(_execute_chat(provider_config, messages, generation_config))


class YECallLLMAdvance(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YECallLLMAdvance",
            display_name="YE Call LLM (Advance)",
            category="yet_essential/llm",
            inputs=[YELLMPipe.Input("llm_pipe")],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, llm_pipe: dict) -> io.NodeOutput:
        pipe = read_llm_pipe_value(llm_pipe, "YECallLLMAdvance", "llm_pipe")
        generation_config = read_llm_config_value(pipe.get("config"), "YECallLLMAdvance", "llm_pipe")
        messages = list(pipe.get("messages", []))
        if not messages:
            raise RuntimeError("YECallLLMAdvance: pipeline has no messages.")
        return io.NodeOutput(_execute_chat(pipe, messages, generation_config))


NODE_LIST = [
    YELLMProvider,
    YELLMConfig,
    YELLMPipeline,
    YELLMMessage,
    YELLMCombineMessages,
    YELLMChatMessage,
    YECallLLM,
    YECallLLMAdvance,
]
