from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from ..core import expand_prompt_wildcards, prompt_has_wildcards
from .common import (
    YEPromptValue,
    format_prompt_text,
    make_prompt_value,
    prompt_input,
    read_prompt_value,
    remove_negative_overlap,
)


class YEPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEPrompt",
            display_name="YE Prompt",
            category="yet_essential/prompt",
            inputs=[prompt_input()],
            outputs=[
                io.String.Output(display_name="prompt"),
                YEPromptValue.Output(display_name="prompt_value"),
            ],
        )

    @classmethod
    def IS_CHANGED(cls, prompt: str):
        if prompt_has_wildcards(prompt):
            return float("nan")
        return prompt

    @classmethod
    def execute(cls, prompt: str) -> io.NodeOutput:
        prompt = expand_prompt_wildcards(prompt)
        prompt = format_prompt_text(prompt)
        return io.NodeOutput(prompt, make_prompt_value(prompt))


class YEClipTextEncodePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEClipTextEncodePrompt",
            display_name="YE Clip Text Encode (Prompt)",
            category="yet_essential/prompt",
            inputs=[
                io.Clip.Input("clip"),
                prompt_input(),
                io.Boolean.Input("format_prompt", default=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="formatted_prompt"),
            ],
        )

    @classmethod
    def IS_CHANGED(cls, clip, prompt: str, format_prompt: bool):
        if prompt_has_wildcards(prompt):
            return float("nan")
        return (id(clip), prompt, bool(format_prompt))

    @classmethod
    def execute(cls, clip, prompt: str, format_prompt: bool) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError(
                "YEClipTextEncodePrompt: clip input is invalid (None). "
                "Ensure your checkpoint/model loader outputs a valid CLIP."
            )
        prompt = expand_prompt_wildcards(prompt)
        if format_prompt:
            prompt = format_prompt_text(prompt)
        tokens = clip.tokenize(prompt)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens), prompt)


class YEPromptUtil(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEPromptUtil",
            display_name="YE Prompt Util",
            category="yet_essential/prompt",
            inputs=[
                YEPromptValue.Input("positive"),
                YEPromptValue.Input("negative"),
            ],
            outputs=[
                io.String.Output(display_name="positive"),
                io.String.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive: Any, negative: Any) -> io.NodeOutput:
        positive_prompt = read_prompt_value(positive, "YEPromptUtil", "positive")
        negative_prompt = read_prompt_value(negative, "YEPromptUtil", "negative")
        positive_prompt = format_prompt_text(positive_prompt)
        negative_prompt = format_prompt_text(negative_prompt)
        negative_prompt = remove_negative_overlap(positive_prompt, negative_prompt)
        return io.NodeOutput(positive_prompt, negative_prompt)


class YEClipTextUtil(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEClipTextUtil",
            display_name="YE Clip Text Util",
            category="yet_essential/prompt",
            inputs=[
                io.Clip.Input("clip"),
                YEPromptValue.Input("positive"),
                YEPromptValue.Input("negative"),
                io.Boolean.Input("format_prompt", default=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, clip, positive: Any, negative: Any, format_prompt: bool) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError(
                "YEClipTextUtil: clip input is invalid (None). "
                "Ensure your checkpoint/model loader outputs a valid CLIP."
            )
        positive_prompt = read_prompt_value(positive, "YEClipTextUtil", "positive")
        negative_prompt = read_prompt_value(negative, "YEClipTextUtil", "negative")
        if format_prompt:
            positive_prompt = format_prompt_text(positive_prompt)
            negative_prompt = format_prompt_text(negative_prompt)
        negative_prompt = remove_negative_overlap(positive_prompt, negative_prompt)
        positive_tokens = clip.tokenize(positive_prompt)
        negative_tokens = clip.tokenize(negative_prompt)
        return io.NodeOutput(
            clip.encode_from_tokens_scheduled(positive_tokens),
            clip.encode_from_tokens_scheduled(negative_tokens),
        )


NODE_LIST = [YEPrompt, YEClipTextEncodePrompt, YEPromptUtil, YEClipTextUtil]
