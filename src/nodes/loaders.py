from __future__ import annotations

from typing import Any

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .common import collect_lora_slot_indexes, read_dynamic_node_inputs


class YELoadCheckpoint(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELoadCheckpoint",
            display_name="YE Load Checkpoint",
            category="yet_essential/loaders",
            inputs=[io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints"))],
            outputs=[io.Model.Output(), io.Clip.Output(), io.Vae.Output()],
        )

    @classmethod
    def execute(cls, ckpt_name: str) -> io.NodeOutput:
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return io.NodeOutput(*out[:3])


class YELoadDiffusionModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELoadDiffusionModel",
            display_name="YE Load Diffusion Model",
            category="yet_essential/loaders",
            inputs=[io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models"))],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name: str) -> io.NodeOutput:
        return io.NodeOutput(comfy.sd.load_diffusion_model(folder_paths.get_full_path("diffusion_models", unet_name)))


class YELoadLora(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELoadLora",
            display_name="YE Load LoRA",
            category="yet_essential/loaders",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Float.Input("strength_clip", default=1.0, min=-20.0, max=20.0, step=0.01),
            ],
            outputs=[io.Model.Output(), io.Clip.Output()],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, clip: io.Clip.Type, lora_name: str, strength_model: float, strength_clip: float) -> io.NodeOutput:
        if strength_model == 0 and strength_clip == 0:
            return io.NodeOutput(model, clip)
        lora = comfy.utils.load_torch_file(folder_paths.get_full_path("loras", lora_name), safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return io.NodeOutput(model_lora, clip_lora)


class YELoadLoraModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELoadLoraModel",
            display_name="YE Load LoRA (Model Only)",
            category="yet_essential/loaders",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-20.0, max=20.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, lora_name: str, strength_model: float) -> io.NodeOutput:
        if strength_model == 0:
            return io.NodeOutput(model)
        lora = comfy.utils.load_torch_file(folder_paths.get_full_path("loras", lora_name), safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return io.NodeOutput(model_lora)


class YELoraStack(io.ComfyNode):
    NONE_OPTION = "None"

    @classmethod
    def define_schema(cls) -> io.Schema:
        lora_options = [cls.NONE_OPTION, *folder_paths.get_filename_list("loras")]
        return io.Schema(
            node_id="YELoraStack",
            display_name="YE LoRA Stack",
            category="yet_essential/loaders",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Boolean.Input("enabled_1", default=True),
                io.Combo.Input("lora_name_1", options=lora_options, default=cls.NONE_OPTION),
                io.Float.Input("strength_model_1", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Float.Input("strength_clip_1", default=1.0, min=-20.0, max=20.0, step=0.01),
            ],
            outputs=[io.Model.Output(), io.Clip.Output()],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def _slot_lora_name(cls, value: Any) -> str:
        text = str(value or "").strip()
        return "" if text == cls.NONE_OPTION else text

    @classmethod
    def execute(cls, model: io.Model.Type, clip: io.Clip.Type, **kwargs: Any) -> io.NodeOutput:
        model_out = model
        clip_out = clip
        dynamic_inputs = read_dynamic_node_inputs(kwargs)
        slot_indexes = collect_lora_slot_indexes(kwargs | dynamic_inputs)
        for idx in slot_indexes:
            enabled = dynamic_inputs.get(f"enabled_{idx}", kwargs.get(f"enabled_{idx}", True))
            if not bool(enabled):
                continue
            lora_name = cls._slot_lora_name(dynamic_inputs.get(f"lora_name_{idx}", kwargs.get(f"lora_name_{idx}")))
            if not lora_name:
                continue
            strength_model = float(dynamic_inputs.get(f"strength_model_{idx}", kwargs.get(f"strength_model_{idx}", 1.0)))
            strength_clip = float(dynamic_inputs.get(f"strength_clip_{idx}", kwargs.get(f"strength_clip_{idx}", 1.0)))
            if strength_model == 0 and strength_clip == 0:
                continue
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise RuntimeError(f"YELoraStack: LoRA file not found: {lora_name}")
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_out, clip_out = comfy.sd.load_lora_for_models(model_out, clip_out, lora, strength_model, strength_clip)
        return io.NodeOutput(model_out, clip_out)


class YELoraStackModel(io.ComfyNode):
    NONE_OPTION = "None"

    @classmethod
    def define_schema(cls) -> io.Schema:
        lora_options = [cls.NONE_OPTION, *folder_paths.get_filename_list("loras")]
        return io.Schema(
            node_id="YELoraStackModel",
            display_name="YE LoRA Stack (Model Only)",
            category="yet_essential/loaders",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled_1", default=True),
                io.Combo.Input("lora_name_1", options=lora_options, default=cls.NONE_OPTION),
                io.Float.Input("strength_model_1", default=1.0, min=-20.0, max=20.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def _slot_lora_name(cls, value: Any) -> str:
        text = str(value or "").strip()
        return "" if text == cls.NONE_OPTION else text

    @classmethod
    def execute(cls, model: io.Model.Type, **kwargs: Any) -> io.NodeOutput:
        model_out = model
        dynamic_inputs = read_dynamic_node_inputs(kwargs)
        slot_indexes = collect_lora_slot_indexes(kwargs | dynamic_inputs)
        for idx in slot_indexes:
            enabled = dynamic_inputs.get(f"enabled_{idx}", kwargs.get(f"enabled_{idx}", True))
            if not bool(enabled):
                continue
            lora_name = cls._slot_lora_name(dynamic_inputs.get(f"lora_name_{idx}", kwargs.get(f"lora_name_{idx}")))
            if not lora_name:
                continue
            strength_model = float(dynamic_inputs.get(f"strength_model_{idx}", kwargs.get(f"strength_model_{idx}", 1.0)))
            if strength_model == 0:
                continue
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise RuntimeError(f"YELoraStackModel: LoRA file not found: {lora_name}")
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_out, _ = comfy.sd.load_lora_for_models(model_out, None, lora, strength_model, 0)
        return io.NodeOutput(model_out)


NODE_LIST = [
    YELoadCheckpoint,
    YELoadDiffusionModel,
    YELoadLora,
    YELoadLoraModel,
    YELoraStack,
    YELoraStackModel,
]
