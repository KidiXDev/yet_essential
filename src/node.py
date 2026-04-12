from __future__ import annotations

import os
import threading
from typing import Any

import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
import torch
from aiohttp import web
from server import PromptServer
from spandrel import ImageModelDescriptor, ModelLoader

from .core import AUTOCOMPLETE_CSV_PATH, SETTINGS_PATH, Settings, TagAutocompleteIndex, slerp_noise, MODEL_PREVIEW_MANAGER

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


@PromptServer.instance.routes.get("/yet_essential/model/preview")
async def get_model_preview(request: web.Request) -> web.Response:
    folder_type = request.query.get("type", "")
    model_name = request.query.get("name", "")
    
    if not folder_type or not model_name:
        return web.Response(status=400)
        
    preview_path = MODEL_PREVIEW_MANAGER.find_preview(folder_type, model_name)
    if not preview_path or not os.path.exists(preview_path):
        return web.Response(status=404)
        
    return web.FileResponse(preview_path)


@PromptServer.instance.routes.get("/yet_essential/model/list")
async def get_model_list(request: web.Request) -> web.Response:
    folder_type = request.query.get("type", "")
    if not folder_type:
        return web.Response(status=400)
    
    models = MODEL_PREVIEW_MANAGER.list_models_with_previews(folder_type)
    return web.json_response(models)


class YEPrompt:
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


class YEImageUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "yet_essential/image"

    def upscale(self, image, upscale_model, upscale_by):
        batch_size, height, width, _ = image.shape
        dest_w = max(8, int((width * upscale_by) // 8 * 8))
        dest_h = max(8, int((height * upscale_by) // 8 * 8))

        device = model_management.get_torch_device()
        model_path = folder_paths.get_full_path_or_raise("upscale_models", upscale_model)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})

        upscale_model_obj = ModelLoader().load_from_state_dict(sd).eval()
        if not isinstance(upscale_model_obj, ImageModelDescriptor):
            raise Exception("YEImageUpscale: Upscale model must be a single-image upscaler.")

        in_img = image.movedim(-1, -3).to(device)
        memory_required = model_management.module_size(upscale_model_obj.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model_obj.scale, 1.0) * 128.0
        model_management.free_memory(memory_required, device)
        upscale_model_obj.to(device)

        tile = 512
        overlap = 32
        oom = True
        try:
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(width, height, tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(
                        in_img,
                        lambda a: upscale_model_obj(a),
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                        upscale_amount=upscale_model_obj.scale,
                        pbar=pbar,
                    )
                    oom = False
                except Exception as e:
                    model_management.raise_non_oom(e)
                    tile //= 2
                    if tile < 128:
                        raise e
        finally:
            upscale_model_obj.to("cpu")

        s = s.movedim(-3, -1).cpu()
        if s.shape[1] != dest_h or s.shape[2] != dest_w:
            s = s.movedim(-1, -3)
            s = comfy.utils.common_upscale(s, dest_w, dest_h, "lanczos", "disabled")
            s = s.movedim(-3, -1)

        return (torch.clamp(s, 0.0, 1.0),)


class YEEmptyLatentImage:
    DIMENSION_PRESETS = {
        "Custom": (1024, 1024),
        "1024 x 1024 (1:1 Square)": (1024, 1024),
        "1152 x 896 (9:7 Landscape)": (1152, 896),
        "1216 x 832 (19:13 Landscape)": (1216, 832),
        "1344 x 768 (7:4 Landscape)": (1344, 768),
        "1536 x 640 (12:5 Landscape)": (1536, 640),
        "896 x 1152 (7:9 Portrait)": (896, 1152),
        "832 x 1216 (13:19 Portrait)": (832, 1216),
        "768 x 1344 (4:7 Portrait)": (768, 1344),
        "640 x 1536 (5:12 Portrait)": (640, 1536),
        "512 x 512 (SD1.5 Square)": (512, 512),
        "512 x 768 (SD1.5 Portrait)": (512, 768),
        "768 x 512 (SD1.5 Landscape)": (768, 512),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.DIMENSION_PRESETS.keys()), {"default": "Custom"}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "yet_essential/latent"

    def generate(self, preset, width, height, batch_size):
        if preset != "Custom":
            width, height = self.DIMENSION_PRESETS[preset]

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent},)


class YEKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "variation_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "yet_essential/sampling"

    def sample(self, model, seed, variation_seed, variation_strength, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"])
        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        base_noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        
        strength = float(min(max(variation_strength, 0.0), 1.0))
        if strength > 0.0:
            variation_noise = comfy.sample.prepare_noise(latent_samples, variation_seed, batch_inds)
            noise = slerp_noise(base_noise, variation_noise, strength)
        else:
            noise = base_noise
            
        noise_mask = latent_image["noise_mask"] if "noise_mask" in latent_image else None
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        out = latent_image.copy()
        out["samples"] = samples
        return (out,)


class YELoadCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": ("YE_MODEL_SELECT", {"folder": "checkpoints"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "yet_essential/loaders"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]


class YELoadDiffusionModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": ("YE_MODEL_SELECT", {"folder": "diffusion_models"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "yet_essential/loaders"

    def load_unet(self, unet_name):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path)
        return (model,)


class YELoadLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": ("YE_MODEL_SELECT", {"folder": "loras"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "yet_essential/loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class YELoadLoraModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": ("YE_MODEL_SELECT", {"folder": "loras"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model"
    CATEGORY = "yet_essential/loaders"

    def load_lora_model(self, model, lora_name, strength_model):
        if strength_model == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return (model_lora,)


NODE_CLASS_MAPPINGS = {
    "YEPrompt": YEPrompt,
    "YEImageUpscale": YEImageUpscale,
    "YEKSampler": YEKSampler,
    "YEEmptyLatentImage": YEEmptyLatentImage,
    "YELoadCheckpoint": YELoadCheckpoint,
    "YELoadDiffusionModel": YELoadDiffusionModel,
    "YELoadLora": YELoadLora,
    "YELoadLoraModel": YELoadLoraModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YEPrompt": "YE Prompt",
    "YEImageUpscale": "YE Image Upscale",
    "YEKSampler": "YE KSampler",
    "YEEmptyLatentImage": "YE Empty Latent Image",
    "YELoadCheckpoint": "YE Load Checkpoint",
    "YELoadDiffusionModel": "YE Load Diffusion Model",
    "YELoadLora": "YE Load LoRA",
    "YELoadLoraModel": "YE Load LoRA (Model Only)",
}
