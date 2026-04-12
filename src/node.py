from __future__ import annotations
import os
import uuid
from typing import Any

import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

from .core import BASE_DIR, SETTINGS_PATH, Settings, TagAutocompleteIndex, slerp_noise, MODEL_PREVIEW_MANAGER, SETTINGS, TAG_INDEX


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


class YEClipTextEncodePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "yet_essential.autocomplete": True,
                        "default": "",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "yet_essential/prompt"

    def encode(self, clip, prompt):
        if clip is None:
            raise RuntimeError(
                "YEClipTextEncodePrompt: clip input is invalid (None). "
                "Ensure your checkpoint/model loader outputs a valid CLIP."
            )
        tokens = clip.tokenize(prompt)
        return (clip.encode_from_tokens_scheduled(tokens),)


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


class YESeedGenerator:
    MAX_SEED = 0x7FFFFFFFFFFFFFFF

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": cls.MAX_SEED}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate_seed"
    CATEGORY = "yet_essential/utils"

    def generate_seed(self, seed):
        return (int(seed),)


class YEImageComparer:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = f"_temp_{uuid.uuid4().hex[:5]}"
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare_images"
    OUTPUT_NODE = True
    CATEGORY = "yet_essential/utils"

    def _save_temp_images(self, images, filename_prefix):
        filename_prefix = f"{filename_prefix}{self.prefix_append}"
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                compress_level=self.compress_level,
            )

            results.append(
                {
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type,
                }
            )
            counter += 1

        return results

    def compare_images(self, image_a=None, image_b=None, prompt=None, extra_pnginfo=None):
        result = {"ui": {"a_images": [], "b_images": []}}

        if image_a is not None and len(image_a) > 0:
            result["ui"]["a_images"] = self._save_temp_images(image_a, "ye.compare.a")

        if image_b is not None and len(image_b) > 0:
            result["ui"]["b_images"] = self._save_temp_images(image_b, "ye.compare.b")

        return result


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
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
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
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
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
                "lora_name": (folder_paths.get_filename_list("loras"),),
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
                "lora_name": (folder_paths.get_filename_list("loras"),),
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


def _clamp_image(image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(image, 0.0, 1.0)


def _channel_mean(image: torch.Tensor) -> torch.Tensor:
    # ITU-R BT.709 luma coefficients
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device, dtype=image.dtype)
    return (image[..., :3] * weights).sum(dim=-1, keepdim=True)


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(round(sigma * 3.0)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur_bhwc(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image
    b, h, w, c = image.shape
    x = image.permute(0, 3, 1, 2)
    kernel = _gaussian_kernel_1d(sigma, image.device, image.dtype)
    ksize = kernel.shape[0]
    pad = ksize // 2
    kx = kernel.view(1, 1, 1, ksize).repeat(c, 1, 1, 1)
    ky = kernel.view(1, 1, ksize, 1).repeat(c, 1, 1, 1)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x.permute(0, 2, 3, 1).reshape(b, h, w, c)


def _shift_channel_2d(channel: torch.Tensor, shift_x: int, shift_y: int) -> torch.Tensor:
    # channel shape: [B, H, W]
    if shift_x == 0 and shift_y == 0:
        return channel
    b, h, w = channel.shape
    pad_l = max(shift_x, 0)
    pad_r = max(-shift_x, 0)
    pad_t = max(shift_y, 0)
    pad_b = max(-shift_y, 0)
    x = F.pad(channel.unsqueeze(1), (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    x = x[:, :, pad_b : pad_b + h, pad_r : pad_r + w]
    return x.squeeze(1)


def _apply_adjust_stage(image: torch.Tensor, brightness: float, contrast: float, saturation: float, sharpness: float) -> torch.Tensor:
    x = image

    if brightness != 0.0:
        x = x + float(brightness)

    if contrast != 1.0:
        x = (x - 0.5) * float(contrast) + 0.5

    if saturation != 1.0 and x.shape[-1] >= 3:
        luma = _channel_mean(x)
        x_rgb = luma + (x[..., :3] - luma) * float(saturation)
        x = torch.cat([x_rgb, x[..., 3:]], dim=-1) if x.shape[-1] > 3 else x_rgb

    if sharpness > 0.0:
        sigma = 0.8
        amount = float(sharpness)
        blurred = _gaussian_blur_bhwc(x, sigma=sigma)
        x = x + amount * (x - blurred)

    return _clamp_image(x)


def _apply_style_stage(
    image: torch.Tensor,
    vignette_strength: float,
    vignette_softness: float,
    film_grain: float,
    grain_seed: int,
    chromatic_aberration: float,
    ca_angle: float,
    bloom_strength: float,
    bloom_radius: float,
    bloom_threshold: float,
) -> torch.Tensor:
    x = image
    b, h, w, c = x.shape

    if vignette_strength > 0.0:
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, w)
        rr = torch.sqrt(xx * xx + yy * yy) / 1.41421356237
        power = 0.5 + (1.0 - float(vignette_softness)) * 2.5
        vig = 1.0 - float(vignette_strength) * torch.pow(torch.clamp(rr, 0.0, 1.0), power)
        vig = torch.clamp(vig, 0.0, 1.0).view(1, h, w, 1)
        x = x * vig

    if film_grain > 0.0:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(grain_seed) & 0x7FFFFFFF)
        noise = torch.randn((b, h, w, 1), generator=g, device="cpu", dtype=x.dtype).to(x.device)
        x = x + noise * (0.12 * float(film_grain))

    if chromatic_aberration > 0.0 and c >= 3:
        radians = float(ca_angle) * 0.01745329252
        shift_x = int(round(float(chromatic_aberration) * torch.cos(torch.tensor(radians)).item()))
        shift_y = int(round(float(chromatic_aberration) * torch.sin(torch.tensor(radians)).item()))
        r = _shift_channel_2d(x[..., 0], shift_x, shift_y)
        gch = x[..., 1]
        bch = _shift_channel_2d(x[..., 2], -shift_x, -shift_y)
        x_rgb = torch.stack([r, gch, bch], dim=-1)
        x = torch.cat([x_rgb, x[..., 3:]], dim=-1) if c > 3 else x_rgb

    if bloom_strength > 0.0:
        threshold = float(max(0.0, min(1.0, bloom_threshold)))
        radius = float(max(0.0, bloom_radius))
        if radius > 0.0:
            luma = _channel_mean(x)
            highlights = torch.clamp((luma - threshold) / max(1e-5, 1.0 - threshold), 0.0, 1.0)
            glow_source = x[..., :3] * highlights
            glow = _gaussian_blur_bhwc(glow_source, sigma=radius)
            x_rgb = x[..., :3] + glow * float(bloom_strength)
            x = torch.cat([x_rgb, x[..., 3:]], dim=-1) if c > 3 else x_rgb

    return _clamp_image(x)


def _normalize_pipeline(pipeline: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(pipeline, dict):
        return {"stages": []}
    stages = pipeline.get("stages")
    if not isinstance(stages, list):
        stages = []
    return {"stages": stages}


class YEPostFXAddAdjustStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "pipeline": ("YE_POSTFX_PIPE",),
            },
        }

    RETURN_TYPES = ("YE_POSTFX_PIPE",)
    FUNCTION = "add_stage"
    CATEGORY = "yet_essential/postfx"

    def add_stage(self, enabled, brightness, contrast, saturation, sharpness, pipeline=None):
        out = _normalize_pipeline(pipeline)
        stages = list(out["stages"])
        if enabled:
            stages.append(
                {
                    "kind": "adjust",
                    "brightness": float(brightness),
                    "contrast": float(contrast),
                    "saturation": float(saturation),
                    "sharpness": float(sharpness),
                }
            )
        return ({"stages": stages},)


class YEPostFXAddStyleStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "vignette_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vignette_softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "film_grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "chromatic_aberration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.1}),
                "ca_angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "bloom_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "bloom_radius": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 12.0, "step": 0.1}),
                "bloom_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "pipeline": ("YE_POSTFX_PIPE",),
            },
        }

    RETURN_TYPES = ("YE_POSTFX_PIPE",)
    FUNCTION = "add_stage"
    CATEGORY = "yet_essential/postfx"

    def add_stage(
        self,
        enabled,
        vignette_strength,
        vignette_softness,
        film_grain,
        grain_seed,
        chromatic_aberration,
        ca_angle,
        bloom_strength,
        bloom_radius,
        bloom_threshold,
        pipeline=None,
    ):
        out = _normalize_pipeline(pipeline)
        stages = list(out["stages"])
        if enabled:
            stages.append(
                {
                    "kind": "style",
                    "vignette_strength": float(vignette_strength),
                    "vignette_softness": float(vignette_softness),
                    "film_grain": float(film_grain),
                    "grain_seed": int(grain_seed),
                    "chromatic_aberration": float(chromatic_aberration),
                    "ca_angle": float(ca_angle),
                    "bloom_strength": float(bloom_strength),
                    "bloom_radius": float(bloom_radius),
                    "bloom_threshold": float(bloom_threshold),
                }
            )
        return ({"stages": stages},)


class YEPostFXMergePipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline_a": ("YE_POSTFX_PIPE",),
                "pipeline_b": ("YE_POSTFX_PIPE",),
            }
        }

    RETURN_TYPES = ("YE_POSTFX_PIPE",)
    FUNCTION = "merge"
    CATEGORY = "yet_essential/postfx"

    def merge(self, pipeline_a, pipeline_b):
        a = _normalize_pipeline(pipeline_a)["stages"]
        b = _normalize_pipeline(pipeline_b)["stages"]
        return ({"stages": list(a) + list(b)},)


class YEPostFXApplyPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pipeline": ("YE_POSTFX_PIPE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_pipeline"
    CATEGORY = "yet_essential/postfx"

    def apply_pipeline(self, image, pipeline):
        out = image
        stages = _normalize_pipeline(pipeline)["stages"]
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            kind = stage.get("kind")
            if kind == "adjust":
                out = _apply_adjust_stage(
                    out,
                    brightness=float(stage.get("brightness", 0.0)),
                    contrast=float(stage.get("contrast", 1.0)),
                    saturation=float(stage.get("saturation", 1.0)),
                    sharpness=float(stage.get("sharpness", 0.0)),
                )
            elif kind == "style":
                out = _apply_style_stage(
                    out,
                    vignette_strength=float(stage.get("vignette_strength", 0.0)),
                    vignette_softness=float(stage.get("vignette_softness", 0.5)),
                    film_grain=float(stage.get("film_grain", 0.0)),
                    grain_seed=int(stage.get("grain_seed", 0)),
                    chromatic_aberration=float(stage.get("chromatic_aberration", 0.0)),
                    ca_angle=float(stage.get("ca_angle", 0.0)),
                    bloom_strength=float(stage.get("bloom_strength", 0.0)),
                    bloom_radius=float(stage.get("bloom_radius", 1.5)),
                    bloom_threshold=float(stage.get("bloom_threshold", 0.7)),
                )
        return (_clamp_image(out),)


NODE_CLASS_MAPPINGS = {
    "YEPrompt": YEPrompt,
    "YEClipTextEncodePrompt": YEClipTextEncodePrompt,
    "YEImageUpscale": YEImageUpscale,
    "YEKSampler": YEKSampler,
    "YEEmptyLatentImage": YEEmptyLatentImage,
    "YESeedGenerator": YESeedGenerator,
    "YEImageComparer": YEImageComparer,
    "YELoadCheckpoint": YELoadCheckpoint,
    "YELoadDiffusionModel": YELoadDiffusionModel,
    "YELoadLora": YELoadLora,
    "YELoadLoraModel": YELoadLoraModel,
    "YEPostFXAddAdjustStage": YEPostFXAddAdjustStage,
    "YEPostFXAddStyleStage": YEPostFXAddStyleStage,
    "YEPostFXMergePipeline": YEPostFXMergePipeline,
    "YEPostFXApplyPipeline": YEPostFXApplyPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YEPrompt": "YE Prompt",
    "YEClipTextEncodePrompt": "YE Clip Text Encode (Prompt)",
    "YEImageUpscale": "YE Image Upscale",
    "YEKSampler": "YE KSampler",
    "YEEmptyLatentImage": "YE Empty Latent Image",
    "YESeedGenerator": "YE Seed Generator",
    "YEImageComparer": "YE Image Comparer",
    "YELoadCheckpoint": "YE Load Checkpoint",
    "YELoadDiffusionModel": "YE Load Diffusion Model",
    "YELoadLora": "YE Load LoRA",
    "YELoadLoraModel": "YE Load LoRA (Model Only)",
    "YEPostFXAddAdjustStage": "YE PostFX - Add Adjust Stage",
    "YEPostFXAddStyleStage": "YE PostFX - Add Style Stage",
    "YEPostFXMergePipeline": "YE PostFX - Merge Pipeline",
    "YEPostFXApplyPipeline": "YE PostFX - Apply Pipeline",
}
