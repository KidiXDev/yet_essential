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

from comfy_api.latest import ComfyExtension, io

from .core import slerp_noise


YEPostFXPipe = io.Custom("YE_POSTFX_PIPE")


def _prompt_input() -> io.String.Input:
    return io.String.Input(
        "prompt",
        multiline=True,
        dynamic_prompts=True,
        default="",
        extra_dict={"yet_essential.autocomplete": True},
    )


class YEPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEPrompt",
            display_name="YE Prompt",
            category="yet_essential/prompt",
            inputs=[_prompt_input()],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, prompt: str) -> io.NodeOutput:
        return io.NodeOutput(prompt)


class YEClipTextEncodePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEClipTextEncodePrompt",
            display_name="YE Clip Text Encode (Prompt)",
            category="yet_essential/prompt",
            inputs=[
                io.Clip.Input("clip"),
                _prompt_input(),
                io.Boolean.Input("format_prompt", default=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="formatted_prompt"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt: str, format_prompt: bool) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError(
                "YEClipTextEncodePrompt: clip input is invalid (None). "
                "Ensure your checkpoint/model loader outputs a valid CLIP."
            )

        if format_prompt:
            prompt = ", ".join([p.strip() for p in prompt.split(",") if p.strip()]).strip()

        tokens = clip.tokenize(prompt)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens), prompt)


class YEImageUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEImageUpscale",
            display_name="YE Image Upscale",
            category="yet_essential/image",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("upscale_model", options=folder_paths.get_filename_list("upscale_models")),
                io.Float.Input("upscale_by", default=2.0, min=0.1, max=10.0, step=0.1),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image: io.Image.Type, upscale_model: str, upscale_by: float) -> io.NodeOutput:
        _, height, width, _ = image.shape
        dest_w = max(8, int((width * upscale_by) // 8 * 8))
        dest_h = max(8, int((height * upscale_by) // 8 * 8))

        device = model_management.get_torch_device()
        model_path = folder_paths.get_full_path_or_raise("upscale_models", upscale_model)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})

        upscale_model_obj = ModelLoader().load_from_state_dict(sd).eval()
        if not isinstance(upscale_model_obj, ImageModelDescriptor):
            raise RuntimeError("YEImageUpscale: Upscale model must be a single-image upscaler.")

        in_img = image.movedim(-1, -3).to(device)
        memory_required = model_management.module_size(upscale_model_obj.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model_obj.scale, 1.0) * 128.0
        model_management.free_memory(memory_required, device)
        upscale_model_obj.to(device)

        tile = 512
        overlap = 32
        out_img = None
        try:
            while True:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                        width,
                        height,
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                    )
                    pbar = comfy.utils.ProgressBar(steps)
                    out_img = comfy.utils.tiled_scale(
                        in_img,
                        lambda a: upscale_model_obj(a),
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                        upscale_amount=upscale_model_obj.scale,
                        pbar=pbar,
                    )
                    break
                except Exception as err:
                    model_management.raise_non_oom(err)
                    tile //= 2
                    if tile < 128:
                        raise
        finally:
            upscale_model_obj.to("cpu")

        out_img = out_img.movedim(-3, -1).cpu()
        if out_img.shape[1] != dest_h or out_img.shape[2] != dest_w:
            out_img = out_img.movedim(-1, -3)
            out_img = comfy.utils.common_upscale(out_img, dest_w, dest_h, "lanczos", "disabled")
            out_img = out_img.movedim(-3, -1)

        return io.NodeOutput(torch.clamp(out_img, 0.0, 1.0))


class YEEmptyLatentImage(io.ComfyNode):
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
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEEmptyLatentImage",
            display_name="YE Empty Latent Image",
            category="yet_essential/latent",
            inputs=[
                io.Combo.Input("preset", options=list(cls.DIMENSION_PRESETS.keys()), default="Custom"),
                io.Int.Input("width", default=1024, min=16, max=8192, step=8),
                io.Int.Input("height", default=1024, min=16, max=8192, step=8),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, preset: str, width: int, height: int, batch_size: int) -> io.NodeOutput:
        if preset != "Custom":
            width, height = cls.DIMENSION_PRESETS[preset]

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return io.NodeOutput({"samples": latent})


class YESeedGenerator(io.ComfyNode):
    MAX_SEED = 0x7FFFFFFFFFFFFFFF

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YESeedGenerator",
            display_name="YE Seed Generator",
            category="yet_essential/utils",
            inputs=[io.Int.Input("seed", default=0, min=0, max=cls.MAX_SEED)],
            outputs=[io.Int.Output(display_name="seed")],
        )

    @classmethod
    def execute(cls, seed: int) -> io.NodeOutput:
        return io.NodeOutput(int(seed))


class YEImageComparer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEImageComparer",
            display_name="YE Image Comparer",
            category="yet_essential/utils",
            is_output_node=True,
            inputs=[
                io.Image.Input("image_a", optional=True),
                io.Image.Input("image_b", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def _save_temp_images(cls, images: io.Image.Type, filename_prefix: str) -> list[dict[str, str]]:
        output_dir = folder_paths.get_temp_directory()
        type_name = "temp"
        prefix_append = f"_temp_{uuid.uuid4().hex[:5]}"
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            f"{filename_prefix}{prefix_append}",
            output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results: list[dict[str, str]] = []
        for batch_number, image in enumerate(images):
            arr = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), compress_level=1)
            results.append({"filename": file, "subfolder": subfolder, "type": type_name})
            counter += 1

        return results

    @classmethod
    def execute(
        cls,
        image_a: io.Image.Type | None = None,
        image_b: io.Image.Type | None = None,
    ) -> io.NodeOutput:
        ui_payload = {"a_images": [], "b_images": []}

        if image_a is not None and len(image_a) > 0:
            ui_payload["a_images"] = cls._save_temp_images(image_a, "ye.compare.a")

        if image_b is not None and len(image_b) > 0:
            ui_payload["b_images"] = cls._save_temp_images(image_b, "ye.compare.b")

        return io.NodeOutput(ui=ui_payload)


class YEKSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEKSampler",
            display_name="YE KSampler",
            category="yet_essential/sampling",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("seed", default=0, min=0, max=0x7FFFFFFFFFFFFFFF),
                io.Int.Input("variation_seed", default=0, min=0, max=0x7FFFFFFFFFFFFFFF),
                io.Float.Input("variation_strength", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=list(comfy.samplers.KSampler.SAMPLERS)),
                io.Combo.Input("scheduler", options=list(comfy.samplers.KSampler.SCHEDULERS)),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(
        cls,
        model: io.Model.Type,
        seed: int,
        variation_seed: int,
        variation_strength: float,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: io.Conditioning.Type,
        negative: io.Conditioning.Type,
        latent_image: io.Latent.Type,
        denoise: float = 1.0,
    ) -> io.NodeOutput:
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
        return io.NodeOutput(out)


class YELoadCheckpoint(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YELoadCheckpoint",
            display_name="YE Load Checkpoint",
            category="yet_essential/loaders",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
            ],
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
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path)
        return io.NodeOutput(model)


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
    def execute(
        cls,
        model: io.Model.Type,
        clip: io.Clip.Type,
        lora_name: str,
        strength_model: float,
        strength_clip: float,
    ) -> io.NodeOutput:
        if strength_model == 0 and strength_clip == 0:
            return io.NodeOutput(model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
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
    def execute(
        cls,
        model: io.Model.Type,
        lora_name: str,
        strength_model: float,
    ) -> io.NodeOutput:
        if strength_model == 0:
            return io.NodeOutput(model)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return io.NodeOutput(model_lora)


def _clamp_image(image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(image, 0.0, 1.0)


def _channel_mean(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device, dtype=image.dtype)
    return (image[..., :3] * weights).sum(dim=-1, keepdim=True)


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(round(sigma * 3.0)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


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
    if shift_x == 0 and shift_y == 0:
        return channel
    _, h, w = channel.shape
    pad_l = max(shift_x, 0)
    pad_r = max(-shift_x, 0)
    pad_t = max(shift_y, 0)
    pad_b = max(-shift_y, 0)
    x = F.pad(channel.unsqueeze(1), (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    x = x[:, :, pad_b : pad_b + h, pad_r : pad_r + w]
    return x.squeeze(1)


def _apply_adjust_stage(
    image: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
    sharpness: float,
) -> torch.Tensor:
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
        blurred = _gaussian_blur_bhwc(x, sigma=0.8)
        x = x + float(sharpness) * (x - blurred)

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
        x = x * torch.clamp(vig, 0.0, 1.0).view(1, h, w, 1)

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


class YEPostFXAddAdjustStage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEPostFXAddAdjustStage",
            display_name="YE PostFX - Add Adjust Stage",
            category="yet_essential/postfx",
            inputs=[
                io.Boolean.Input("enabled", default=True),
                io.Float.Input("brightness", default=0.0, min=-1.0, max=1.0, step=0.01),
                io.Float.Input("contrast", default=1.0, min=0.0, max=3.0, step=0.01),
                io.Float.Input("saturation", default=1.0, min=0.0, max=3.0, step=0.01),
                io.Float.Input("sharpness", default=0.0, min=0.0, max=3.0, step=0.01),
                YEPostFXPipe.Input("pipeline", optional=True),
            ],
            outputs=[YEPostFXPipe.Output(display_name="pipeline")],
        )

    @classmethod
    def execute(
        cls,
        enabled: bool,
        brightness: float,
        contrast: float,
        saturation: float,
        sharpness: float,
        pipeline: dict[str, Any] | None = None,
    ) -> io.NodeOutput:
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
        return io.NodeOutput({"stages": stages})


class YEPostFXAddStyleStage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEPostFXAddStyleStage",
            display_name="YE PostFX - Add Style Stage",
            category="yet_essential/postfx",
            inputs=[
                io.Boolean.Input("enabled", default=True),
                io.Float.Input("vignette_strength", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("vignette_softness", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Float.Input("film_grain", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("grain_seed", default=0, min=0, max=0x7FFFFFFF),
                io.Float.Input("chromatic_aberration", default=0.0, min=0.0, max=8.0, step=0.1),
                io.Float.Input("ca_angle", default=0.0, min=-180.0, max=180.0, step=1.0),
                io.Float.Input("bloom_strength", default=0.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("bloom_radius", default=1.5, min=0.0, max=12.0, step=0.1),
                io.Float.Input("bloom_threshold", default=0.7, min=0.0, max=1.0, step=0.01),
                YEPostFXPipe.Input("pipeline", optional=True),
            ],
            outputs=[YEPostFXPipe.Output(display_name="pipeline")],
        )

    @classmethod
    def execute(
        cls,
        enabled: bool,
        vignette_strength: float,
        vignette_softness: float,
        film_grain: float,
        grain_seed: int,
        chromatic_aberration: float,
        ca_angle: float,
        bloom_strength: float,
        bloom_radius: float,
        bloom_threshold: float,
        pipeline: dict[str, Any] | None = None,
    ) -> io.NodeOutput:
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
        return io.NodeOutput({"stages": stages})


class YEPostFXMergePipeline(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEPostFXMergePipeline",
            display_name="YE PostFX - Merge Pipeline",
            category="yet_essential/postfx",
            inputs=[
                YEPostFXPipe.Input("pipeline_a"),
                YEPostFXPipe.Input("pipeline_b"),
            ],
            outputs=[YEPostFXPipe.Output(display_name="pipeline")],
        )

    @classmethod
    def execute(cls, pipeline_a: dict[str, Any], pipeline_b: dict[str, Any]) -> io.NodeOutput:
        a = _normalize_pipeline(pipeline_a)["stages"]
        b = _normalize_pipeline(pipeline_b)["stages"]
        return io.NodeOutput({"stages": list(a) + list(b)})


class YEPostFXApplyPipeline(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEPostFXApplyPipeline",
            display_name="YE PostFX - Apply Pipeline",
            category="yet_essential/postfx",
            inputs=[
                io.Image.Input("image"),
                YEPostFXPipe.Input("pipeline"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image: io.Image.Type, pipeline: dict[str, Any]) -> io.NodeOutput:
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
        return io.NodeOutput(_clamp_image(out))


NODE_LIST: list[type[io.ComfyNode]] = [
    YEPrompt,
    YEClipTextEncodePrompt,
    YEImageUpscale,
    YEKSampler,
    YEEmptyLatentImage,
    YESeedGenerator,
    YEImageComparer,
    YELoadCheckpoint,
    YELoadDiffusionModel,
    YELoadLora,
    YELoadLoraModel,
    YEPostFXAddAdjustStage,
    YEPostFXAddStyleStage,
    YEPostFXMergePipeline,
    YEPostFXApplyPipeline,
]


class YetEssentialExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODE_LIST

