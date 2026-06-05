from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
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

from .core import BASE_DIR, slerp_noise


YEPostFXPipe = io.Custom("YE_POSTFX_PIPE")
YEPromptValue = io.Custom("YE_PROMPT_VALUE")


def _prompt_input() -> io.String.Input:
    return io.String.Input(
        "prompt",
        multiline=True,
        dynamic_prompts=True,
        default="",
        extra_dict={"yet_essential.autocomplete": True},
    )


def _format_prompt_text(prompt: str) -> str:
    return ", ".join([part.strip() for part in prompt.split(",") if part.strip()]).strip()


def _normalize_prompt_part(prompt_part: str) -> str:
    return " ".join(prompt_part.lower().split())


def _remove_negative_overlap(positive_prompt: str, negative_prompt: str) -> str:
    positive_parts = [part.strip() for part in positive_prompt.split(",") if part.strip()]
    negative_parts = [part.strip() for part in negative_prompt.split(",") if part.strip()]
    positive_keys = {_normalize_prompt_part(part) for part in positive_parts}
    filtered_negative_parts = [
        part for part in negative_parts if _normalize_prompt_part(part) not in positive_keys
    ]
    return ", ".join(filtered_negative_parts)


def _make_prompt_value(prompt: str) -> dict[str, str]:
    return {"text": prompt}


def _read_prompt_value(prompt_value: Any, node_name: str, input_name: str) -> str:
    if isinstance(prompt_value, dict):
        text = prompt_value.get("text")
        if isinstance(text, str):
            return text
    raise RuntimeError(
        f"{node_name}: invalid '{input_name}' input. Connect it from YE Prompt output."
    )


class YEPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEPrompt",
            display_name="YE Prompt",
            category="yet_essential/prompt",
            inputs=[_prompt_input()],
            outputs=[
                io.String.Output(display_name="prompt"),
                YEPromptValue.Output(display_name="prompt_value"),
            ],
        )

    @classmethod
    def execute(cls, prompt: str) -> io.NodeOutput:
        prompt = _format_prompt_text(prompt)
        return io.NodeOutput(prompt, _make_prompt_value(prompt))


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
            prompt = _format_prompt_text(prompt)

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
    def execute(
        cls,
        positive: Any,
        negative: Any,
    ) -> io.NodeOutput:
        positive_prompt = _read_prompt_value(positive, "YEPromptUtil", "positive")
        negative_prompt = _read_prompt_value(negative, "YEPromptUtil", "negative")
        positive_prompt = _format_prompt_text(positive_prompt)
        negative_prompt = _format_prompt_text(negative_prompt)
        negative_prompt = _remove_negative_overlap(positive_prompt, negative_prompt)
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
    def execute(
        cls,
        clip,
        positive: Any,
        negative: Any,
        format_prompt: bool,
    ) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError(
                "YEClipTextUtil: clip input is invalid (None). "
                "Ensure your checkpoint/model loader outputs a valid CLIP."
            )

        positive_prompt = _read_prompt_value(positive, "YEClipTextUtil", "positive")
        negative_prompt = _read_prompt_value(negative, "YEClipTextUtil", "negative")
        if format_prompt:
            positive_prompt = _format_prompt_text(positive_prompt)
            negative_prompt = _format_prompt_text(negative_prompt)
        negative_prompt = _remove_negative_overlap(positive_prompt, negative_prompt)

        positive_tokens = clip.tokenize(positive_prompt)
        negative_tokens = clip.tokenize(negative_prompt)

        positive_conditioning = clip.encode_from_tokens_scheduled(positive_tokens)
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
        return io.NodeOutput(positive_conditioning, negative_conditioning)


def _load_upscale_model_descriptor(upscale_model: str) -> ImageModelDescriptor:
    model_path = folder_paths.get_full_path_or_raise("upscale_models", upscale_model)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    upscale_model_obj = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(upscale_model_obj, ImageModelDescriptor):
        raise RuntimeError("Upscale model must be a single-image upscaler.")
    return upscale_model_obj


def _run_upscale_model(image: torch.Tensor, upscale_model_obj: ImageModelDescriptor) -> torch.Tensor:
    device = model_management.get_torch_device()
    in_img = image.movedim(-1, -3).to(device)
    _, _, source_h, source_w = in_img.shape

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
                    source_w,
                    source_h,
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

    if out_img is None:
        raise RuntimeError("Upscale model returned no output.")
    return out_img.movedim(-3, -1).cpu()


def _resize_image_bhwc(image: torch.Tensor, width: int, height: int, method: str, crop: str) -> torch.Tensor:
    scaled = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, method, crop)
    return scaled.movedim(1, -1)


def _upscale_image_to_resolution(
    image: torch.Tensor,
    upscale_model: str,
    dest_w: int,
    dest_h: int,
) -> torch.Tensor:
    upscale_model_obj = _load_upscale_model_descriptor(upscale_model)
    out_img = _run_upscale_model(image, upscale_model_obj)
    if out_img.shape[1] != dest_h or out_img.shape[2] != dest_w:
        out_img = _resize_image_bhwc(out_img, dest_w, dest_h, "lanczos", "disabled")
    return torch.clamp(out_img, 0.0, 1.0)


def _upscale_image_by_factor(image: torch.Tensor, upscale_model: str, upscale_by: float) -> torch.Tensor:
    _, height, width, _ = image.shape
    dest_w = max(8, int((width * float(upscale_by)) // 8 * 8))
    dest_h = max(8, int((height * float(upscale_by)) // 8 * 8))
    return _upscale_image_to_resolution(image, upscale_model, dest_w, dest_h)


def _parse_hex_color(color_hex: str) -> tuple[float, float, float]:
    text = str(color_hex or "").strip().lower()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise RuntimeError("YEImageResize: pad_color must be a HEX color like #000000 or #ffffff.")
    try:
        r = int(text[0:2], 16) / 255.0
        g = int(text[2:4], 16) / 255.0
        b = int(text[4:6], 16) / 255.0
    except ValueError as err:
        raise RuntimeError("YEImageResize: invalid pad_color HEX string.") from err
    return r, g, b


def _ensure_mask_bhw(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
        else:
            mask = mask.mean(dim=-1)
    elif mask.ndim != 3:
        raise RuntimeError("YEMaskUtility: unsupported mask shape.")
    return torch.clamp(mask.float(), 0.0, 1.0)


def _match_histogram_channel(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    source_flat = source.reshape(-1)
    ref_flat = reference.reshape(-1)
    if source_flat.numel() == 0 or ref_flat.numel() == 0:
        return source

    source_sorted, source_indices = torch.sort(source_flat)
    ref_sorted = torch.sort(ref_flat).values
    if ref_sorted.numel() != source_sorted.numel():
        ref_pos = torch.linspace(0, ref_sorted.numel() - 1, source_sorted.numel(), device=source.device)
        low = torch.floor(ref_pos).long()
        high = torch.ceil(ref_pos).long()
        weight = (ref_pos - low).to(source.dtype)
        ref_sorted = ref_sorted[low] * (1.0 - weight) + ref_sorted[high] * weight

    matched = torch.empty_like(source_flat)
    matched[source_indices] = ref_sorted
    return matched.reshape_as(source)


def _histogram_match_rgb(image_rgb: torch.Tensor, reference_rgb: torch.Tensor) -> torch.Tensor:
    out = image_rgb.clone()
    ref_batch = reference_rgb.shape[0]
    for batch_idx in range(image_rgb.shape[0]):
        ref_idx = min(batch_idx, ref_batch - 1)
        for channel in range(3):
            out[batch_idx, :, :, channel] = _match_histogram_channel(
                image_rgb[batch_idx, :, :, channel],
                reference_rgb[ref_idx, :, :, channel],
            )
    return out


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(torch.clamp(x, min=0.0), 1.0 / 2.4) - 0.055)


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    rgb_lin = _srgb_to_linear(torch.clamp(rgb, 0.0, 1.0))
    matrix = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.matmul(rgb_lin, matrix.T)
    white = rgb.new_tensor([0.95047, 1.0, 1.08883])
    xyz = xyz / white

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    f_xyz = torch.where(
        xyz > epsilon,
        torch.pow(torch.clamp(xyz, min=0.0), 1.0 / 3.0),
        (kappa * xyz + 16.0) / 116.0,
    )

    l = 116.0 * f_xyz[..., 1] - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return torch.stack([l, a, b], dim=-1)


def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + (lab[..., 1] / 500.0)
    fz = fy - (lab[..., 2] / 200.0)

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def _f_inv(t: torch.Tensor) -> torch.Tensor:
        cube = t * t * t
        return torch.where(cube > epsilon, cube, (116.0 * t - 16.0) / kappa)

    xyz = torch.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], dim=-1)
    white = lab.new_tensor([0.95047, 1.0, 1.08883])
    xyz = xyz * white

    matrix = lab.new_tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb_lin = torch.matmul(xyz, matrix.T)
    rgb = _linear_to_srgb(torch.clamp(rgb_lin, min=0.0))
    return torch.clamp(rgb, 0.0, 1.0)


def _reinhard_color_transfer(image_rgb: torch.Tensor, reference_rgb: torch.Tensor) -> torch.Tensor:
    src_lab = _rgb_to_lab(image_rgb)
    ref_lab = _rgb_to_lab(reference_rgb)

    out_lab = src_lab.clone()
    ref_batch = ref_lab.shape[0]
    for batch_idx in range(src_lab.shape[0]):
        ref_idx = min(batch_idx, ref_batch - 1)
        src_pixels = src_lab[batch_idx].reshape(-1, 3)
        ref_pixels = ref_lab[ref_idx].reshape(-1, 3)
        src_mean = src_pixels.mean(dim=0)
        src_std = src_pixels.std(dim=0, unbiased=False).clamp_min(1e-6)
        ref_mean = ref_pixels.mean(dim=0)
        ref_std = ref_pixels.std(dim=0, unbiased=False).clamp_min(1e-6)
        out_lab[batch_idx] = (src_lab[batch_idx] - src_mean) * (ref_std / src_std) + ref_mean

    return _lab_to_rgb(out_lab)


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
        out_img = _upscale_image_by_factor(image, upscale_model, upscale_by)
        return io.NodeOutput(out_img)


class YEImageResize(io.ComfyNode):
    RESIZE_MODES = [
        "Crop to Fit",
        "Pad to Fit",
        "Stretch",
        "Keep Aspect Ratio (Width)",
        "Keep Aspect Ratio (Height)",
    ]
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "bicubic", "area", "lanczos"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEImageResize",
            display_name="YE Image Resize",
            category="yet_essential/image",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=1024, min=8, max=8192, step=1),
                io.Int.Input("height", default=1024, min=8, max=8192, step=1),
                io.Combo.Input("resize_mode", options=cls.RESIZE_MODES, default="Crop to Fit"),
                io.Combo.Input("upscale_method", options=cls.UPSCALE_METHODS, default="lanczos"),
                io.String.Input("pad_color", default="#000000"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: io.Image.Type,
        width: int,
        height: int,
        resize_mode: str,
        upscale_method: str,
        pad_color: str,
    ) -> io.NodeOutput:
        batch, source_h, source_w, channels = image.shape
        target_w = max(8, int(width))
        target_h = max(8, int(height))

        if resize_mode == "Keep Aspect Ratio (Width)":
            out_h = max(1, int(round(source_h * target_w / max(1, source_w))))
            out_img = _resize_image_bhwc(image, target_w, out_h, upscale_method, "disabled")
            out_mask = torch.zeros((batch, out_h, target_w), dtype=out_img.dtype, device=out_img.device)
            return io.NodeOutput(torch.clamp(out_img, 0.0, 1.0), out_mask, target_w, out_h)

        if resize_mode == "Keep Aspect Ratio (Height)":
            out_w = max(1, int(round(source_w * target_h / max(1, source_h))))
            out_img = _resize_image_bhwc(image, out_w, target_h, upscale_method, "disabled")
            out_mask = torch.zeros((batch, target_h, out_w), dtype=out_img.dtype, device=out_img.device)
            return io.NodeOutput(torch.clamp(out_img, 0.0, 1.0), out_mask, out_w, target_h)

        if resize_mode == "Stretch":
            out_img = _resize_image_bhwc(image, target_w, target_h, upscale_method, "disabled")
            out_mask = torch.zeros((batch, target_h, target_w), dtype=out_img.dtype, device=out_img.device)
            return io.NodeOutput(torch.clamp(out_img, 0.0, 1.0), out_mask, target_w, target_h)

        if resize_mode == "Crop to Fit":
            scale = max(target_w / max(1, source_w), target_h / max(1, source_h))
            scaled_w = max(target_w, int(math.ceil(source_w * scale)))
            scaled_h = max(target_h, int(math.ceil(source_h * scale)))
            scaled = _resize_image_bhwc(image, scaled_w, scaled_h, upscale_method, "disabled")
            x0 = max(0, (scaled_w - target_w) // 2)
            y0 = max(0, (scaled_h - target_h) // 2)
            out_img = scaled[:, y0 : y0 + target_h, x0 : x0 + target_w, :]
            out_mask = torch.zeros((batch, target_h, target_w), dtype=out_img.dtype, device=out_img.device)
            return io.NodeOutput(torch.clamp(out_img, 0.0, 1.0), out_mask, target_w, target_h)

        # Pad to Fit
        scale = min(target_w / max(1, source_w), target_h / max(1, source_h))
        scaled_w = min(target_w, max(1, int(round(source_w * scale))))
        scaled_h = min(target_h, max(1, int(round(source_h * scale))))
        scaled = _resize_image_bhwc(image, scaled_w, scaled_h, upscale_method, "disabled")
        color_r, color_g, color_b = _parse_hex_color(pad_color)

        color_values = torch.ones((channels,), dtype=scaled.dtype, device=scaled.device)
        if channels >= 1:
            color_values[0] = color_r
        if channels >= 2:
            color_values[1] = color_g
        if channels >= 3:
            color_values[2] = color_b

        out_img = color_values.view(1, 1, 1, channels).expand(batch, target_h, target_w, channels).clone()
        out_mask = torch.ones((batch, target_h, target_w), dtype=scaled.dtype, device=scaled.device)
        x0 = max(0, (target_w - scaled_w) // 2)
        y0 = max(0, (target_h - scaled_h) // 2)
        out_img[:, y0 : y0 + scaled_h, x0 : x0 + scaled_w, :] = scaled
        out_mask[:, y0 : y0 + scaled_h, x0 : x0 + scaled_w] = 0.0

        return io.NodeOutput(torch.clamp(out_img, 0.0, 1.0), out_mask, target_w, target_h)


class YEColorMatch(io.ComfyNode):
    METHODS = ["Histogram Matching", "Reinhard (LAb Mean/Std)"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEColorMatch",
            display_name="YE Color Match",
            category="yet_essential/image",
            inputs=[
                io.Image.Input("image"),
                io.Image.Input("reference"),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("method", options=cls.METHODS, default="Histogram Matching"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        image: io.Image.Type,
        reference: io.Image.Type,
        strength: float,
        method: str,
    ) -> io.NodeOutput:
        if image.shape[-1] < 3 or reference.shape[-1] < 3:
            raise RuntimeError("YEColorMatch: image and reference must have at least 3 color channels.")

        strength = float(min(max(strength, 0.0), 1.0))
        if strength <= 0.0:
            return io.NodeOutput(torch.clamp(image, 0.0, 1.0))

        src_rgb = image[..., :3]
        ref_rgb = reference[..., :3]
        if method == "Reinhard (LAb Mean/Std)":
            matched_rgb = _reinhard_color_transfer(src_rgb, ref_rgb)
        else:
            matched_rgb = _histogram_match_rgb(src_rgb, ref_rgb)

        out_rgb = torch.lerp(src_rgb, matched_rgb, strength)
        if image.shape[-1] > 3:
            out = torch.cat([out_rgb, image[..., 3:]], dim=-1)
        else:
            out = out_rgb
        return io.NodeOutput(torch.clamp(out, 0.0, 1.0))


class YEMaskUtility(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEMaskUtility",
            display_name="YE Mask Utility",
            category="yet_essential/image",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input("invert", default=False),
                io.Int.Input("grow_shrink", default=0, min=-256, max=256, step=1),
                io.Float.Input("blur", default=0.0, min=0.0, max=64.0, step=0.1),
                io.Float.Input("threshold", default=0.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(
        cls,
        mask: io.Mask.Type,
        invert: bool,
        grow_shrink: int,
        blur: float,
        threshold: float,
    ) -> io.NodeOutput:
        out = _ensure_mask_bhw(mask)

        if invert:
            out = 1.0 - out

        if grow_shrink != 0:
            radius = abs(int(grow_shrink))
            kernel = radius * 2 + 1
            out_4d = out.unsqueeze(1)
            if grow_shrink > 0:
                out_4d = F.max_pool2d(out_4d, kernel_size=kernel, stride=1, padding=radius)
            else:
                out_4d = 1.0 - F.max_pool2d(1.0 - out_4d, kernel_size=kernel, stride=1, padding=radius)
            out = out_4d[:, 0]

        if blur > 0.0:
            out = _gaussian_blur_bhwc(out.unsqueeze(-1), float(blur)).squeeze(-1)

        threshold = float(min(max(threshold, 0.0), 1.0))
        if threshold > 0.0:
            out = (out > threshold).to(out.dtype)

        return io.NodeOutput(torch.clamp(out, 0.0, 1.0))


class YEHiResFix(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YEHiResFix",
            display_name="YE HiRes Fix",
            category="yet_essential/sampling",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                io.Combo.Input("upscale_model", options=folder_paths.get_filename_list("upscale_models")),
                io.Float.Input("upscale_by", default=1.5, min=1.0, max=4.0, step=0.05),
                io.Float.Input("denoise", default=0.45, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0x7FFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=list(comfy.samplers.KSampler.SAMPLERS)),
                io.Combo.Input("scheduler", options=list(comfy.samplers.KSampler.SCHEDULERS)),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: io.Model.Type,
        positive: io.Conditioning.Type,
        negative: io.Conditioning.Type,
        latent: io.Latent.Type,
        vae: io.Vae.Type,
        upscale_model: str,
        upscale_by: float,
        denoise: float,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
    ) -> io.NodeOutput:
        decoded = vae.decode(latent["samples"])
        if len(decoded.shape) == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])

        upscaled_image = _upscale_image_by_factor(decoded, upscale_model, upscale_by)
        upscaled_latent = {"samples": vae.encode(upscaled_image[:, :, :, :3])}

        latent_samples = comfy.sample.fix_empty_latent_channels(model, upscaled_latent["samples"])
        noise = comfy.sample.prepare_noise(latent_samples, seed, None)
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        refined_samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=float(min(max(denoise, 0.0), 1.0)),
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=None,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        out_latent = {"samples": refined_samples}
        out_image = vae.decode(refined_samples)
        if len(out_image.shape) == 5:
            out_image = out_image.reshape(-1, out_image.shape[-3], out_image.shape[-2], out_image.shape[-1])

        return io.NodeOutput(out_latent, torch.clamp(out_image, 0.0, 1.0))


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


class YELoraStack(io.ComfyNode):
    MAX_SLOTS = 8
    NONE_OPTION = "None"

    @classmethod
    def define_schema(cls) -> io.Schema:
        lora_options = [cls.NONE_OPTION, *folder_paths.get_filename_list("loras")]
        inputs: list[Any] = [
            io.Model.Input("model"),
            io.Clip.Input("clip"),
        ]
        for idx in range(1, cls.MAX_SLOTS + 1):
            inputs.extend(
                [
                    io.Combo.Input(f"lora_name_{idx}", options=lora_options, default=cls.NONE_OPTION),
                    io.Float.Input(f"strength_model_{idx}", default=1.0, min=-20.0, max=20.0, step=0.01),
                    io.Float.Input(f"strength_clip_{idx}", default=1.0, min=-20.0, max=20.0, step=0.01),
                ]
            )

        return io.Schema(
            node_id="YELoraStack",
            display_name="YE LoRA Stack",
            category="yet_essential/loaders",
            inputs=inputs,
            outputs=[io.Model.Output(), io.Clip.Output()],
        )

    @classmethod
    def _slot_lora_name(cls, value: Any) -> str:
        text = str(value or "").strip()
        return "" if text == cls.NONE_OPTION else text

    @classmethod
    def execute(cls, model: io.Model.Type, clip: io.Clip.Type, **kwargs: Any) -> io.NodeOutput:
        model_out = model
        clip_out = clip

        for idx in range(1, cls.MAX_SLOTS + 1):
            lora_name = cls._slot_lora_name(kwargs.get(f"lora_name_{idx}"))
            if not lora_name:
                continue

            strength_model = float(kwargs.get(f"strength_model_{idx}", 1.0))
            strength_clip = float(kwargs.get(f"strength_clip_{idx}", 1.0))
            if strength_model == 0 and strength_clip == 0:
                continue

            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise RuntimeError(f"YELoraStack: LoRA file not found: {lora_name}")
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_out, clip_out = comfy.sd.load_lora_for_models(
                model_out,
                clip_out,
                lora,
                strength_model,
                strength_clip,
            )

        return io.NodeOutput(model_out, clip_out)


class YELoraStackModel(io.ComfyNode):
    MAX_SLOTS = 8
    NONE_OPTION = "None"

    @classmethod
    def define_schema(cls) -> io.Schema:
        lora_options = [cls.NONE_OPTION, *folder_paths.get_filename_list("loras")]
        inputs: list[Any] = [
            io.Model.Input("model"),
        ]
        for idx in range(1, cls.MAX_SLOTS + 1):
            inputs.extend(
                [
                    io.Combo.Input(f"lora_name_{idx}", options=lora_options, default=cls.NONE_OPTION),
                    io.Float.Input(f"strength_model_{idx}", default=1.0, min=-20.0, max=20.0, step=0.01),
                ]
            )

        return io.Schema(
            node_id="YELoraStackModel",
            display_name="YE LoRA Stack (Model Only)",
            category="yet_essential/loaders",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )

    @classmethod
    def _slot_lora_name(cls, value: Any) -> str:
        text = str(value or "").strip()
        return "" if text == cls.NONE_OPTION else text

    @classmethod
    def execute(cls, model: io.Model.Type, **kwargs: Any) -> io.NodeOutput:
        model_out = model

        for idx in range(1, cls.MAX_SLOTS + 1):
            lora_name = cls._slot_lora_name(kwargs.get(f"lora_name_{idx}"))
            if not lora_name:
                continue

            strength_model = float(kwargs.get(f"strength_model_{idx}", 1.0))
            if strength_model == 0:
                continue

            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise RuntimeError(f"YELoraStackModel: LoRA file not found: {lora_name}")
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_out, _ = comfy.sd.load_lora_for_models(
                model_out,
                None,
                lora,
                strength_model,
                0,
            )

        return io.NodeOutput(model_out)


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


@dataclass(slots=True)
class _LUTData:
    tensor: torch.Tensor  # [1, 3, D, H, W]
    domain_min: torch.Tensor  # [3]
    domain_max: torch.Tensor  # [3]


_LUT_CACHE: dict[str, tuple[int, _LUTData]] = {}


def _lut_search_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = [("config/luts", BASE_DIR / "config" / "luts")]
    models_dir = getattr(folder_paths, "models_dir", None)
    if models_dir:
        roots.append(("models/luts", Path(models_dir) / "luts"))
    try:
        for idx, root in enumerate(folder_paths.get_folder_paths("luts")):
            roots.append((f"luts_{idx}", Path(root)))
    except Exception:
        pass
    return roots


def _lut_file_map() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for prefix, root in _lut_search_roots():
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.cube")):
            rel = path.relative_to(root).as_posix()
            label = f"{prefix}/{rel}"
            files[label] = path
    return files


def _lut_options() -> list[str]:
    files = _lut_file_map()
    return sorted(files.keys())


def _resolve_lut_path(lut_name: str) -> Path:
    files = _lut_file_map()
    if lut_name in files:
        return files[lut_name]

    matches = [path for path in files.values() if path.name == lut_name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"YEPostFXAddLUTStage: multiple LUT files found for '{lut_name}'.")
    raise RuntimeError(f"YEPostFXAddLUTStage: LUT file not found: {lut_name}")


def _parse_cube_lut(path: Path) -> _LUTData:
    size = 0
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    values: list[list[float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            head = parts[0].upper()
            if head == "TITLE":
                continue
            if head == "LUT_3D_SIZE" and len(parts) >= 2:
                size = int(parts[1])
                continue
            if head == "DOMAIN_MIN" and len(parts) >= 4:
                domain_min = [float(parts[1]), float(parts[2]), float(parts[3])]
                continue
            if head == "DOMAIN_MAX" and len(parts) >= 4:
                domain_max = [float(parts[1]), float(parts[2]), float(parts[3])]
                continue
            if len(parts) >= 3:
                values.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if size <= 1:
        raise RuntimeError(f"YEPostFXAddLUTStage: invalid LUT_3D_SIZE in {path.name}")

    expected = size * size * size
    if len(values) < expected:
        raise RuntimeError(
            f"YEPostFXAddLUTStage: LUT '{path.name}' has {len(values)} entries, expected {expected}."
        )

    lut_values = torch.tensor(values[:expected], dtype=torch.float32).view(size, size, size, 3)
    lut_tensor = lut_values.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return _LUTData(
        tensor=lut_tensor,
        domain_min=torch.tensor(domain_min, dtype=torch.float32),
        domain_max=torch.tensor(domain_max, dtype=torch.float32),
    )


def _load_lut(path: Path) -> _LUTData:
    cache_key = str(path)
    mtime_ns = path.stat().st_mtime_ns
    cached = _LUT_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]
    lut_data = _parse_cube_lut(path)
    _LUT_CACHE[cache_key] = (mtime_ns, lut_data)
    return lut_data


def _apply_lut_stage(image: torch.Tensor, lut_path: str, strength: float) -> torch.Tensor:
    strength = float(min(max(strength, 0.0), 1.0))
    if strength <= 0.0 or image.shape[-1] < 3:
        return image

    path = Path(lut_path)
    lut_data = _load_lut(path)
    rgb = image[..., :3]

    domain_min = lut_data.domain_min.to(device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, 3)
    domain_max = lut_data.domain_max.to(device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, 3)
    domain_scale = torch.clamp(domain_max - domain_min, min=1e-6)
    rgb_norm = torch.clamp((rgb - domain_min) / domain_scale, 0.0, 1.0)

    grid = rgb_norm.mul(2.0).sub(1.0).unsqueeze(1)
    volume = lut_data.tensor.to(device=rgb.device, dtype=rgb.dtype).expand(rgb.shape[0], -1, -1, -1, -1)
    sampled = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    graded_rgb = sampled.squeeze(2).permute(0, 2, 3, 1)
    out_rgb = torch.lerp(rgb, graded_rgb, strength)
    out = torch.cat([out_rgb, image[..., 3:]], dim=-1) if image.shape[-1] > 3 else out_rgb
    return _clamp_image(out)


class YEPostFXAddLUTStage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = _lut_options()
        if len(options) == 0:
            options = ["<no .cube files found>"]
        return io.Schema(
            node_id="YEPostFXAddLUTStage",
            display_name="YE PostFX - Add LUT Stage",
            category="yet_essential/postfx",
            inputs=[
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input("lut_name", options=options, default=options[0]),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                YEPostFXPipe.Input("pipeline", optional=True),
            ],
            outputs=[YEPostFXPipe.Output(display_name="pipeline")],
        )

    @classmethod
    def execute(
        cls,
        enabled: bool,
        lut_name: str,
        strength: float,
        pipeline: dict[str, Any] | None = None,
    ) -> io.NodeOutput:
        out = _normalize_pipeline(pipeline)
        stages = list(out["stages"])
        if enabled and lut_name != "<no .cube files found>":
            lut_path = _resolve_lut_path(lut_name)
            stages.append(
                {
                    "kind": "lut",
                    "lut_name": str(lut_name),
                    "lut_path": str(lut_path),
                    "strength": float(strength),
                }
            )
        return io.NodeOutput({"stages": stages})


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
            elif kind == "lut":
                lut_path = str(stage.get("lut_path", ""))
                if lut_path:
                    out = _apply_lut_stage(
                        out,
                        lut_path=lut_path,
                        strength=float(stage.get("strength", 1.0)),
                    )
        return io.NodeOutput(_clamp_image(out))


NODE_LIST: list[type[io.ComfyNode]] = [
    YEPrompt,
    YEClipTextEncodePrompt,
    YEPromptUtil,
    YEClipTextUtil,
    YEImageUpscale,
    YEImageResize,
    YEColorMatch,
    YEMaskUtility,
    YEKSampler,
    YEHiResFix,
    YEEmptyLatentImage,
    YESeedGenerator,
    YEImageComparer,
    YELoadCheckpoint,
    YELoadDiffusionModel,
    YELoadLora,
    YELoadLoraModel,
    YELoraStack,
    YELoraStackModel,
    YEPostFXAddAdjustStage,
    YEPostFXAddLUTStage,
    YEPostFXAddStyleStage,
    YEPostFXMergePipeline,
    YEPostFXApplyPipeline,
]


class YetEssentialExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODE_LIST
