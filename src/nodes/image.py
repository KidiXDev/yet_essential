from __future__ import annotations

import math

import comfy.model_management as model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
import torch
import torch.nn.functional as F
from spandrel import ImageModelDescriptor, ModelLoader

from comfy_api.latest import io


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
                    source_w, source_h, tile_x=tile, tile_y=tile, overlap=overlap
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
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, method, crop).movedim(1, -1)


def _upscale_image_to_resolution(image: torch.Tensor, upscale_model: str, dest_w: int, dest_h: int) -> torch.Tensor:
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
        return int(text[0:2], 16) / 255.0, int(text[2:4], 16) / 255.0, int(text[4:6], 16) / 255.0
    except ValueError as err:
        raise RuntimeError("YEImageResize: invalid pad_color HEX string.") from err


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
                image_rgb[batch_idx, :, :, channel], reference_rgb[ref_idx, :, :, channel]
            )
    return out


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(torch.clamp(x, min=0.0), 1.0 / 2.4) - 0.055)


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    rgb_lin = _srgb_to_linear(torch.clamp(rgb, 0.0, 1.0))
    matrix = rgb.new_tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = torch.matmul(rgb_lin, matrix.T)
    white = rgb.new_tensor([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f_xyz = torch.where(xyz > epsilon, torch.pow(torch.clamp(xyz, min=0.0), 1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)
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
    xyz = xyz * lab.new_tensor([0.95047, 1.0, 1.08883])
    matrix = lab.new_tensor([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    rgb_lin = torch.matmul(xyz, matrix.T)
    return torch.clamp(_linear_to_srgb(torch.clamp(rgb_lin, min=0.0)), 0.0, 1.0)


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
        return io.Schema(node_id="YEImageUpscale", display_name="YE Image Upscale", category="yet_essential/image", inputs=[io.Image.Input("image"), io.Combo.Input("upscale_model", options=folder_paths.get_filename_list("upscale_models")), io.Float.Input("upscale_by", default=2.0, min=0.1, max=10.0, step=0.1)], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, image: io.Image.Type, upscale_model: str, upscale_by: float) -> io.NodeOutput:
        return io.NodeOutput(_upscale_image_by_factor(image, upscale_model, upscale_by))


class YEImageResize(io.ComfyNode):
    RESIZE_MODES = ["Crop to Fit", "Pad to Fit", "Stretch", "Keep Aspect Ratio (Width)", "Keep Aspect Ratio (Height)"]
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "bicubic", "area", "lanczos"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="YEImageResize", display_name="YE Image Resize", category="yet_essential/image", inputs=[io.Image.Input("image"), io.Int.Input("width", default=1024, min=8, max=8192, step=1), io.Int.Input("height", default=1024, min=8, max=8192, step=1), io.Combo.Input("resize_mode", options=cls.RESIZE_MODES, default="Crop to Fit"), io.Combo.Input("upscale_method", options=cls.UPSCALE_METHODS, default="lanczos"), io.String.Input("pad_color", default="#000000")], outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask"), io.Int.Output(display_name="width"), io.Int.Output(display_name="height")])

    @classmethod
    def execute(cls, image: io.Image.Type, width: int, height: int, resize_mode: str, upscale_method: str, pad_color: str) -> io.NodeOutput:
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
        scale = min(target_w / max(1, source_w), target_h / max(1, source_h))
        scaled_w = min(target_w, max(1, int(round(source_w * scale))))
        scaled_h = min(target_h, max(1, int(round(source_h * scale))))
        scaled = _resize_image_bhwc(image, scaled_w, scaled_h, upscale_method, "disabled")
        color_r, color_g, color_b = _parse_hex_color(pad_color)
        color_values = torch.ones((channels,), dtype=scaled.dtype, device=scaled.device)
        if channels >= 1: color_values[0] = color_r
        if channels >= 2: color_values[1] = color_g
        if channels >= 3: color_values[2] = color_b
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
        return io.Schema(node_id="YEColorMatch", display_name="YE Color Match", category="yet_essential/image", inputs=[io.Image.Input("image"), io.Image.Input("reference"), io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01), io.Combo.Input("method", options=cls.METHODS, default="Histogram Matching")], outputs=[io.Image.Output()])
    @classmethod
    def execute(cls, image: io.Image.Type, reference: io.Image.Type, strength: float, method: str) -> io.NodeOutput:
        if image.shape[-1] < 3 or reference.shape[-1] < 3:
            raise RuntimeError("YEColorMatch: image and reference must have at least 3 color channels.")
        strength = float(min(max(strength, 0.0), 1.0))
        if strength <= 0.0:
            return io.NodeOutput(torch.clamp(image, 0.0, 1.0))
        src_rgb = image[..., :3]
        ref_rgb = reference[..., :3]
        matched_rgb = _reinhard_color_transfer(src_rgb, ref_rgb) if method == "Reinhard (LAb Mean/Std)" else _histogram_match_rgb(src_rgb, ref_rgb)
        out_rgb = torch.lerp(src_rgb, matched_rgb, strength)
        out = torch.cat([out_rgb, image[..., 3:]], dim=-1) if image.shape[-1] > 3 else out_rgb
        return io.NodeOutput(torch.clamp(out, 0.0, 1.0))


class YEMaskUtility(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="YEMaskUtility", display_name="YE Mask Utility", category="yet_essential/image", inputs=[io.Mask.Input("mask"), io.Boolean.Input("invert", default=False), io.Int.Input("grow_shrink", default=0, min=-256, max=256, step=1), io.Float.Input("blur", default=0.0, min=0.0, max=64.0, step=0.1), io.Float.Input("threshold", default=0.0, min=0.0, max=1.0, step=0.01)], outputs=[io.Mask.Output()])
    @classmethod
    def execute(cls, mask: io.Mask.Type, invert: bool, grow_shrink: int, blur: float, threshold: float) -> io.NodeOutput:
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
        return io.Schema(node_id="YEHiResFix", display_name="YE HiRes Fix", category="yet_essential/sampling", inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"), io.Conditioning.Input("negative"), io.Latent.Input("latent"), io.Vae.Input("vae"), io.Combo.Input("upscale_model", options=folder_paths.get_filename_list("upscale_models")), io.Float.Input("upscale_by", default=1.5, min=1.0, max=4.0, step=0.05), io.Float.Input("denoise", default=0.45, min=0.0, max=1.0, step=0.01), io.Int.Input("seed", default=0, min=0, max=0x7FFFFFFFFFFFFFFF), io.Int.Input("steps", default=20, min=1, max=10000), io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01), io.Combo.Input("sampler_name", options=list(comfy.samplers.KSampler.SAMPLERS)), io.Combo.Input("scheduler", options=list(comfy.samplers.KSampler.SCHEDULERS))], outputs=[io.Latent.Output(display_name="latent"), io.Image.Output(display_name="image")])
    @classmethod
    def execute(cls, model: io.Model.Type, positive: io.Conditioning.Type, negative: io.Conditioning.Type, latent: io.Latent.Type, vae: io.Vae.Type, upscale_model: str, upscale_by: float, denoise: float, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str) -> io.NodeOutput:
        decoded = vae.decode(latent["samples"])
        if len(decoded.shape) == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        upscaled_image = _upscale_image_by_factor(decoded, upscale_model, upscale_by)
        upscaled_latent = {"samples": vae.encode(upscaled_image[:, :, :, :3])}
        latent_samples = comfy.sample.fix_empty_latent_channels(model, upscaled_latent["samples"])
        noise = comfy.sample.prepare_noise(latent_samples, seed, None)
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        refined_samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_samples, denoise=float(min(max(denoise, 0.0), 1.0)), disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, callback=callback, disable_pbar=disable_pbar, seed=seed)
        out_latent = {"samples": refined_samples}
        out_image = vae.decode(refined_samples)
        if len(out_image.shape) == 5:
            out_image = out_image.reshape(-1, out_image.shape[-3], out_image.shape[-2], out_image.shape[-1])
        return io.NodeOutput(out_latent, torch.clamp(out_image, 0.0, 1.0))


NODE_LIST = [YEImageUpscale, YEImageResize, YEColorMatch, YEMaskUtility, YEHiResFix]
