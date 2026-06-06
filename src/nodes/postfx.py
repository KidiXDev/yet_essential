from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ..core import BASE_DIR
from .common import YEPostFXPipe


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
        blurred = _gaussian_blur_bhwc(x, sigma=0.8)
        x = x + float(sharpness) * (x - blurred)
    return _clamp_image(x)


def _apply_style_stage(image: torch.Tensor, vignette_strength: float, vignette_softness: float, film_grain: float, grain_seed: int, chromatic_aberration: float, ca_angle: float, bloom_strength: float, bloom_radius: float, bloom_threshold: float) -> torch.Tensor:
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
    tensor: torch.Tensor
    domain_min: torch.Tensor
    domain_max: torch.Tensor


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
            files[f"{prefix}/{rel}"] = path
    return files


def _lut_options() -> list[str]:
    return sorted(_lut_file_map().keys())


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
                size = int(parts[1]); continue
            if head == "DOMAIN_MIN" and len(parts) >= 4:
                domain_min = [float(parts[1]), float(parts[2]), float(parts[3])]; continue
            if head == "DOMAIN_MAX" and len(parts) >= 4:
                domain_max = [float(parts[1]), float(parts[2]), float(parts[3])]; continue
            if len(parts) >= 3:
                values.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if size <= 1:
        raise RuntimeError(f"YEPostFXAddLUTStage: invalid LUT_3D_SIZE in {path.name}")
    expected = size * size * size
    if len(values) < expected:
        raise RuntimeError(f"YEPostFXAddLUTStage: LUT '{path.name}' has {len(values)} entries, expected {expected}.")
    lut_values = torch.tensor(values[:expected], dtype=torch.float32).view(size, size, size, 3)
    return _LUTData(tensor=lut_values.permute(3, 0, 1, 2).unsqueeze(0).contiguous(), domain_min=torch.tensor(domain_min, dtype=torch.float32), domain_max=torch.tensor(domain_max, dtype=torch.float32))


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
    sampled = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)
    graded_rgb = sampled.squeeze(2).permute(0, 2, 3, 1)
    out_rgb = torch.lerp(rgb, graded_rgb, strength)
    out = torch.cat([out_rgb, image[..., 3:]], dim=-1) if image.shape[-1] > 3 else out_rgb
    return _clamp_image(out)


class YEPostFXAddLUTStage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = _lut_options() or ["<no .cube files found>"]
        return io.Schema(node_id="YEPostFXAddLUTStage", display_name="YE PostFX - Add LUT Stage", category="yet_essential/postfx", inputs=[io.Boolean.Input("enabled", default=True), io.Combo.Input("lut_name", options=options, default=options[0]), io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01), YEPostFXPipe.Input("pipeline", optional=True)], outputs=[YEPostFXPipe.Output(display_name="pipeline")])
    @classmethod
    def execute(cls, enabled: bool, lut_name: str, strength: float, pipeline: dict[str, Any] | None = None) -> io.NodeOutput:
        out = _normalize_pipeline(pipeline)
        stages = list(out["stages"])
        if enabled and lut_name != "<no .cube files found>":
            lut_path = _resolve_lut_path(lut_name)
            stages.append({"kind": "lut", "lut_name": str(lut_name), "lut_path": str(lut_path), "strength": float(strength)})
        return io.NodeOutput({"stages": stages})


class YEPostFXAddAdjustStage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="YEPostFXAddAdjustStage", display_name="YE PostFX - Add Adjust Stage", category="yet_essential/postfx", inputs=[io.Boolean.Input("enabled", default=True), io.Float.Input("brightness", default=0.0, min=-1.0, max=1.0, step=0.01), io.Float.Input("contrast", default=1.0, min=0.0, max=3.0, step=0.01), io.Float.Input("saturation", default=1.0, min=0.0, max=3.0, step=0.01), io.Float.Input("sharpness", default=0.0, min=0.0, max=3.0, step=0.01), YEPostFXPipe.Input("pipeline", optional=True)], outputs=[YEPostFXPipe.Output(display_name="pipeline")])
    @classmethod
    def execute(cls, enabled: bool, brightness: float, contrast: float, saturation: float, sharpness: float, pipeline: dict[str, Any] | None = None) -> io.NodeOutput:
        out = _normalize_pipeline(pipeline)
        stages = list(out["stages"])
        if enabled:
            stages.append({"kind": "adjust", "brightness": float(brightness), "contrast": float(contrast), "saturation": float(saturation), "sharpness": float(sharpness)})
        return io.NodeOutput({"stages": stages})


class YEPostFXAddStyleStage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="YEPostFXAddStyleStage", display_name="YE PostFX - Add Style Stage", category="yet_essential/postfx", inputs=[io.Boolean.Input("enabled", default=True), io.Float.Input("vignette_strength", default=0.0, min=0.0, max=1.0, step=0.01), io.Float.Input("vignette_softness", default=0.5, min=0.0, max=1.0, step=0.01), io.Float.Input("film_grain", default=0.0, min=0.0, max=1.0, step=0.01), io.Int.Input("grain_seed", default=0, min=0, max=0x7FFFFFFF), io.Float.Input("chromatic_aberration", default=0.0, min=0.0, max=8.0, step=0.1), io.Float.Input("ca_angle", default=0.0, min=-180.0, max=180.0, step=1.0), io.Float.Input("bloom_strength", default=0.0, min=0.0, max=2.0, step=0.01), io.Float.Input("bloom_radius", default=1.5, min=0.0, max=12.0, step=0.1), io.Float.Input("bloom_threshold", default=0.7, min=0.0, max=1.0, step=0.01), YEPostFXPipe.Input("pipeline", optional=True)], outputs=[YEPostFXPipe.Output(display_name="pipeline")])
    @classmethod
    def execute(cls, enabled: bool, vignette_strength: float, vignette_softness: float, film_grain: float, grain_seed: int, chromatic_aberration: float, ca_angle: float, bloom_strength: float, bloom_radius: float, bloom_threshold: float, pipeline: dict[str, Any] | None = None) -> io.NodeOutput:
        out = _normalize_pipeline(pipeline)
        stages = list(out["stages"])
        if enabled:
            stages.append({"kind": "style", "vignette_strength": float(vignette_strength), "vignette_softness": float(vignette_softness), "film_grain": float(film_grain), "grain_seed": int(grain_seed), "chromatic_aberration": float(chromatic_aberration), "ca_angle": float(ca_angle), "bloom_strength": float(bloom_strength), "bloom_radius": float(bloom_radius), "bloom_threshold": float(bloom_threshold)})
        return io.NodeOutput({"stages": stages})


class YEPostFXMergePipeline(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="YEPostFXMergePipeline", display_name="YE PostFX - Merge Pipeline", category="yet_essential/postfx", inputs=[YEPostFXPipe.Input("pipeline_a"), YEPostFXPipe.Input("pipeline_b")], outputs=[YEPostFXPipe.Output(display_name="pipeline")])
    @classmethod
    def execute(cls, pipeline_a: dict[str, Any], pipeline_b: dict[str, Any]) -> io.NodeOutput:
        a = _normalize_pipeline(pipeline_a)["stages"]
        b = _normalize_pipeline(pipeline_b)["stages"]
        return io.NodeOutput({"stages": list(a) + list(b)})


class YEPostFXApplyPipeline(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="YEPostFXApplyPipeline", display_name="YE PostFX - Apply Pipeline", category="yet_essential/postfx", inputs=[io.Image.Input("image"), YEPostFXPipe.Input("pipeline")], outputs=[io.Image.Output()])
    @classmethod
    def execute(cls, image: io.Image.Type, pipeline: dict[str, Any]) -> io.NodeOutput:
        out = image
        for stage in _normalize_pipeline(pipeline)["stages"]:
            if not isinstance(stage, dict):
                continue
            kind = stage.get("kind")
            if kind == "adjust":
                out = _apply_adjust_stage(out, float(stage.get("brightness", 0.0)), float(stage.get("contrast", 1.0)), float(stage.get("saturation", 1.0)), float(stage.get("sharpness", 0.0)))
            elif kind == "style":
                out = _apply_style_stage(out, float(stage.get("vignette_strength", 0.0)), float(stage.get("vignette_softness", 0.5)), float(stage.get("film_grain", 0.0)), int(stage.get("grain_seed", 0)), float(stage.get("chromatic_aberration", 0.0)), float(stage.get("ca_angle", 0.0)), float(stage.get("bloom_strength", 0.0)), float(stage.get("bloom_radius", 1.5)), float(stage.get("bloom_threshold", 0.7)))
            elif kind == "lut":
                lut_path = str(stage.get("lut_path", ""))
                if lut_path:
                    out = _apply_lut_stage(out, lut_path=lut_path, strength=float(stage.get("strength", 1.0)))
        return io.NodeOutput(_clamp_image(out))


NODE_LIST = [YEPostFXAddAdjustStage, YEPostFXAddLUTStage, YEPostFXAddStyleStage, YEPostFXMergePipeline, YEPostFXApplyPipeline]
