from __future__ import annotations

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_api.latest import io

from ..core import slerp_noise


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
    def execute(cls, model: io.Model.Type, seed: int, variation_seed: int, variation_strength: float, steps: int, cfg: float, sampler_name: str, scheduler: str, positive: io.Conditioning.Type, negative: io.Conditioning.Type, latent_image: io.Latent.Type, denoise: float = 1.0) -> io.NodeOutput:
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"])
        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        base_noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        strength = float(min(max(variation_strength, 0.0), 1.0))
        noise = slerp_noise(base_noise, comfy.sample.prepare_noise(latent_samples, variation_seed, batch_inds), strength) if strength > 0.0 else base_noise
        noise_mask = latent_image["noise_mask"] if "noise_mask" in latent_image else None
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_samples,
            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=seed,
        )
        out = latent_image.copy()
        out["samples"] = samples
        return io.NodeOutput(out)


NODE_LIST = [YEKSampler]
