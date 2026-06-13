from __future__ import annotations

import torch
from comfy_api.latest import io


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
                io.Int.Input("width", default=1024, min=16, max=8192, step=8, optional=True),
                io.Int.Input("height", default=1024, min=16, max=8192, step=8, optional=True),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, preset: str, width: int = 1024, height: int = 1024, batch_size: int = 1) -> io.NodeOutput:
        if preset != "Custom":
            width, height = cls.DIMENSION_PRESETS[preset]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return io.NodeOutput({"samples": latent})


NODE_LIST = [YEEmptyLatentImage]
