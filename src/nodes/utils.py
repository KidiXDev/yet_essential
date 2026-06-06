from __future__ import annotations

import os
import uuid

import folder_paths
import numpy as np
from PIL import Image
from comfy_api.latest import io


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
    def execute(cls, image_a: io.Image.Type | None = None, image_b: io.Image.Type | None = None) -> io.NodeOutput:
        ui_payload = {"a_images": [], "b_images": []}
        if image_a is not None and len(image_a) > 0:
            ui_payload["a_images"] = cls._save_temp_images(image_a, "ye.compare.a")
        if image_b is not None and len(image_b) > 0:
            ui_payload["b_images"] = cls._save_temp_images(image_b, "ye.compare.b")
        return io.NodeOutput(ui=ui_payload)


NODE_LIST = [YESeedGenerator, YEImageComparer]
