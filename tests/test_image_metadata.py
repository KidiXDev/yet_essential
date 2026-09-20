import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))

from src.nodes.image import YEImageMetadataConnector, _metadata_model


class ImageMetadataTests(unittest.TestCase):
    def test_connector_accepts_model_socket(self):
        model_input = next(item for item in YEImageMetadataConnector.define_schema().inputs if item.id == "model")
        self.assertEqual(model_input.get_io_type(), "MODEL")
        self.assertTrue(model_input.optional)

    def test_model_object_resolves_registered_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            diffusion_dir = Path(directory) / "diffusion_models"

            for model_dir in (checkpoint_dir, diffusion_dir):
                model_path = model_dir / "subfolder" / "model.safetensors"
                model = types.SimpleNamespace(cached_patcher_init=(None, (str(model_path),)))

                with patch(
                    "src.nodes.image.folder_paths.get_folder_paths",
                    side_effect=lambda name: [str(checkpoint_dir if name == "checkpoints" else diffusion_dir)],
                ):
                    self.assertEqual(
                        _metadata_model(model),
                        (os.path.join("subfolder", "model.safetensors"), str(model_path)),
                    )
            self.assertEqual(_metadata_model(object()), ("", None))


if __name__ == "__main__":
    unittest.main()
