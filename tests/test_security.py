import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class SecurityRegressionTests(unittest.TestCase):
    def test_csv_file_stays_in_tag_directory(self):
        settings_module = load_module("settings_test", ROOT / "src/services/settings.py")
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            (config_dir / "tag").mkdir()
            (config_dir / "tag/tags.csv").touch()
            settings = settings_module.Settings(config_dir / "setting.cfg")

            settings.update({"csv_file": "tags.csv"})
            self.assertEqual(settings.csv_file, "tags.csv")
            with self.assertRaises(ValueError):
                settings.update({"csv_file": "../secret.csv"})

    def test_model_catalog_ignores_custom_provider_url(self):
        llm = load_module("llm_test", ROOT / "src/services/llm.py")
        with patch.object(llm, "_fetch_model_catalog_uncached", return_value=[]) as fetch:
            llm.fetch_model_catalog("openrouter", "http://127.0.0.1/")
        self.assertEqual(fetch.call_args.kwargs["base_url"], "https://openrouter.ai/api/v1/")

    def test_preview_rejects_unbounded_sizes(self):
        sys.modules.setdefault("folder_paths", types.SimpleNamespace())
        preview = load_module("preview_test", ROOT / "src/services/model_preview.py")
        manager = object.__new__(preview.ModelPreviewManager)
        with self.assertRaises(ValueError):
            manager.find_preview("checkpoints", "model.safetensors", res=301)


if __name__ == "__main__":
    unittest.main()
