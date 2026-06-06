from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import folder_paths
from PIL import Image, ImageOps


class ModelPreviewManager:
    def __init__(self, base_dir: Path) -> None:
        self._cache: dict[str, str | None] = {}
        self._lock = threading.Lock()
        self._supported_exts = [".png", ".jpg", ".jpeg", ".webp"]
        self._thumb_dir = base_dir / "cache" / "thumbnails"
        self._thumb_dir.mkdir(parents=True, exist_ok=True)

    def find_preview(self, folder_type: str, model_name: str, res: int | None = None) -> str | None:
        cache_key = f"{folder_type}:{model_name}"
        if res:
            cache_key = f"{cache_key}:{res}"

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        preview_path = self._find_on_disk(folder_type, model_name)
        if not preview_path:
            return None

        if res:
            preview_path = self._get_thumbnail(preview_path, res)

        with self._lock:
            self._cache[cache_key] = preview_path
        return preview_path

    def _get_thumbnail(self, path: str, size: int) -> str:
        orig_p = Path(path)
        mtime = int(orig_p.stat().st_mtime)
        safe_name = orig_p.name.replace(".", "_")
        thumb_name = f"{safe_name}_{mtime}_{size}.webp"
        thumb_path = self._thumb_dir / thumb_name

        if thumb_path.exists():
            return str(thumb_path)

        try:
            with Image.open(orig_p) as img:
                img = ImageOps.exif_transpose(img)
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                img.save(thumb_path, "WEBP", quality=80)
            return str(thumb_path)
        except Exception as e:
            print(f"[yet_essential] Failed to generate thumbnail: {e}")
            return path

    def _find_on_disk(self, folder_type: str, model_name: str) -> str | None:
        full_path = folder_paths.get_full_path(folder_type, model_name)
        if not full_path:
            return None

        p = Path(full_path)
        parent = p.parent
        base_name = p.stem
        full_name = p.name

        candidates = []
        for ext in self._supported_exts:
            candidates.append(parent / f"{base_name}.preview{ext}")
            candidates.append(parent / f"{full_name}.preview{ext}")
            candidates.append(parent / f"{base_name}{ext}")
            candidates.append(parent / f"{full_name}{ext}")

        for cand in candidates:
            if cand.is_file():
                return str(cand)

        return None

    def list_models_with_previews(self, folder_type: str) -> list[dict[str, Any]]:
        models = folder_paths.get_filename_list(folder_type)
        return [
            {
                "name": model_name,
                "has_preview": self.find_preview(folder_type, model_name) is not None,
            }
            for model_name in models
        ]

    def _read_cm_info(self, folder_type: str, model_name: str) -> dict[str, Any] | None:
        full_path = folder_paths.get_full_path(folder_type, model_name)
        if not full_path:
            return None

        model_path = Path(full_path)
        parent = model_path.parent
        full_name = model_path.name
        stem = model_path.stem
        candidates = [parent / f"{full_name}.cm-info.json", parent / f"{stem}.cm-info.json"]

        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return None

    def _extract_base_model(self, metadata: dict[str, Any] | None) -> str | None:
        if not isinstance(metadata, dict):
            return None
        for value in (
            metadata.get("BaseModel"),
            metadata.get("baseModel"),
            metadata.get("base_model"),
        ):
            text = str(value or "").strip()
            if text:
                return text
        return None

    def list_models_with_metadata(self, folder_type: str) -> list[dict[str, Any]]:
        models = folder_paths.get_filename_list(folder_type)
        results = []
        for model_name in models:
            preview_path = self.find_preview(folder_type, model_name)
            metadata = self._read_cm_info(folder_type, model_name)
            results.append(
                {
                    "name": model_name,
                    "has_preview": preview_path is not None,
                    "base_model": self._extract_base_model(metadata),
                }
            )
        return results
