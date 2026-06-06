from __future__ import annotations

import hashlib
import json
import sqlite3
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
        self._cache_dir = base_dir / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._thumb_dir = self._cache_dir / "thumbnails"
        self._thumb_dir.mkdir(parents=True, exist_ok=True)
        self._thumb_db_path = self._cache_dir / "thumb.db"
        self._init_thumb_db()

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

    def _init_thumb_db(self) -> None:
        with sqlite3.connect(self._thumb_db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            columns = {
                row[1]: row[2]
                for row in conn.execute("PRAGMA table_info(thumbnails)").fetchall()
            }
            if columns and "thumb_path" not in columns:
                conn.execute("DROP TABLE thumbnails")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thumbnails (
                    source_path TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    thumb_path TEXT NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                    PRIMARY KEY (source_path, mtime_ns, size)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_thumbnails_lookup
                ON thumbnails (source_path, size, mtime_ns)
                """
            )

    def _get_thumbnail(self, path: str, size: int) -> str:
        orig_p = Path(path)
        try:
            mtime_ns = orig_p.stat().st_mtime_ns
        except OSError:
            return path

        path_hash = hashlib.sha1(str(orig_p).encode("utf-8")).hexdigest()[:12]
        thumb_name = f"{orig_p.stem}_{path_hash}_{mtime_ns}_{size}.webp"
        thumb_path = self._thumb_dir / thumb_name
        cached = self._read_thumbnail_from_db(str(orig_p), mtime_ns, size)
        if cached:
            cached_path = Path(cached)
            if cached_path.is_file():
                return str(cached_path)

        if thumb_path.is_file():
            self._write_thumbnail_to_db(str(orig_p), mtime_ns, size, str(thumb_path))
            return str(thumb_path)

        try:
            with Image.open(orig_p) as img:
                img = ImageOps.exif_transpose(img)
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                img.save(thumb_path, "WEBP", quality=80)
            self._write_thumbnail_to_db(str(orig_p), mtime_ns, size, str(thumb_path))
            return str(thumb_path)
        except Exception as e:
            print(f"[yet_essential] Failed to generate thumbnail: {e}")
            return path

    def _read_thumbnail_from_db(self, source_path: str, mtime_ns: int, size: int) -> str | None:
        with sqlite3.connect(self._thumb_db_path) as conn:
            row = conn.execute(
                """
                SELECT thumb_path
                FROM thumbnails
                WHERE source_path = ? AND mtime_ns = ? AND size = ?
                """,
                (source_path, mtime_ns, size),
            ).fetchone()
        return str(row[0]) if row else None

    def _write_thumbnail_to_db(self, source_path: str, mtime_ns: int, size: int, thumb_path: str) -> None:
        with sqlite3.connect(self._thumb_db_path) as conn:
            stale_rows = conn.execute(
                "SELECT thumb_path FROM thumbnails WHERE source_path = ? AND size = ? AND mtime_ns != ?",
                (source_path, size, mtime_ns),
            ).fetchall()
            conn.execute(
                """
                DELETE FROM thumbnails
                WHERE source_path = ? AND size = ? AND mtime_ns != ?
                """,
                (source_path, size, mtime_ns),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO thumbnails (source_path, mtime_ns, size, thumb_path)
                VALUES (?, ?, ?, ?)
                """,
                (source_path, mtime_ns, size, thumb_path),
            )
        for row in stale_rows:
            stale_path = Path(str(row[0]))
            if stale_path != Path(thumb_path) and stale_path.is_file():
                try:
                    stale_path.unlink()
                except OSError:
                    continue

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
