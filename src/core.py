from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import folder_paths
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = BASE_DIR / "config" / "setting.cfg"


class Settings:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.algorithm = "fuzzy"
        self.limit = 20
        self.sort_mode = "score"
        self.insertion_suffix = ", "
        self.smart_suffix = True
        self.spacing_mode = "space"
        self.escape_parentheses = True
        self.show_post_count = False
        self.autocomplete_position = "bottom_left"
        self.csv_file = ""
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.save()
            return

        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(("#", ";")):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key == "search_algorithm":
                        self.algorithm = value.lower()
                    elif key == "search_limit":
                        try:
                            self.limit = min(200, max(1, int(value)))
                        except ValueError:
                            pass
                    elif key == "sort_mode":
                        self.sort_mode = value.lower()
                    elif key == "insertion_suffix":
                        self.insertion_suffix = value.replace('"', '').replace("'", "")
                    elif key == "spacing_mode":
                        self.spacing_mode = value.lower()
                    elif key == "escape_parentheses":
                        self.escape_parentheses = value.lower() == "true"
                    elif key == "show_post_count":
                        self.show_post_count = value.lower() == "true"
                    elif key == "autocomplete_position":
                        self.autocomplete_position = self._normalize_autocomplete_position(value)
                    elif key == "smart_suffix":
                        self.smart_suffix = value.lower() == "true"
                    elif key == "csv_file":
                        self.csv_file = value
        except Exception as e:
            print(f"[yet_essential] Failed to load settings: {e}")

    def save(self) -> None:
        lines = [
            "# yet_essential.prompt_autocomplete settings",
            "",
            "# [Search]",
            f"search_algorithm={self.algorithm}",
            f"csv_file={self.csv_file}",
            f"search_limit={self.limit}",
            f"sort_mode={self.sort_mode}",
            "",
            "# [Formatting]",
            f'insertion_suffix="{self.insertion_suffix}"',
            f"spacing_mode={self.spacing_mode}",
            f"escape_parentheses={'true' if self.escape_parentheses else 'false'}",
            f"smart_suffix={'true' if self.smart_suffix else 'false'}",
            "",
            "# [UI]",
            f"show_post_count={'true' if self.show_post_count else 'false'}",
            f"autocomplete_position={self.autocomplete_position}",
        ]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            print(f"[yet_essential] Failed to save settings: {e}")

    def update(self, data: dict[str, Any]) -> None:
        if "search_algorithm" in data:
            self.algorithm = str(data["search_algorithm"]).lower()
        if "csv_file" in data:
            self.csv_file = str(data["csv_file"])
        if "search_limit" in data:
            try:
                self.limit = min(200, max(1, int(data["search_limit"])))
            except (ValueError, TypeError):
                pass
        if "sort_mode" in data:
            self.sort_mode = str(data["sort_mode"]).lower()
        if "insertion_suffix" in data:
            self.insertion_suffix = str(data["insertion_suffix"])
        if "spacing_mode" in data:
            self.spacing_mode = str(data["spacing_mode"]).lower()
        if "escape_parentheses" in data:
            self.escape_parentheses = bool(data["escape_parentheses"])
        if "show_post_count" in data:
            self.show_post_count = bool(data["show_post_count"])
        if "autocomplete_position" in data:
            self.autocomplete_position = self._normalize_autocomplete_position(
                str(data["autocomplete_position"])
            )
        if "smart_suffix" in data:
            self.smart_suffix = bool(data["smart_suffix"])
        self.save()

    @staticmethod
    def _normalize_autocomplete_position(value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        allowed = {
            "bottom_center",
            "bottom_right",
            "bottom_left",
            "top_center",
            "top_left",
            "top_right",
        }
        return normalized if normalized in allowed else "bottom_left"


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_term(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


@dataclass(slots=True)
class TagEntry:
    tag: str
    category: int
    total_post: int
    aliases: tuple[str, ...]
    searchable_terms: tuple[str, ...]


class TagAutocompleteIndex:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._lock = threading.Lock()
        self._last_mtime_ns = -1
        self._entries: list[TagEntry] = []
        self._prefix_buckets: dict[str, list[int]] = {}
        self._top_entry_ids: list[int] = []

    def update_path(self, csv_path: Path) -> None:
        if self.csv_path == csv_path:
            return
        with self._lock:
            self.csv_path = csv_path
            self._last_mtime_ns = -1
            self._entries = []
            self._prefix_buckets = {}
            self._top_entry_ids = []

    def _reload_if_needed(self) -> None:
        try:
            mtime_ns = self.csv_path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime_ns = -1

        if mtime_ns == self._last_mtime_ns:
            return

        with self._lock:
            if mtime_ns == self._last_mtime_ns:
                return

            entries: list[TagEntry] = []
            prefix_buckets: dict[str, list[int]] = {}

            if self.csv_path.is_file():
                with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    for row in reader:
                        if not row:
                            continue

                        tag = row[0].strip() if len(row) > 0 else ""
                        if not tag:
                            continue

                        # Skip header rows if present.
                        if tag.lower() == "tag":
                            continue

                        category = _safe_int(row[1].strip(), 0) if len(row) > 1 else 0
                        total_post = _safe_int(row[2].strip(), 0) if len(row) > 2 else 0

                        aliases: tuple[str, ...] = ()
                        if len(row) > 3:
                            raw_aliases = row[3].strip()
                            if raw_aliases and raw_aliases.lower() != "null":
                                aliases = tuple(
                                    alias.strip() for alias in raw_aliases.split(",") if alias.strip()
                                )

                        searchable_terms = {
                            _normalize_term(tag),
                            *(_normalize_term(alias) for alias in aliases),
                        }
                        searchable_terms.discard("")

                        if not searchable_terms:
                            continue

                        entry_idx = len(entries)
                        entry = TagEntry(
                            tag=tag,
                            category=category,
                            total_post=total_post,
                            aliases=aliases,
                            searchable_terms=tuple(sorted(searchable_terms)),
                        )
                        entries.append(entry)

                        for term in entry.searchable_terms:
                            for prefix_len in range(1, min(3, len(term)) + 1):
                                prefix = term[:prefix_len]
                                prefix_buckets.setdefault(prefix, []).append(entry_idx)

            self._entries = entries
            self._prefix_buckets = prefix_buckets
            self._top_entry_ids = sorted(
                range(len(entries)),
                key=lambda idx: (-entries[idx].total_post, entries[idx].tag),
            )
            self._last_mtime_ns = mtime_ns

    def _score_entry(self, entry: TagEntry, query: str, algorithm: str = "fuzzy") -> tuple[int, str]:
        if not query:
            return 1, entry.tag

        best_score = 0
        matched_on = entry.tag
        normalized_tag = _normalize_term(entry.tag)

        # Base scores for different qualities of match
        EXACT_SCORE = 140
        PREFIX_SCORE = 100
        CONTAINS_SCORE = 60
        FUZZY_SCORE = 30

        for term in entry.searchable_terms:
            score = 0
            if term == query:
                score = EXACT_SCORE
            elif term.startswith(query):
                score = PREFIX_SCORE
            elif algorithm != "prefix" and query in term:
                score = CONTAINS_SCORE
            elif algorithm == "fuzzy":
                # Check fuzzy match
                it = iter(term)
                if all(c in it for c in query):
                    score = FUZZY_SCORE

            if score <= 0:
                continue

            if term == normalized_tag:
                score += 10

            if score > best_score:
                best_score = score
                matched_on = term

        return best_score, matched_on

    def search(self, query: str, limit: int = 20, algorithm: str = "fuzzy", sort_mode: str = "score") -> list[dict[str, Any]]:
        self._reload_if_needed()
        if limit <= 0:
            return []

        normalized_query = _normalize_term(query)
        with self._lock:
            entries = self._entries
            top_entry_ids = self._top_entry_ids
            candidate_ids = range(len(entries))

        if not entries:
            return []

        ranked: list[tuple[int, int, str, str, TagEntry]] = []
        seen_ids: set[int] = set()

        for entry_idx in candidate_ids:
            if entry_idx in seen_ids:
                continue
            seen_ids.add(entry_idx)

            entry = entries[entry_idx]
            score, matched_on = self._score_entry(entry, normalized_query, algorithm=algorithm)
            if normalized_query and score <= 0:
                continue

            ranked.append((score, entry.total_post, entry.tag, matched_on, entry))

        if sort_mode == "alphabet":
            ranked.sort(key=lambda item: (item[2].lower(), -item[1]))
        elif sort_mode == "count":
            ranked.sort(key=lambda item: (-item[1], item[2].lower()))
        else:  # Default to score
            ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))

        results: list[dict[str, Any]] = []
        for score, _, _, matched_on, entry in ranked[:limit]:
            results.append(
                {
                    "label": entry.tag,
                    "insert_text": entry.tag,
                    "tag": entry.tag,
                    "category": entry.category,
                    "total_post": entry.total_post,
                    "aliases": list(entry.aliases[:5]),
                    "matched_on": matched_on,
                    "score": score,
                }
            )

        return results


def slerp_noise(noise_a: torch.Tensor, noise_b: torch.Tensor, strength: float) -> torch.Tensor:
    t = float(min(max(strength, 0.0), 1.0))
    if t <= 0.0:
        return noise_a
    if t >= 1.0:
        return noise_b

    # Shape: (batch, channels, height, width)
    # We want to interpolate correctly in high-dimensional space
    a = noise_a.reshape(noise_a.shape[0], -1)
    b = noise_b.reshape(noise_b.shape[0], -1)

    a_norm = torch.nn.functional.normalize(a, dim=1)
    b_norm = torch.nn.functional.normalize(b, dim=1)

    # Cosine of angle between vectors
    dot = torch.sum(a_norm * b_norm, dim=1, keepdim=True).clamp(-0.9995, 0.9995)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # Use LERP if vectors are nearly linear
    near_linear = torch.abs(sin_omega) < 1e-6
    interp = (torch.sin((1.0 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b
    lerp = (1.0 - t) * a + t * b

    out = torch.where(near_linear, lerp, interp)
    return out.reshape_as(noise_a)


class ModelPreviewManager:
    def __init__(self) -> None:
        self._cache: dict[str, str | None] = {}
        self._lock = threading.Lock()
        self._supported_exts = [".png", ".jpg", ".jpeg", ".webp"]
        self._thumb_dir = BASE_DIR / "cache" / "thumbnails"
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
        # Create a stable hash or name for the thumbnail
        # We'll use the path and mtime to ensure we update if the image changes
        mtime = int(orig_p.stat().st_mtime)
        safe_name = orig_p.name.replace(".", "_")
        thumb_name = f"{safe_name}_{mtime}_{size}.webp"
        thumb_path = self._thumb_dir / thumb_name

        if thumb_path.exists():
            return str(thumb_path)

        try:
            with Image.open(orig_p) as img:
                # Optimized resize
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
        # model.safetensors -> model
        base_name = p.stem 
        # model.safetensors -> model.safetensors
        full_name = p.name

        # Patterns to check
        # 1. model.preview.jpg
        # 2. model.safetensors.preview.jpg
        # 3. model.jpg
        # 4. model.safetensors.jpg
        
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
        results = []
        for model_name in models:
            preview_path = self.find_preview(folder_type, model_name)
            results.append({
                "name": model_name,
                "has_preview": preview_path is not None
            })
        return results


MODEL_PREVIEW_MANAGER = ModelPreviewManager()
SETTINGS = Settings(SETTINGS_PATH)
TAG_INDEX = TagAutocompleteIndex(BASE_DIR / "config" / "tag" / SETTINGS.csv_file)
