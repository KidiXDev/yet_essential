from __future__ import annotations

from pathlib import Path
from typing import Any


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
                        self.insertion_suffix = value.replace('"', "").replace("'", "")
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
