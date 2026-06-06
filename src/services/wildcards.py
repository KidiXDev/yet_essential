from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_WILDCARD_TOKEN_RE = re.compile(r"__(?P<name>[A-Za-z0-9_./\\-]+?)__")


@dataclass(slots=True)
class WildcardEntry:
    name: str
    path: Path
    prompts: tuple[str, ...]


class WildcardIndex:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self._lock = threading.Lock()
        self._state_key: tuple[tuple[str, int], ...] = ()
        self._entries: dict[str, WildcardEntry] = {}

    def _build_state_key(self) -> tuple[tuple[str, int], ...]:
        if not self.root_dir.exists():
            return ()
        files: list[tuple[str, int]] = []
        for path in sorted(self.root_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root_dir).with_suffix("").as_posix().lower()
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                mtime_ns = -1
            files.append((rel, mtime_ns))
        return tuple(files)

    def _reload_if_needed(self) -> None:
        state_key = self._build_state_key()
        if state_key == self._state_key:
            return

        with self._lock:
            state_key = self._build_state_key()
            if state_key == self._state_key:
                return

            entries: dict[str, WildcardEntry] = {}
            if self.root_dir.exists():
                for path in sorted(self.root_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    name = path.relative_to(self.root_dir).with_suffix("").as_posix()
                    try:
                        with path.open("r", encoding="utf-8") as handle:
                            prompts = tuple(line.strip() for line in handle if line.strip())
                    except OSError:
                        continue
                    if not prompts:
                        continue
                    entries[name.lower()] = WildcardEntry(name=name, path=path, prompts=prompts)

            self._entries = entries
            self._state_key = state_key

    def get(self, name: str) -> WildcardEntry | None:
        self._reload_if_needed()
        return self._entries.get(name.strip().replace("\\", "/").lower())

    def search(self, query: str, limit: int = 200) -> list[dict[str, Any]]:
        self._reload_if_needed()
        if limit <= 0:
            return []
        normalized_query = query.strip().replace("\\", "/").lower()
        entries = list(self._entries.values())
        if normalized_query:
            entries = [entry for entry in entries if normalized_query in entry.name.lower()]
        entries.sort(key=lambda entry: entry.name.lower())
        return [
            {
                "label": entry.name,
                "insert_text": f"__{entry.name}__",
                "tag": entry.name,
                "category": "Wildcard",
                "total_post": len(entry.prompts),
                "aliases": [],
                "matched_on": entry.name,
                "score": 0,
                "kind": "wildcard",
            }
            for entry in entries[:limit]
        ]

    def expand_prompt(self, prompt: str, max_depth: int = 10) -> str:
        expanded = str(prompt)
        for _ in range(max_depth):
            changed = False

            def replace(match: re.Match[str]) -> str:
                nonlocal changed
                entry = self.get(match.group("name"))
                if entry is None or len(entry.prompts) == 0:
                    return match.group(0)
                changed = True
                return random.choice(entry.prompts)

            expanded = _WILDCARD_TOKEN_RE.sub(replace, expanded)
            if not changed:
                break
        return expanded


def expand_prompt_wildcards(index: WildcardIndex, prompt: str) -> str:
    return index.expand_prompt(prompt)


def prompt_has_wildcards(prompt: str) -> bool:
    return _WILDCARD_TOKEN_RE.search(str(prompt)) is not None
