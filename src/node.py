from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import web
from server import PromptServer


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

    def _score_entry(self, entry: TagEntry, query: str) -> tuple[int, str]:
        if not query:
            return 1, entry.tag

        best_score = 0
        matched_on = entry.tag
        normalized_tag = _normalize_term(entry.tag)

        for term in entry.searchable_terms:
            if term == query:
                score = 140
            elif term.startswith(query):
                score = 100
            elif query in term:
                score = 60
            else:
                continue

            if term == normalized_tag:
                score += 10

            if score > best_score:
                best_score = score
                matched_on = term

        return best_score, matched_on

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        self._reload_if_needed()
        if limit <= 0:
            return []

        normalized_query = _normalize_term(query)
        with self._lock:
            entries = self._entries
            top_entry_ids = self._top_entry_ids
            prefix_buckets = self._prefix_buckets

        if not entries:
            return []

        if not normalized_query:
            candidate_ids = top_entry_ids
        else:
            prefix_key = normalized_query[: min(3, len(normalized_query))]
            candidate_ids = prefix_buckets.get(prefix_key, [])
            if not candidate_ids:
                candidate_ids = range(len(entries))

        ranked: list[tuple[int, int, str, str, TagEntry]] = []
        seen_ids: set[int] = set()

        for entry_idx in candidate_ids:
            if entry_idx in seen_ids:
                continue
            seen_ids.add(entry_idx)

            entry = entries[entry_idx]
            score, matched_on = self._score_entry(entry, normalized_query)
            if normalized_query and score <= 0:
                continue

            ranked.append((score, entry.total_post, entry.tag, matched_on, entry))

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


BASE_DIR = Path(__file__).resolve().parent.parent
AUTOCOMPLETE_CSV_PATH = BASE_DIR / "config" / "autocomplete.csv"
TAG_INDEX = TagAutocompleteIndex(AUTOCOMPLETE_CSV_PATH)


@PromptServer.instance.routes.get("/yet_essential/autocomplete/search")
async def search_autocomplete(request: web.Request) -> web.Response:
    query = request.query.get("q", "")
    try:
        limit = int(request.query.get("limit", "20"))
    except ValueError:
        limit = 20

    limit = max(1, min(limit, 50))
    return web.json_response(
        {
            "query": query,
            "items": TAG_INDEX.search(query=query, limit=limit),
        }
    )


class YEPromptAutocomplete:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str, dict[str, Any]]]]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "yet_essential.autocomplete": True,
                        "default": "",
                        "placeholder": "Type a prompt. Autocomplete comes from config/autocomplete.csv",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "output_prompt"
    CATEGORY = "yet_essential/prompt"

    def output_prompt(self, prompt: str) -> tuple[str]:
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "YEPromptAutocomplete": YEPromptAutocomplete,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YEPromptAutocomplete": "YE Prompt Autocomplete",
}
