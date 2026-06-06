from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
                        if not tag or tag.lower() == "tag":
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
        exact_score = 140
        prefix_score = 100
        contains_score = 60
        fuzzy_score = 30

        for term in entry.searchable_terms:
            score = 0
            if term == query:
                score = exact_score
            elif term.startswith(query):
                score = prefix_score
            elif algorithm != "prefix" and query in term:
                score = contains_score
            elif algorithm == "fuzzy":
                it = iter(term)
                if all(c in it for c in query):
                    score = fuzzy_score

            if score <= 0:
                continue

            if term == normalized_tag:
                score += 10

            if score > best_score:
                best_score = score
                matched_on = term

        return best_score, matched_on

    def search(
        self,
        query: str,
        limit: int = 20,
        algorithm: str = "fuzzy",
        sort_mode: str = "score",
        category: int | None = None,
    ) -> list[dict[str, Any]]:
        self._reload_if_needed()
        if limit <= 0:
            return []

        normalized_query = _normalize_term(query)
        with self._lock:
            entries = self._entries
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
            if category is not None and entry.category != category:
                continue
            score, matched_on = self._score_entry(entry, normalized_query, algorithm=algorithm)
            if normalized_query and score <= 0:
                continue

            ranked.append((score, entry.total_post, entry.tag, matched_on, entry))

        if sort_mode == "alphabet":
            ranked.sort(key=lambda item: (item[2].lower(), -item[1]))
        elif sort_mode == "count":
            ranked.sort(key=lambda item: (-item[1], item[2].lower()))
        else:
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
