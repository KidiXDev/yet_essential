# Repository Guidelines

## Project Structure & Module Organization

This repository is a ComfyUI custom-node package. Python entry points live in `__init__.py` and `src/`. Add node definitions under `src/nodes/`, shared behavior under `src/services/`, and registration or HTTP integration in `src/registry.py` and `src/routes.py`. Browser extensions live in `web/`; reusable frontend helpers belong in `web/shared/`. User settings and autocomplete data are stored under `config/`. Treat `cache/` as generated runtime data, not source.

## Coding Style & Naming Conventions

Follow `.editorconfig`: UTF-8, LF line endings, four-space indentation, trimmed trailing whitespace, and a final newline. Use Python type hints, small functions, and descriptive `snake_case` names; classes use `PascalCase`. Node classes and IDs follow the existing `YE...` prefix. JavaScript uses four spaces, `camelCase` functions and variables, and `UPPER_SNAKE_CASE` constants. Reuse utilities from `src/nodes/common.py`, `src/services/`, or `web/shared/` before adding another helper. Do not add dependencies for behavior covered by the standard library or existing ComfyUI APIs.
