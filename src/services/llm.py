from __future__ import annotations

import json
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_NANOGPT_BASE_URL = "https://nano-gpt.com/api/v1"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
MODEL_CACHE_TTL_SECONDS = 300
_MODEL_CACHE: dict[tuple[str, str, str], tuple[float, list[dict[str, str]]]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/"


def default_base_url(provider: str) -> str:
    if provider == "openrouter":
        return DEFAULT_OPENROUTER_BASE_URL
    if provider == "nanogpt":
        return DEFAULT_NANOGPT_BASE_URL
    return DEFAULT_OPENAI_BASE_URL


def normalize_provider(provider: str) -> str:
    value = (provider or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"openrouter"}:
        return "openrouter"
    if value in {"nanogpt", "nano_gpt"}:
        return "nanogpt"
    return "openai_compatible"


def make_provider_config(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    provider_key = normalize_provider(provider)
    return {
        "provider": provider_key,
        "base_url": normalize_base_url(base_url or default_base_url(provider_key)),
        "api_key": (api_key or "").strip(),
        "model": (model or "").strip(),
        "timeout": max(int(timeout), 1),
    }


class LLMClient:
    def __init__(self, provider_config: dict[str, Any]) -> None:
        self.provider = normalize_provider(str(provider_config.get("provider", "")))
        self.base_url = normalize_base_url(
            str(provider_config.get("base_url", "")) or default_base_url(self.provider)
        )
        self.api_key = str(provider_config.get("api_key", "")).strip()
        self.model = str(provider_config.get("model", "")).strip()
        self.timeout = max(int(provider_config.get("timeout", 60) or 60), 1)

        if not self.api_key:
            raise ValueError("API key is required.")
        if not self.model:
            raise ValueError("Model is required.")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        }
        if self.provider == "nanogpt":
            headers["x-api-key"] = self.api_key
        return headers

    def _request_json(self, method: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        url = urljoin(self.base_url, endpoint.lstrip("/"))
        request = Request(url=url, data=body, method=method.upper(), headers=self._headers())
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API error {err.code}: {detail}") from err
        except URLError as err:
            raise RuntimeError(f"LLM API connection failed: {err.reason}") from err
        except json.JSONDecodeError as err:
            raise RuntimeError("LLM API returned invalid JSON.") from err

    def chat_completions(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        seed: int | None = None,
        json_mode: bool = False,
        stop: list[str] | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "presence_penalty": float(presence_penalty),
            "frequency_penalty": float(frequency_penalty),
        }
        if top_k > 0:
            payload["top_k"] = int(top_k)
        if max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)
        if seed is not None and seed >= 0:
            payload["seed"] = int(seed)
        if stop:
            payload["stop"] = stop
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = self._request_json("POST", "/chat/completions", payload)
        return {
            "text": self._extract_text(data),
            "raw": data,
            "model": str(data.get("model", self.model)),
        }

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            return "".join(text_parts)
        return str(content or "")


def _make_headers(provider: str, api_key: str) -> dict[str, str]:
    normalized_provider = normalize_provider(provider)
    headers = {
        "Accept": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if normalized_provider == "nanogpt" and token:
        headers["x-api-key"] = token
    return headers


def _request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = dict(headers)
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = Request(url=url, data=body, method=method.upper(), headers=request_headers)
    try:
        with urlopen(request, timeout=max(int(timeout), 1)) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API error {err.code}: {detail}") from err
    except URLError as err:
        raise RuntimeError(f"LLM API connection failed: {err.reason}") from err
    except json.JSONDecodeError as err:
        raise RuntimeError("LLM API returned invalid JSON.") from err


def fetch_model_catalog(
    provider: str,
    base_url: str,
    api_key: str = "",
    timeout: int = 60,
) -> list[dict[str, str]]:
    provider_key = normalize_provider(provider)
    resolved_base_url = normalize_base_url(default_base_url(provider_key))
    cache_key = (provider_key, resolved_base_url, str(api_key or "").strip())
    now = time.time()

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached and now - cached[0] < MODEL_CACHE_TTL_SECONDS:
            return list(cached[1])

    models = _fetch_model_catalog_uncached(
        provider=provider_key,
        base_url=resolved_base_url,
        api_key=api_key,
        timeout=timeout,
    )
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = (now, list(models))
    return models


def _fetch_model_catalog_uncached(
    provider: str,
    base_url: str,
    api_key: str,
    timeout: int,
) -> list[dict[str, str]]:
    if provider == "openrouter":
        data = _request_json(
            "GET",
            urljoin(base_url, "models"),
            headers=_make_headers(provider, api_key),
            timeout=timeout,
        )
        return _parse_model_list(data, prefer_name=True)

    if provider == "nanogpt":
        token = str(api_key or "").strip()
        if not token:
            raise RuntimeError("NanoGPT model list requires an API key.")
        data = _request_json(
            "GET",
            urljoin(base_url, "models?detailed=true"),
            headers=_make_headers(provider, token),
            timeout=timeout,
        )
        return _parse_model_list(data, prefer_name=True)

    return []


def _parse_model_list(data: dict[str, Any], prefer_name: bool) -> list[dict[str, str]]:
    items = data.get("data", [])
    if not isinstance(items, list):
        return []

    models: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        model_name = str(item.get("name", "")).strip()
        if not model_id:
            continue
        label = model_id
        if prefer_name and model_name and model_name != model_id:
            label = f"{model_name} ({model_id})"
        models.append({"id": model_id, "label": label})

    models.sort(key=lambda item: item["label"].lower())
    return models
