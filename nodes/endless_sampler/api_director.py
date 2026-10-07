"""Cloud replacement for the local Gemma 4 director (Onyx addition).

Exposes the two methods the Gemma pipeline calls on its llama.cpp runtime:

- ``create_chat_completion(messages=..., max_tokens=..., response_format=...)``
- ``append_user_chat_completion(llama=..., content=..., ...)``

The conversation history is kept in Python, so the existing prompt templates,
JSON validators and repair loops in gemma4.py work unchanged — only the model
behind them moves from a local GGUF to Gemini (AI Studio / Vertex) or Grok.
No llama-cpp-python, no CUDA DLL, no VRAM used.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Sequence

import requests

BACKEND_GEMMA = "Gemma 4 (local)"
BACKEND_VERTEX = "Vertex AI"
BACKEND_GEMINI = "Gemini API"
BACKEND_GROK = "Grok API"
BACKEND_TIMELINE = "Timeline split (no LLM)"
DIRECTOR_BACKENDS = (BACKEND_TIMELINE, BACKEND_VERTEX, BACKEND_GEMINI, BACKEND_GROK, BACKEND_GEMMA)

DEFAULT_MODELS = {
    BACKEND_VERTEX: "gemini-3.6-flash",
    BACKEND_GEMINI: "gemini-3.6-flash",
    BACKEND_GROK: "grok-4-fast",
}

# Thinking models count their reasoning inside the output budget: never give
# them less than this, whatever the Gemma-sized max_tokens asked for.
_MIN_OUTPUT_TOKENS = 8192
_TIMEOUT = 180
_RETRIES = 4

# Same per-JSON cooldown as the other Onyx Vertex nodes (429 protection).
_VERTEX_JSON_COOLDOWN = 11.0
_VERTEX_LOCK = threading.Lock()
_VERTEX_LAST_CALL: dict[str, float] = {}
_VERTEX_ROTATION = {"offset": 0}


class ApiDirectorError(RuntimeError):
    pass


def _vertex_cooldown_wait(json_path: str) -> None:
    key = os.path.abspath(json_path)
    with _VERTEX_LOCK:
        now = time.monotonic()
        ready_at = _VERTEX_LAST_CALL.get(key, 0.0) + _VERTEX_JSON_COOLDOWN
        start = max(now, ready_at)
        _VERTEX_LAST_CALL[key] = start
    wait = start - time.monotonic()
    if wait > 0:
        print(f"[Onyx Endless Director] Vertex cooldown {wait:.1f}s ({os.path.basename(json_path)})")
        time.sleep(wait)


def _vertex_json_files(folder: str) -> list[str]:
    folder = str(folder or "").strip().strip('"')
    if not folder:
        raise ApiDirectorError("Vertex AI director needs vertex_json_folder (folder of service-account .json files).")
    if os.path.isfile(folder) and folder.lower().endswith(".json"):
        return [folder]
    if not os.path.isdir(folder):
        raise ApiDirectorError(f"Vertex JSON folder not found: {folder}")
    files = sorted(os.path.join(folder, n) for n in os.listdir(folder) if n.lower().endswith(".json"))
    if not files:
        raise ApiDirectorError(f"No .json file found in: {folder}")
    return files


def _split_data_url(url: str) -> tuple[str, str]:
    if not url.startswith("data:") or ";base64," not in url:
        raise ApiDirectorError("Director images must be base64 data URLs")
    header, data = url.split(";base64,", 1)
    return header[5:] or "image/jpeg", data


def _to_gemini(messages: Sequence[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else json.dumps(content))
            continue
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"text": content})
        else:
            for item in content or ():
                if item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    mime, data = _split_data_url(item["image_url"]["url"])
                    parts.append({"inlineData": {"mimeType": mime, "data": data}})
        if not parts:
            parts.append({"text": ""})
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    return "\n\n".join(system_parts), contents


class ApiDirectorClient:
    """Stateful chat stand-in for llama.cpp's Llama + MTMD handler pair."""

    def __init__(self, config: dict[str, Any]):
        self.backend = config.get("backend") or BACKEND_VERTEX
        if self.backend not in DEFAULT_MODELS:
            raise ApiDirectorError(f"Unknown director backend: {self.backend}")
        self.model = (str(config.get("model") or "").strip() or DEFAULT_MODELS[self.backend])
        self.api_key = str(config.get("api_key") or "").strip()
        self.vertex_json_folder = str(config.get("vertex_json_folder") or "")
        self.history: list[dict[str, Any]] = []

    # --- llama.cpp-compatible surface ---------------------------------
    def create_chat_completion(self, messages: Sequence[dict[str, Any]], max_tokens: int = 2048,
                               temperature: float = 1.0, top_p: float = 0.95,
                               response_format: dict[str, Any] | None = None, **_ignored: Any) -> dict[str, Any]:
        self.history = [dict(m) for m in messages]
        return self._complete(max_tokens, temperature, top_p)

    def append_user_chat_completion(self, llama: Any = None, content: Any = "", max_tokens: int = 2048,
                                    temperature: float = 1.0, top_p: float = 0.95,
                                    response_format: dict[str, Any] | None = None,
                                    **_ignored: Any) -> dict[str, Any]:
        self.history.append({"role": "user", "content": content})
        return self._complete(max_tokens, temperature, top_p)

    def close(self) -> None:
        self.history = []

    # --- internals ----------------------------------------------------
    def _complete(self, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        max_tokens = max(int(max_tokens or 0), _MIN_OUTPUT_TOKENS)
        last_error: Exception | None = None
        for attempt in range(1, _RETRIES + 1):
            try:
                if self.backend == BACKEND_GROK:
                    text = self._grok(max_tokens, temperature, top_p)
                else:
                    text = self._gemini(max_tokens, temperature, top_p)
                break
            except ApiDirectorError as error:
                last_error = error
                retryable = any(code in str(error) for code in ("HTTP 429", "HTTP 500", "HTTP 502",
                                                                "HTTP 503", "HTTP 504", "connect", "empty"))
                if not retryable or attempt == _RETRIES:
                    raise
                delay = 5 * attempt
                logging.warning("Onyx Endless Director %s: %s — retry %d/%d in %ds",
                                self.backend, error, attempt, _RETRIES - 1, delay)
                time.sleep(delay)
        else:  # pragma: no cover
            raise ApiDirectorError(str(last_error))
        self.history.append({"role": "assistant", "content": text})
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    def _gemini_body(self, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        system, contents = _to_gemini(self.history)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature),
                "topP": float(top_p),
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _gemini(self, max_tokens: int, temperature: float, top_p: float) -> str:
        body = self._gemini_body(max_tokens, temperature, top_p)
        if self.backend == BACKEND_VERTEX:
            url, headers, label = self._vertex_target()
        else:
            key = self.api_key or os.getenv("GEMINI_API_KEY", "")
            if not key:
                raise ApiDirectorError("Gemini API director needs director_api_key (or GEMINI_API_KEY).")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            headers = {"x-goog-api-key": key}
            label = "AI Studio"
        print(f"[Onyx Endless Director] {self.backend} ({label}) | model={self.model} | turns={len(body['contents'])}")
        try:
            response = requests.post(url, headers={**headers, "Content-Type": "application/json"},
                                     json=body, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            raise ApiDirectorError(f"Could not connect to {self.backend}: {exc}") from exc
        if response.status_code != 200:
            raise ApiDirectorError(f"{self.backend} HTTP {response.status_code}: {response.text[:600]}")
        try:
            parts = response.json()["candidates"][0]["content"]["parts"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ApiDirectorError(f"{self.backend} returned an empty/blocked response: {response.text[:600]}") from exc
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
        if not text.strip():
            raise ApiDirectorError(f"{self.backend} returned an empty response")
        return text

    def _vertex_target(self) -> tuple[str, dict[str, str], str]:
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
        except ImportError as exc:
            raise ApiDirectorError("Vertex AI director needs the google-auth package.") from exc
        files = _vertex_json_files(self.vertex_json_folder)
        index = _VERTEX_ROTATION["offset"] % len(files)
        _VERTEX_ROTATION["offset"] = index + 1
        json_path = files[index]
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                project_id = json.load(fh).get("project_id", "")
            credentials = service_account.Credentials.from_service_account_file(
                json_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(Request())
        except Exception as exc:
            raise ApiDirectorError(f"Invalid Vertex service-account file {os.path.basename(json_path)}: {exc}") from exc
        _vertex_cooldown_wait(json_path)
        url = (f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/"
               f"publishers/google/models/{self.model}:generateContent")
        label = f"{os.path.basename(json_path)} {index + 1}/{len(files)}"
        return url, {"Authorization": f"Bearer {credentials.token}"}, label

    def _grok(self, max_tokens: int, temperature: float, top_p: float) -> str:
        key = self.api_key or os.getenv("XAI_API_KEY", "")
        if not key:
            raise ApiDirectorError("Grok director needs director_api_key (or XAI_API_KEY).")
        body = {
            "model": self.model,
            "messages": self.history,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        print(f"[Onyx Endless Director] Grok | model={self.model} | turns={len(self.history)}")
        try:
            response = requests.post("https://api.x.ai/v1/chat/completions",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                                     json=body, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            raise ApiDirectorError(f"Could not connect to Grok/xAI: {exc}") from exc
        if response.status_code != 200:
            raise ApiDirectorError(f"Grok HTTP {response.status_code}: {response.text[:600]}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ApiDirectorError("Grok returned an invalid response") from exc
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not str(content).strip():
            raise ApiDirectorError("Grok returned an empty response")
        return str(content)


def is_api_backend(backend: str | None) -> bool:
    return bool(backend) and backend not in (BACKEND_GEMMA, BACKEND_TIMELINE)
