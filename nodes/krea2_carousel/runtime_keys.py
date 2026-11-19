"""API keys for the Krea2 carousel nodes (Pose List Prep, Story Director).

The key typed in the node lives in the browser (localStorage) and is pushed to the
server, never written into the workflow. A server restart or a new browser / new
pod URL used to lose it ("The Grok API key is missing") even though it was typed.
Lookup order now:
  1. the key pushed by the node UI (process memory),
  2. this computer's saved copy of that key (user folder, onyx_studio/krea2_keys.json),
  3. the key saved in Onyx Studio for the same provider (onyx_studio/keys.json),
  4. the environment (XAI_API_KEY / GEMINI_API_KEY).
"""

from __future__ import annotations

import json
import os
import threading

from .pose_generator import RUNTIME_KEYS

_STUDIO_PROVIDER = {"Grok API": "xai", "Gemini API": "gemini"}
_ENV = {"Grok API": "XAI_API_KEY", "Gemini API": "GEMINI_API_KEY"}
SUPPORTED = tuple(_STUDIO_PROVIDER)
_lock = threading.Lock()


def _studio_dir() -> str:
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:
        return ""
    if os.path.isdir(os.path.join(base, "default")):
        base = os.path.join(base, "default")
    return os.path.join(base, "onyx_studio")


def _read(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _saved_path() -> str:
    d = _studio_dir()
    return os.path.join(d, "krea2_keys.json") if d else ""


def _write_saved(data: dict) -> None:
    path = _saved_path()
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def remember(backend: str, value: str, explicit: bool = False) -> None:
    """A key pushed by the node UI. An empty value only clears when the user emptied the field."""
    if backend not in SUPPORTED:
        return
    value = (value or "").strip()[:500]
    with _lock:
        saved = _read(_saved_path())
        if value:
            RUNTIME_KEYS[backend] = value
            if saved.get(backend) != value:
                saved[backend] = value
                _write_saved(saved)
        elif explicit:
            RUNTIME_KEYS.pop(backend, None)
            if backend in saved:
                saved.pop(backend)
                _write_saved(saved)


def get(backend: str) -> str:
    """The key to use for `backend` ("Grok API" / "Gemini API"), or ""."""
    key = (RUNTIME_KEYS.get(backend) or "").strip()
    if key:
        return key
    key = (_read(_saved_path()).get(backend) or "").strip()
    if not key and backend in _STUDIO_PROVIDER and _studio_dir():
        key = (_read(os.path.join(_studio_dir(), "keys.json")).get(_STUDIO_PROVIDER[backend]) or "").strip()
    if not key and backend in _ENV:
        key = (os.getenv(_ENV[backend]) or "").strip()
    if key:
        RUNTIME_KEYS[backend] = key
    return key
