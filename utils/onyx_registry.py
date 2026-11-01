# -*- coding: utf-8 -*-
"""
Onyx registry — the single list of models (and API prices) for the whole pack.

Before this, the Gemini and Grok model lists were copied by hand into five
Python files and two JS files, and had drifted apart (h3_context_ir knew
gemini-3.8-flash, the prompt generator did not; lora_caption had no Grok 2...).

Sources, merged in this order (later wins):
  1. utils/registry_defaults.json         shipped with the pack
  2. the Worker's /registry endpoint      fetched in the background at startup,
                                          cached in _cache/registry.json (12 h)
  3. _cache/registry_override.json        local hand edits, never overwritten

Merging rules:
  * A model entry is matched by "id". Remote/override entries update fields of
    a known model or add a new one.
  * An id is never removed from a dropdown: ComfyUI stores the chosen string in
    saved workflows and refuses to load one whose value vanished. Retired ids
    are flagged `"deprecated": true` (+ optional "replacement") and sorted last.

Environment:
  ONYX_REGISTRY_URL   override the Worker URL, or "off" to never fetch.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry_defaults.json")
CACHE_DIR = os.path.join(_PACK_ROOT, "_cache")
_REMOTE_CACHE = os.path.join(CACHE_DIR, "registry.json")
_OVERRIDE_FILE = os.path.join(CACHE_DIR, "registry_override.json")
_PRICING_OVERRIDE_FILE = os.path.join(CACHE_DIR, "pricing_override.json")

_DEFAULT_URL = "https://onyx-sync.onyx-ai-pro.workers.dev/registry"
_REMOTE_TTL = 12 * 3600
_FETCH_TIMEOUT = 6

_lock = threading.RLock()
_merged: dict | None = None
_warned: set = set()


# ─────────────────────────────────────────────────────────────────────────────
# Loading / merging
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _merge_models(base: dict, extra: dict) -> None:
    for family, spec in (extra or {}).items():
        if not isinstance(spec, dict):
            continue
        dst = base.setdefault(family, {"default": None, "entries": []})
        if spec.get("default"):
            dst["default"] = spec["default"]
        index = {e["id"]: e for e in dst["entries"] if isinstance(e, dict) and "id" in e}
        for entry in spec.get("entries") or []:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            if entry["id"] in index:
                index[entry["id"]].update(entry)
            else:
                new = dict(entry)
                dst["entries"].append(new)
                index[new["id"]] = new


def _deep_merge(base: dict, extra: dict) -> None:
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def _build() -> dict:
    data = _read_json(_DEFAULTS_FILE)
    data.setdefault("models", {})
    data.setdefault("pricing", {})
    for layer in (_read_json(_REMOTE_CACHE).get("data") or {}, _read_json(_OVERRIDE_FILE)):
        _merge_models(data["models"], layer.get("models"))
        _deep_merge(data["pricing"], layer.get("pricing"))
    _deep_merge(data["pricing"], _read_json(_PRICING_OVERRIDE_FILE))
    return data


def registry() -> dict:
    """The merged registry (a fresh copy is built after a remote refresh)."""
    global _merged
    with _lock:
        if _merged is None:
            _merged = _build()
        return _merged


def reload() -> dict:
    global _merged
    with _lock:
        _merged = _build()
        return _merged


# ─────────────────────────────────────────────────────────────────────────────
# Remote refresh (background, never blocks startup, never raises)
# ─────────────────────────────────────────────────────────────────────────────

def _registry_url() -> str | None:
    url = os.environ.get("ONYX_REGISTRY_URL", "").strip()
    if url.lower() in ("off", "0", "false", "none"):
        return None
    return url or _DEFAULT_URL


def refresh_from_worker(force: bool = False) -> bool:
    """Fetch the Worker copy if the cached one is older than the TTL.
    Returns True when a new copy was stored."""
    url = _registry_url()
    if not url:
        return False
    cached = _read_json(_REMOTE_CACHE)
    if not force and time.time() - float(cached.get("fetched_at") or 0) < _REMOTE_TTL:
        return False
    try:
        import requests
        resp = requests.get(url, timeout=_FETCH_TIMEOUT)
        if resp.status_code != 200:
            return False      # endpoint not deployed yet / Worker down: keep what we have
        data = resp.json()
        if not isinstance(data, dict) or not ("models" in data or "pricing" in data):
            return False
    except Exception:
        return False
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = _REMOTE_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"fetched_at": time.time(), "data": data}, fh, indent=1)
        os.replace(tmp, _REMOTE_CACHE)
    except OSError:
        return False
    reload()
    return True


def start_background_refresh() -> None:
    threading.Thread(target=refresh_from_worker, name="onyx-registry-refresh", daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Queries used by the nodes
# ─────────────────────────────────────────────────────────────────────────────

def entries(family: str) -> list:
    spec = registry()["models"].get(family) or {}
    return [e for e in spec.get("entries") or [] if isinstance(e, dict) and e.get("id")]


def entry(family: str, model_id: str) -> dict:
    for e in entries(family):
        if e["id"] == model_id:
            return e
    return {}


def model_ids(family: str, *, vision: bool | None = None, include_deprecated: bool = True,
              always: tuple = ()) -> list:
    """Dropdown values for one family: current models first (registry order),
    deprecated ones last. `vision` filters on the entry's vision flag.
    `always` lists ids a node already shipped: they stay even if filtered out,
    so a saved workflow that used them keeps loading."""
    current, old = [], []
    for e in entries(family):
        if vision is not None and bool(e.get("vision", True)) != vision:
            continue
        if e.get("deprecated"):
            if include_deprecated:
                old.append(e["id"])
        else:
            current.append(e["id"])
    out = current + old
    for mid in always:
        if mid not in out:
            out.append(mid)
    return out


def default_model(family: str, fallback: str = "") -> str:
    spec = registry()["models"].get(family) or {}
    d = spec.get("default")
    ids = model_ids(family)
    if d in ids:
        return d
    return fallback or (ids[0] if ids else "")


def thinking_support(family: str = "gemini_text") -> dict:
    return {e["id"]: set(e.get("thinking") or []) for e in entries(family)}


def no_sampling_models(family: str = "gemini_text") -> set:
    return {e["id"] for e in entries(family) if e.get("no_sampling")}


def warn_if_deprecated(model_id: str, families=("gemini_text", "grok", "kie_grok"), node: str = "") -> None:
    """Print one console warning per process when a retired model is used."""
    for fam in families:
        e = entry(fam, model_id)
        if e.get("deprecated"):
            key = (fam, model_id)
            if key in _warned:
                return
            _warned.add(key)
            repl = e.get("replacement")
            where = f"[{node}] " if node else ""
            print(f"⚠️  [Onyx] {where}model '{model_id}' is retired by its provider and may stop "
                  f"answering." + (f" Switch to '{repl}'." if repl else ""))
            return


def public_payload() -> dict:
    """What the frontend gets from GET /onyx/registry."""
    reg = registry()
    return {
        "updated": reg.get("updated"),
        "models": {
            fam: {
                "default": default_model(fam),
                "entries": entries(fam),
            } for fam in reg["models"]
        },
        "pricing": reg.get("pricing", {}),
    }


def register_routes() -> None:
    """GET /onyx/registry — so the JS dropdown filters stop hard-coding lists."""
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    inst = PromptServer.instance
    if getattr(inst, "_onyx_registry_routes", False):
        return
    inst._onyx_registry_routes = True

    @inst.routes.get("/onyx/registry")
    async def _onyx_registry(_request):
        return web.json_response(public_payload())
