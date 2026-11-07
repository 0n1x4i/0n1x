# -*- coding: utf-8 -*-
"""
Onyx protected DATA — factory presets, prompt library, post looks.

The pack ships without them: they live on the Worker in the core
"onyx_data_core" (dicts GROK_PRESETS, NB_AIO_PRESETS, PROMPTS, LOOKS) and are
fetched with the Onyx key, like the protected code.

    from ..utils import onyx_remote_data as _rd
    presets = _rd.get("GROK_PRESETS", {})

* Never blocks ComfyUI: the fetch runs in a background thread. get() returns
  what is cached (or `default`), optionally waiting up to `wait` seconds for a
  fetch already in flight (used once at startup by the prompt selector).
* Refreshed every 10 minutes; a failed fetch (no key yet, offline) is retried
  after 1 minute and never raises.
* The user's own presets / files stay local and always win over the factory ones.
"""

from __future__ import annotations

import threading
import time

CORE = "onyx_data_core"
_TTL = 600.0
_RETRY = 60.0

_lock = threading.Lock()
_state = {"data": {}, "at": 0.0, "failed_at": 0.0, "thread": None}


def _fetch():
    try:
        from ..nodes.onyx_remote_exec import load_remote
        ns = load_remote(CORE)
        data = {k: v for k, v in ns.items() if k.isupper() and isinstance(v, (dict, list))}
        with _lock:
            _state.update(data=data, at=time.time(), failed_at=0.0)
    except Exception as e:  # no key yet, offline, core not published: silent
        with _lock:
            _state["failed_at"] = time.time()
        print(f"[Onyx] factory data not loaded yet ({str(e)[:120]})")
    finally:
        with _lock:
            _state["thread"] = None


def refresh(force: bool = False) -> threading.Thread | None:
    """Start a background fetch when the cache is stale (or `force`)."""
    with _lock:
        now = time.time()
        th = _state["thread"]
        if th is not None:
            return th
        fresh = _state["at"] and now - _state["at"] < _TTL
        recently_failed = _state["failed_at"] and now - _state["failed_at"] < _RETRY
        if not force and (fresh or recently_failed):
            return None
        th = threading.Thread(target=_fetch, name="onyx-remote-data", daemon=True)
        _state["thread"] = th
    th.start()
    return th


def get(key: str, default=None, wait: float = 0.0):
    # A caller that waits needs the data now: when nothing is cached, do not sit out the 1-minute retry window
    # of a fetch that failed earlier (fresh cloud worker: the startup fetch fails before the session exists,
    # and the first run then got the bare node defaults instead of the factory looks).
    with _lock:
        force = wait > 0 and not _state["data"] and time.time() - _state["failed_at"] > 3
    th = refresh(force=force)
    if wait > 0:
        th = th or _state["thread"]
        if th is not None and not _state["data"]:
            th.join(wait)
    with _lock:
        value = _state["data"].get(key)
    return default if value is None else value
