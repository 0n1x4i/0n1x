# -*- coding: utf-8 -*-
"""
Onyx remote exec — fetches protected node implementations from the Worker
instead of shipping them as .py files.

Boring name on purpose, same philosophy as onyx_render_profile.py: this is
the one function that matters here, and it does exactly what its name says
and nothing more devious-sounding.

Design:
  - The node's INPUT_TYPES / RETURN_TYPES / class scaffolding stays local, so
    ComfyUI can register the node at startup with no network call.
  - The actual FUNCTION body (and whatever helpers/constants it needs) is
    fetched fresh on every single execution and exec()'d into a throwaway
    namespace — never written to disk, never kept between calls. A folder
    copy of this pack therefore ships plumbing, not the logic.
  - No in-process cache, deliberately: a revoked key stops the node on its
    very next run instead of waiting for a ComfyUI restart. The round trip
    (~50-250ms) is once per node execution, not per internal loop iteration,
    so this is cheap next to the seconds-to-minutes the node itself takes.
  - Reuses the token already entered in the Onyx Session node: read straight
    from the same on-disk cache onyx_render_profile.py writes, so there is
    no second place for the user to paste a key.
"""

import requests

from .onyx_render_profile import _read_cache, _get_node_id

_CODE_URL = "https://onyx-sync.onyx-ai-pro.workers.dev/code"
_REQUEST_TIMEOUT = 15


def load_remote(name: str, extra_globals: dict = None) -> dict:
    """Fetch protected source `name` and exec() it into a fresh namespace.

    Returns the namespace dict (use ns["some_function"](...) to call what's
    in it). Raises RuntimeError — same vague "Sync error" shape as
    ensure_profile_ready() — on any failure: no token cached, network down,
    revoked key, or the name doesn't exist server-side.

    `extra_globals`, when given, is merged into the exec() namespace BEFORE
    the source runs. Used for the rare protected module whose original file
    had a relative import (`from .foo import bar`) or relied on `__file__`:
    those statements can't survive being exec()'d outside the package, so the
    local wrapper resolves the real object once (a normal import, since the
    wrapper itself is a real module) and hands it over as an already-bound
    name instead. nano_banana_aio.py is the first to need this.
    """
    cache = _read_cache()
    token = (cache.get("token") or "").strip()
    if not token:
        raise RuntimeError(
            "Sync error SS-205: place the Onyx Session node anywhere in this "
            "workflow and run it once, then retry."
        )

    node_id = _get_node_id()
    try:
        resp = requests.post(
            _CODE_URL, json={"token": token, "node_id": node_id, "name": name},
            timeout=_REQUEST_TIMEOUT,
        )
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Sync error SS-205: could not reach the server ({e}).")

    if not data.get("valid"):
        raise RuntimeError(
            f"Sync error SS-205: {data.get('reason', 'unknown')} — this node "
            f"could not load its implementation. Check your Onyx Session key."
        )

    source = data.get("source") or ""
    ns: dict = {"__name__": f"onyx_remote_{name}"}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(source, f"<onyx-remote:{name}>", "exec"), ns)
    return ns
