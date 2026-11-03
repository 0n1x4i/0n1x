# -*- coding: utf-8 -*-
"""
Onyx Setup Check — "will this workflow run on this machine?", answered BEFORE
the queue instead of at node 7 of 12.

The frontend (js/onyx_setup_check.js) sends the node types and the model file
names found in the open workflow. This compares them with what ComfyUI really
has:
  * node types  -> nodes.NODE_CLASS_MAPPINGS (every node class installed);
  * model files -> folder_paths: every file in every models/ sub-folder.
and answers with what is missing, plus which pack / which folder / which link,
from the "requirements" block of the registry (utils/registry_defaults.json,
refreshed from the Worker).

ComfyUI itself already lists missing NODE types when a workflow is loaded, but
it says nothing about missing MODEL files until the run fails, and it cannot
name packs that are not in the ComfyUI registry. Those two are the point here.
"""

from __future__ import annotations

import os
import time

from . import onyx_registry as _reg

MODEL_EXTENSIONS = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".sft")

_cache = {"at": 0.0, "files": None}
_CACHE_TTL = 20


def _norm(name: str) -> str:
    return name.replace("\\", "/").strip()


def _installed_files() -> dict:
    """{relative path: folder} and {basename: [relative paths]} of every model file."""
    if _cache["files"] is not None and time.time() - _cache["at"] < _CACHE_TTL:
        return _cache["files"]
    by_path, by_base = {}, {}
    try:
        import folder_paths
        for folder in list(getattr(folder_paths, "folder_names_and_paths", {}).keys()):
            try:
                names = folder_paths.get_filename_list(folder)
            except Exception:
                continue
            for n in names:
                rel = _norm(n)
                by_path.setdefault(rel, folder)
                by_base.setdefault(os.path.basename(rel).lower(), []).append((rel, folder))
    except Exception:
        pass
    _cache.update(at=time.time(), files={"by_path": by_path, "by_base": by_base})
    return _cache["files"]


def _installed_node_types() -> set:
    try:
        import nodes
        return set(nodes.NODE_CLASS_MAPPINGS)
    except Exception:
        return set()


def check(node_types, models) -> dict:
    """node_types: [str]. models: [{"node_type", "node_title", "widget", "value"}]."""
    req = _reg.registry().get("requirements") or {}
    packs = req.get("node_packs") or {}
    known_models = {k.lower(): v for k, v in (req.get("models") or {}).items() if isinstance(v, dict)}

    installed = _installed_node_types()
    missing_nodes = []
    if installed:                                   # empty = not inside ComfyUI: say nothing
        for t in sorted(set(node_types or [])):
            if t and t not in installed:
                info = packs.get(t) or {}
                missing_nodes.append({"type": t, "pack": info.get("pack"), "url": info.get("url")})

    files = _installed_files()
    missing_models, elsewhere = [], []
    seen = set()
    for m in models or []:
        value = _norm(str(m.get("value") or ""))
        if not value.lower().endswith(MODEL_EXTENSIONS) or value in seen:
            continue
        seen.add(value)
        if value in files["by_path"]:
            continue
        base = os.path.basename(value).lower()
        info = known_models.get(base) or {}
        if base in files["by_base"]:
            rel, folder = files["by_base"][base][0]
            # Present, but under another sub-folder name than the workflow expects:
            # ComfyUI will refuse the value, re-selecting it in the widget fixes it.
            elsewhere.append({"file": value, "found_as": rel, "folder": folder,
                              "node_type": m.get("node_type"), "node_title": m.get("node_title"),
                              "widget": m.get("widget")})
            continue
        missing_models.append({"file": value, "node_type": m.get("node_type"),
                               "node_title": m.get("node_title"), "widget": m.get("widget"),
                               "folder": info.get("folder"), "url": info.get("url")})

    return {
        "ok": not (missing_nodes or missing_models or elsewhere),
        "missing_nodes": missing_nodes,
        "missing_models": missing_models,
        "elsewhere": elsewhere,
        "checked_models": len(seen),
        "inside_comfyui": bool(installed),
    }


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    inst = PromptServer.instance
    if getattr(inst, "_onyx_setup_check_routes", False):
        return
    inst._onyx_setup_check_routes = True

    @inst.routes.post("/onyx/setup_check")
    async def _setup_check(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        return web.json_response(check(body.get("node_types") or [], body.get("models") or []))
