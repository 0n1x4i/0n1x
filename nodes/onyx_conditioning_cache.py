# -*- coding: utf-8 -*-
"""
Onyx Conditioning Cache — a CLIP Text Encode that never loads the text encoder
again once a prompt has been encoded.

For workflows whose prompt never changes (a Flux 2 Klein refiner, an upscale
or restyle template): the first run encodes the prompt and saves the result
to disk; every later run — even after a ComfyUI restart — reads it back and
the text encoder (Klein's qwen_3_4b is ~8 GB) is never loaded at all.

How the encoder is skipped: `clip` is a LAZY input. ComfyUI evaluates it only
when check_lazy_status asks for it, and it is asked for only when the cache
has no entry for this exact text. On a hit nothing upstream runs, so the
CLIPLoader node is not executed and the encoder file is never read (as long
as nothing else in the graph uses that CLIPLoader).

Key: the exact prompt text + cache_name + the text-encoder chain as written in
the workflow (CLIPLoader file name, LoRA loaders and their strengths...). The
encoder is never loaded on a hit, so it is identified by the graph, not by its
weights: switch the encoder file or a CLIP LoRA strength and it encodes again.

Files: <pack>/_cache/conditioning/<cache_name>_<hash>.pt (CPU tensors, loaded
with torch.load(weights_only=True)). One file per distinct prompt, ~0.5-20 MB
each depending on the encoder. The folder is capped (ONYX_COND_CACHE_MB,
default 300 MB): past that, the least recently used files are deleted.

Place it where the CLIP Text Encode was: BEFORE any node that adds per-run
data to the conditioning (ReferenceLatent, guide / control nodes), since those
change every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict

from .onyx_render_profile import ensure_profile_ready

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_PACK_ROOT, "_cache", "conditioning")

MAX_DISK_MB = float(os.environ.get("ONYX_COND_CACHE_MB", "300") or 300)
_MEM_MAX = 8                # conditionings kept in RAM (a few MB each)
_memory: "OrderedDict[str, object]" = OrderedDict()     # path -> conditioning, LRU


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()) or "prompt"
    return name[:60]


def _path(text: str, cache_name: str, fingerprint: str = "") -> str:
    key = text if not fingerprint else f"{text}\0{fingerprint}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{_safe_name(cache_name)}_{digest}.pt")


def encoder_fingerprint(graph, node_id, input_name: str = "clip") -> str:
    """Identify the text encoder WITHOUT loading it: hash every node upstream of
    `input_name` (class + widget values) from the workflow ComfyUI is running.
    Empty when the graph is unknown or the input is not connected."""
    if not isinstance(graph, dict) or node_id is None:
        return ""
    link = ((graph.get(str(node_id)) or {}).get("inputs") or {}).get(input_name)
    if not (isinstance(link, list) and len(link) == 2):
        return ""
    parts, seen, todo = [], set(), [str(link[0])]
    while todo and len(seen) < 64:
        nid = todo.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph.get(nid) or {}
        values = {}
        for k, v in sorted((node.get("inputs") or {}).items()):
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                todo.append(str(v[0]))
                values[k] = f"<link:{v[1]}>"
            else:
                values[k] = v
        parts.append(json.dumps([node.get("class_type"), values], sort_keys=True, default=str))
    return hashlib.sha256("|".join(sorted(parts)).encode("utf-8")).hexdigest()[:16]


def _to_cpu(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _remember(path: str, cond):
    _memory[path] = cond
    _memory.move_to_end(path)
    while len(_memory) > _MEM_MAX:
        _memory.popitem(last=False)


def _load(path: str):
    if path in _memory:
        if os.path.isfile(path):
            _memory.move_to_end(path)
            return _memory[path]
        del _memory[path]                       # cleared from disk: forget it too
    if not os.path.isfile(path):
        return None
    import torch
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        cond = data["conditioning"]
    except Exception as e:
        print(f"⚠️  [Onyx Conditioning Cache] unreadable cache file, re-encoding: {e}")
        return None
    try:
        os.utime(path)                          # "recently used" for the size cap
    except OSError:
        pass
    _remember(path, cond)
    return cond


def lookup(text: str, cache_name: str, fingerprint: str = ""):
    """The cached conditioning for this text, or None."""
    return _load(_path(text, cache_name, fingerprint))


def store(text: str, cache_name: str, cond, fingerprint: str = "") -> bool:
    """Save a conditioning (atomic; refuses what cannot be reloaded safely), then
    keep the folder under the size cap. False when it could not be saved."""
    path = _path(text, cache_name, fingerprint)
    try:
        import torch
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        cpu = _to_cpu(cond)
        torch.save({"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text_preview": text[:200], "conditioning": cpu}, tmp)
        torch.load(tmp, map_location="cpu", weights_only=True)
        os.replace(tmp, path)
        _remember(path, cpu)
    except Exception as e:
        try:
            os.remove(path + ".tmp")
        except OSError:
            pass
        print(f"⚠️  [Onyx Conditioning Cache] could not cache this encoder's output ({e}); "
              f"it works, but the encoder will load every run.")
        return False
    prune()
    return True


def stats() -> dict:
    files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR)] if os.path.isdir(CACHE_DIR) else []
    files = [f for f in files if f.endswith(".pt") and os.path.isfile(f)]
    return {"files": len(files), "bytes": sum(os.path.getsize(f) for f in files),
            "cap_bytes": int(MAX_DISK_MB * 1024 * 1024), "dir": CACHE_DIR}


def prune(max_bytes: int | None = None) -> int:
    """Delete the least recently used files until the folder fits the cap. -> files deleted."""
    cap = int(MAX_DISK_MB * 1024 * 1024) if max_bytes is None else int(max_bytes)
    if not os.path.isdir(CACHE_DIR):
        return 0
    files = []
    for f in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, f)
        if f.endswith(".pt") and os.path.isfile(p):
            st = os.stat(p)
            files.append((st.st_mtime, st.st_size, p))
    total, removed = sum(s for _, s, _ in files), 0
    for _, size, p in sorted(files):
        if total <= cap:
            break
        try:
            os.remove(p)
            _memory.pop(p, None)
            total -= size
            removed += 1
        except OSError:
            pass
    return removed


def clear() -> int:
    _memory.clear()
    return prune(0)


class OnyxConditioningCache:
    """CLIP Text Encode with a disk cache: the text encoder is loaded only for a prompt never seen before."""

    DESCRIPTION = (
        "Drop-in replacement for CLIP Text Encode in workflows with a fixed prompt. The first run "
        "encodes and saves; later runs (even after a restart) read the saved result and the text "
        "encoder is never loaded. Change the text (or the encoder / its LoRAs) and it encodes once "
        "more. Put it before ReferenceLatent / guide nodes. The cache folder is capped at "
        f"{int(MAX_DISK_MB)} MB (oldest entries deleted first)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                    "tooltip": "The prompt. Same text = no encoder load."}),
                "cache_name": ("STRING", {
                    "default": "klein_refiner",
                    "tooltip": "Label of the cache files (only used to name them)."}),
            },
            "optional": {
                "clip": ("CLIP", {
                    "lazy": True,
                    "tooltip": "Text encoder (CLIPLoader). LAZY: only loaded when this prompt "
                               "has never been encoded. Must be connected for the first run."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "onyx_graph": "PROMPT"},
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Onyx/Conditioning"

    @classmethod
    def IS_CHANGED(cls, text="", cache_name="", clip=None, unique_id=None, onyx_graph=None, **_):
        path = _path(text, cache_name, encoder_fingerprint(onyx_graph, unique_id))
        return f"{path}:{os.path.isfile(path)}"

    def check_lazy_status(self, text="", cache_name="", clip=None, unique_id=None, onyx_graph=None, **_):
        if lookup(text, cache_name, encoder_fingerprint(onyx_graph, unique_id)) is not None:
            return []                       # hit: the encoder branch is never executed
        return ["clip"]

    def encode(self, text, cache_name, clip=None, unique_id=None, onyx_graph=None, **_):
        ensure_profile_ready()
        fp = encoder_fingerprint(onyx_graph, unique_id)
        cond = lookup(text, cache_name, fp)
        if cond is not None:
            print(f"♻️  [Onyx Conditioning Cache] '{cache_name}': prompt read from cache, "
                  f"text encoder not loaded.")
            return (cond,)
        if clip is None:
            raise RuntimeError("Onyx Conditioning Cache: this prompt is not cached yet — connect the "
                               "text encoder (CLIPLoader) to `clip` for this first run.")
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        if store(text, cache_name, cond, fp):
            print(f"💾 [Onyx Conditioning Cache] '{cache_name}': prompt encoded and saved — "
                  f"next runs will not load the text encoder.")
        return (cond,)


NODE_CLASS_MAPPINGS = {"OnyxConditioningCache": OnyxConditioningCache}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxConditioningCache": "Onyx Conditioning Cache (text encode)"}
