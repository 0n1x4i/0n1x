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

Key: the exact prompt text + cache_name. Changing one character makes a new
entry (the old one stays and is reused if you go back to it). The text
ENCODER is not part of the key — it is never loaded on a hit, so it cannot be
fingerprinted. If you switch encoders, change cache_name.

Files: <pack>/_cache/conditioning/<cache_name>_<hash>.pt (CPU tensors, loaded
with torch.load(weights_only=True)).

Place it where the CLIP Text Encode was: BEFORE any node that adds per-run
data to the conditioning (ReferenceLatent, guide / control nodes), since those
change every run.
"""

from __future__ import annotations

import hashlib
import os
import re

from .onyx_render_profile import ensure_profile_ready

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_PACK_ROOT, "_cache", "conditioning")

_memory: dict = {}          # path -> conditioning, so a hit costs nothing after the first read


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()) or "prompt"
    return name[:60]


def _path(text: str, cache_name: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{_safe_name(cache_name)}_{digest}.pt")


def _to_cpu(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _load(path: str):
    if path in _memory:
        return _memory[path]
    if not os.path.isfile(path):
        return None
    import torch
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        cond = data["conditioning"]
    except Exception as e:
        print(f"⚠️  [Onyx Conditioning Cache] unreadable cache file, re-encoding: {e}")
        return None
    _memory[path] = cond
    return cond


class OnyxConditioningCache:
    """CLIP Text Encode with a disk cache: the text encoder is loaded only for a prompt never seen before."""

    DESCRIPTION = (
        "Drop-in replacement for CLIP Text Encode in workflows with a fixed prompt. The first run "
        "encodes and saves; later runs (even after a restart) read the saved result and the text "
        "encoder is never loaded. Change the text and it encodes once more. Put it before "
        "ReferenceLatent / guide nodes. Changed the text encoder? Change cache_name."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                    "tooltip": "The prompt. Same text = no encoder load."}),
                "cache_name": ("STRING", {
                    "default": "klein_refiner",
                    "tooltip": "Name of the cache entry. Use a different name per text encoder "
                               "(e.g. klein_4b / klein_9b): the encoder is not part of the key."}),
            },
            "optional": {
                "clip": ("CLIP", {
                    "lazy": True,
                    "tooltip": "Text encoder (CLIPLoader). LAZY: only loaded when this prompt "
                               "has never been encoded. Must be connected for the first run."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Onyx/Conditioning"

    @classmethod
    def IS_CHANGED(cls, text="", cache_name="", clip=None):
        path = _path(text, cache_name)
        return f"{path}:{os.path.isfile(path)}"

    def check_lazy_status(self, text="", cache_name="", clip=None):
        if _load(_path(text, cache_name)) is not None:
            return []                       # hit: the encoder branch is never executed
        return ["clip"]

    def encode(self, text, cache_name, clip=None):
        ensure_profile_ready()
        path = _path(text, cache_name)
        cond = _load(path)
        if cond is not None:
            print(f"♻️  [Onyx Conditioning Cache] '{cache_name}': prompt read from cache, "
                  f"text encoder not loaded.")
            return (cond,)
        if clip is None:
            raise RuntimeError("Onyx Conditioning Cache: this prompt is not cached yet — connect the "
                               "text encoder (CLIPLoader) to `clip` for this first run.")
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        try:
            import torch
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = path + ".tmp"
            torch.save({"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "text_preview": text[:200], "conditioning": _to_cpu(cond)}, tmp)
            torch.load(tmp, map_location="cpu", weights_only=True)     # refuse what cannot be reloaded safely
            os.replace(tmp, path)
            _memory[path] = _to_cpu(cond)
            print(f"💾 [Onyx Conditioning Cache] '{cache_name}': prompt encoded and saved — "
                  f"next runs will not load the text encoder.")
        except Exception as e:
            try:
                os.remove(path + ".tmp")
            except OSError:
                pass
            print(f"⚠️  [Onyx Conditioning Cache] could not cache this encoder's output ({e}); "
                  f"it works, but the encoder will load every run.")
        return (cond,)


NODE_CLASS_MAPPINGS = {"OnyxConditioningCache": OnyxConditioningCache}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxConditioningCache": "Onyx Conditioning Cache (text encode)"}
