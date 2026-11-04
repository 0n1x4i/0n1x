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

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('h3_tools_bundle', 'onyx_conditioning_cache', __package__)
    return ns


_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_PACK_ROOT, "_cache", "conditioning")

MAX_DISK_MB = float(os.environ.get("ONYX_COND_CACHE_MB", "300") or 300)
_MEM_MAX = 8                # conditionings kept in RAM (a few MB each)
_memory: "OrderedDict[str, object]" = OrderedDict()     # path -> conditioning, LRU


def _safe_name(*args, **kwargs):
    return _onyx_core()['_safe_name'](*args, **kwargs)



def _path(*args, **kwargs):
    return _onyx_core()['_path'](*args, **kwargs)



def encoder_fingerprint(*args, **kwargs):
    return _onyx_core()['encoder_fingerprint'](*args, **kwargs)



def _to_cpu(*args, **kwargs):
    return _onyx_core()['_to_cpu'](*args, **kwargs)



def _remember(*args, **kwargs):
    return _onyx_core()['_remember'](*args, **kwargs)



def _load(*args, **kwargs):
    return _onyx_core()['_load'](*args, **kwargs)



def lookup(*args, **kwargs):
    return _onyx_core()['lookup'](*args, **kwargs)



def store(*args, **kwargs):
    return _onyx_core()['store'](*args, **kwargs)



def stats(*args, **kwargs):
    return _onyx_core()['stats'](*args, **kwargs)



def prune(*args, **kwargs):
    return _onyx_core()['prune'](*args, **kwargs)



def clear(*args, **kwargs):
    return _onyx_core()['clear'](*args, **kwargs)



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

    def encode(self, *args, **kwargs):
        return _onyx_core()['OnyxConditioningCache'].encode(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {"OnyxConditioningCache": OnyxConditioningCache}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxConditioningCache": "Onyx Conditioning Cache (text encode)"}
