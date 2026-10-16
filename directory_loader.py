"""
ComfyUI node — onyx Directory Image Loader.
Loads images from a directory one at a time, auto-incrementing through all files.
Designed for batch workflows like Seedream 4.5 Edit.
"""

import os
import logging

import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths


from .nodes.onyx_render_profile import ensure_profile_ready

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('utils_bundle', 'directory_loader', __package__)
    return ns

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _list_input_subdirs():
    """List subdirectories inside ComfyUI's input folder."""
    input_dir = folder_paths.get_input_directory()
    dirs = []
    for f in os.listdir(input_dir):
        full = os.path.join(input_dir, f)
        if os.path.isdir(full):
            dirs.append(f)
    dirs.sort(key=str.lower)
    if not dirs:
        dirs = ["(no folders found)"]
    return dirs


try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/directory_loader/list_folders")
    async def _list_folders(request):
        return web.json_response(_list_input_subdirs())

    @PromptServer.instance.routes.get("/directory_loader/status")
    async def _get_status(request):
        """Return current counter state for the frontend to poll."""
        return web.json_response(OnyxDirectoryImageLoaderNode._counters)

    @PromptServer.instance.routes.post("/directory_loader/reset_batch")
    async def _reset_batch(request):
        """Reset batch counters so a failed batch doesn't leave stale state."""
        OnyxDirectoryImageLoaderNode._batch_runs.clear()
        return web.json_response({"ok": True})
except Exception:
    pass


class OnyxDirectoryImageLoaderNode:
    """Load images from a directory one at a time.

    Set run_count to how many images to process.
    Queue once — it auto-queues the rest via frontend JS.
    """

    _counters = {}  # {directory: index} — which image to load next
    _batch_runs = {}  # {directory: count} — how many runs done in current batch

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_absolute_path": ("BOOLEAN", {"default": False}),
                "run_count": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
            },
            "optional": {
                "input_folder": (_list_input_subdirs(), {"default": _list_input_subdirs()[0]}),
                "absolute_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "filename", "filepath", "current_index", "total_count")
    FUNCTION = "load_image"
    OUTPUT_NODE = True
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load_image(self, *args, **kwargs):
        return _onyx_core()['OnyxDirectoryImageLoaderNode'].load_image(self, *args, **kwargs)
