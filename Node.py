"""
ComfyUI custom nodes for Nano Banana Pro and Nano Banana 2 APIs.
- Nano Banana Pro Edit: edit with images (Gemini 3 Pro).
- Nano Banana 2 Edit: edit with images (Gemini 3.1 Flash).
Each node returns image as ComfyUI IMAGE tensor [B, H, W, C].
"""

import logging
import io
import base64
import random as _random

import torch
import numpy as np
from PIL import Image


from .nodes.onyx_render_profile import ensure_profile_ready
from .utils.onyx_http import post_json, get_json, download as _http_download, OnyxAPIError
from .utils.onyx_parallel import map_parallel
from .utils import onyx_api_cache as _cache
from .utils import onyx_cost as _cost

_SAFETY_THRESHOLDS = [
    "OFF",
    "BLOCK_NONE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE",
]

_HARM_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)

_ASPECT_RATIOS = [
    "auto", "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
]

_IMAGE_SIZES = ["1K", "2K", "4K"]

# Widget appended at the END of every node's optional inputs: ComfyUI saves
# widget values by position, so a widget inserted in the middle would shift
# every saved workflow's settings onto the wrong fields.
_CACHE_WIDGET = {"cache_mode": (_cache.CACHE_MODES, {"default": _cache.CACHE_OFF,
                                                     "tooltip": _cache.CACHE_TOOLTIP})}
_HIDDEN = {"unique_id": "UNIQUE_ID"}


def _run_with_cache_and_cost(*args, **kwargs):
    return _onyx_core()['_run_with_cache_and_cost'](*args, **kwargs)



# ---------- Nano Banana Pro Edit ----------

class NanoBananaProEditAPINode:
    """Edit/Generate images via Gemini 3 Pro (gemimg.py reference implementation)."""

    MODEL_ID = "gemini-3-pro-image"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (_IMAGE_SIZES, {"default": "2K"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "safety_threshold": (_SAFETY_THRESHOLDS, {"default": "OFF"}),
                **_CACHE_WIDGET,
            },
            "hidden": _HIDDEN,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, *args, **kwargs):
        return _onyx_core()['NanoBananaProEditAPINode'].edit(self, *args, **kwargs)


    def _generate(self, *args, **kwargs):
        return _onyx_core()['NanoBananaProEditAPINode']._generate(self, *args, **kwargs)



# ---------- Nano Banana 2 Edit ----------

class NanoBanana2EditAPINode:
    """Edit images via Gemini 3.1 Flash API."""

    MODEL_ID = "gemini-3.1-flash-image"
    BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (_IMAGE_SIZES, {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "safety_threshold": (_SAFETY_THRESHOLDS, {"default": "OFF"}),
                **_CACHE_WIDGET,
            },
            "hidden": _HIDDEN,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, *args, **kwargs):
        return _onyx_core()['NanoBanana2EditAPINode'].edit(self, *args, **kwargs)


    def _generate(self, *args, **kwargs):
        return _onyx_core()['NanoBanana2EditAPINode']._generate(self, *args, **kwargs)



# ---------- Seedream 4.5 Edit (WaveSpeed) ----------

import time as _time

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('utils_bundle', 'api_nodes', __package__)
    return ns


_SEEDREAM_SIZES = [
    "auto — match input (2K)",
    "1920*1920 — square (1:1)",
    "2560*1920 — landscape_4_3 (4:3)",
    "1920*2560 — portrait_4_3 (3:4)",
    "1536*1920 — portrait_4_5 (4:5)",
    "1920*1536 — landscape_5_4 (5:4)",
    "2560*1440 — landscape_16_9 (16:9)",
    "1440*2560 — portrait_16_9 (9:16)",
]


class SeedreamEditAPINode:
    """Edit images via ByteDance Seedream 4.5 Edit on WaveSpeed API.

    Supports 1-10 reference images for context-aware editing with
    facial feature, lighting, and color tone preservation up to 4K.
    """

    BASE_URL = "https://api.wavespeed.ai/api/v3/bytedance/seedream-v4.5/edit"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "wavespeed_apikey": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "size": (_SEEDREAM_SIZES, {"default": "auto — match input (2K)"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "use_custom_size": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                **_CACHE_WIDGET,
            },
            "hidden": _HIDDEN,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, *args, **kwargs):
        return _onyx_core()['SeedreamEditAPINode'].edit(self, *args, **kwargs)


    def _generate(self, *args, **kwargs):
        return _onyx_core()['SeedreamEditAPINode']._generate(self, *args, **kwargs)



# ---------- Seedream 4.5 Edit (fal.ai) ----------

_FAL_SEEDREAM_SIZES = [
    "auto — match input (2K)",
    "1920*1920 — square (1:1)",
    "2560*1920 — landscape (4:3)",
    "1920*2560 — portrait (3:4)",
    "1536*1920 — portrait (4:5)",
    "1920*1536 — landscape (5:4)",
    "2560*1440 — landscape (16:9)",
    "1440*2560 — portrait (9:16)",
]

_FAL_SIZE_MAP = {
    "1920*1920": "square_hd",
    "2560*1920": "landscape_4_3",
    "1920*2560": "portrait_4_3",
    "1536*1920": {"width": 1536, "height": 1920},
    "1920*1536": {"width": 1920, "height": 1536},
    "2560*1440": "landscape_16_9",
    "1440*2560": "portrait_16_9",
}


class SeedreamEditFalAPINode:
    """Edit images via ByteDance Seedream 4.5 Edit on fal.ai.

    Supports 1-10 reference images for context-aware editing with
    facial feature, lighting, and color tone preservation up to 4K.
    """

    BASE_URL = "https://fal.run/fal-ai/bytedance/seedream/v4.5/edit"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fal_apikey": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "size": (_FAL_SEEDREAM_SIZES, {"default": "auto — match input (2K)"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "use_custom_size": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                **_CACHE_WIDGET,
            },
            "hidden": _HIDDEN,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, *args, **kwargs):
        return _onyx_core()['SeedreamEditFalAPINode'].edit(self, *args, **kwargs)


    def _generate(self, *args, **kwargs):
        return _onyx_core()['SeedreamEditFalAPINode']._generate(self, *args, **kwargs)



# ---------- Helpers ----------

def _png_b64(*args, **kwargs):
    return _onyx_core()['_png_b64'](*args, **kwargs)



def _encode_batch(*args, **kwargs):
    return _onyx_core()['_encode_batch'](*args, **kwargs)



def _images_to_parts(*args, **kwargs):
    return _onyx_core()['_images_to_parts'](*args, **kwargs)



def _images_to_data_uris(*args, **kwargs):
    return _onyx_core()['_images_to_data_uris'](*args, **kwargs)



def _batch_tensors(*args, **kwargs):
    return _onyx_core()['_batch_tensors'](*args, **kwargs)



def _apply_gemini_safety(*args, **kwargs):
    return _onyx_core()['_apply_gemini_safety'](*args, **kwargs)



def _gemini_response_summary(*args, **kwargs):
    return _onyx_core()['_gemini_response_summary'](*args, **kwargs)




def _largest_image_bytes(*args, **kwargs):
    return _onyx_core()['_largest_image_bytes'](*args, **kwargs)


def _extract_image_from_gemini_response(*args, **kwargs):
    return _onyx_core()['_extract_image_from_gemini_response'](*args, **kwargs)



def _image_bytes_to_tensor(*args, **kwargs):
    return _onyx_core()['_image_bytes_to_tensor'](*args, **kwargs)



# ---------- ComfyUI registration ----------

NODE_CLASS_MAPPINGS = {
    "NanoBananaProEditAPINode": NanoBananaProEditAPINode,
    "NanoBanana2EditAPINode": NanoBanana2EditAPINode,
    "SeedreamEditAPINode": SeedreamEditAPINode,
    "SeedreamEditFalAPINode": SeedreamEditFalAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaProEditAPINode": "Onyx Nano Banana Pro Edit",
    "NanoBanana2EditAPINode": "Onyx Nano Banana 2 Edit",
    "SeedreamEditAPINode": "Onyx Seedream 4.5 Edit(Wavespeed)",
    "SeedreamEditFalAPINode": "Onyx Seedream 4.5 Edit(fal.ai)",
}
