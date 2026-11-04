# -*- coding: utf-8 -*-
"""
Onyx More JPEG — contrast / sharpness / JPEG re-encode finishing pass.

Same widget names and order as the "MoreJPEG" node used in the Krea 4K
workflows (so their saved values load unchanged), written from scratch for
the Onyx pack: the original repository publishes no licence, so none of its
code is used here.

Order of operations, per pass:
  pixelate -> blur -> noise -> posterize -> contrast -> sharpness -> effects
  (grayscale, invert, emboss, edge enhance) -> JPEG re-encode (when < 100)
`passes` repeats the whole chain (generation loss, like re-sharing a photo).

Recipes set the main values in one click ("custom" = the widgets as typed):
lite, noise, artifact_hell, pro_plus.
"""

from __future__ import annotations

import io

from ..utils import onyx_post as _onyx_post
from .onyx_render_profile import ensure_profile_ready

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'onyx_more_jpeg', __package__)
    return ns


RECIPE_NAMES = ["custom", "lite", "noise", "artifact_hell", "pro_plus"]



def _process(*args, **kwargs):
    return _onyx_core()['_process'](*args, **kwargs)



class OnyxMoreJPEG:
    """Contrast, sharpness, noise and JPEG re-encode finish (MoreJPEG-compatible widgets)."""

    DESCRIPTION = ("Finishing pass: contrast, sharpness, noise, posterize, blur, pixelate and real JPEG "
                   "re-encoding, repeated `passes` times. Widget-compatible with MoreJPEG.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "recipe": (RECIPE_NAMES, {"default": "custom"}),
                "jpeg_quality": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1,
                                         "tooltip": "100 = no re-encode."}),
                "passes": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "noise_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0,
                                          "tooltip": "Gaussian noise, in 0-255 levels."}),
                "sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                        "tooltip": "1 = unchanged."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                       "tooltip": "1 = unchanged."}),
                "posterize_bits": ("INT", {"default": 8, "min": 1, "max": 8, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "pixelate_box": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
                "grayscale": ("BOOLEAN", {"default": False}),
                "emboss": ("BOOLEAN", {"default": False}),
                "edge_enhance": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": dict(_onyx_post.POST_INPUT),
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, *args, **kwargs):
        return _onyx_core()['OnyxMoreJPEG'].apply(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {"OnyxMoreJPEG": OnyxMoreJPEG}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxMoreJPEG": "Onyx More JPEG"}
