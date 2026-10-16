"""
Onyx Camera Look — pipeline complet, zero dependance externe (numpy + PIL uniquement).
"""
import io
import random
import math
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

import os as _onyx_os, importlib.util as _onyx_ilu

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'camera_look', __package__)
    return ns

_onyx_prof_path = _onyx_os.path.join(_onyx_os.path.dirname(_onyx_os.path.dirname(_onyx_os.path.abspath(__file__))), "nodes", "onyx_render_profile.py")
_onyx_prof_spec = _onyx_ilu.spec_from_file_location("onyx_render_profile_pp", _onyx_prof_path)
_onyx_prof_mod = _onyx_ilu.module_from_spec(_onyx_prof_spec)
_onyx_prof_spec.loader.exec_module(_onyx_prof_mod)
ensure_profile_ready = _onyx_prof_mod.ensure_profile_ready
def _noise_params(*args, **kwargs):
    return _onyx_core()['_noise_params'](*args, **kwargs)



def _gaussian_kernel_1d(*args, **kwargs):
    return _onyx_core()['_gaussian_kernel_1d'](*args, **kwargs)



def _gaussian_blur_np(*args, **kwargs):
    return _onyx_core()['_gaussian_blur_np'](*args, **kwargs)



def _box_blur_1d(*args, **kwargs):
    return _onyx_core()['_box_blur_1d'](*args, **kwargs)



def _shift_channel(*args, **kwargs):
    return _onyx_core()['_shift_channel'](*args, **kwargs)



def _bayer_demosaic_effect(*args, **kwargs):
    return _onyx_core()['_bayer_demosaic_effect'](*args, **kwargs)



def _motion_blur(*args, **kwargs):
    return _onyx_core()['_motion_blur'](*args, **kwargs)



def _simulate_camera_pipeline(*args, **kwargs):
    return _onyx_core()['_simulate_camera_pipeline'](*args, **kwargs)



# ---------------------------------------------------------------------------
# Node ComfyUI
# ---------------------------------------------------------------------------

class Onyx_Camera_Look:
    """
    Onyx Camera Look — pipeline camera physiquement inspire.
    Ordre : bruit capteur -> demosaic -> motion blur -> JPEG.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on":  "Enabled",
                    "label_off": "Bypassed",
                }),

                "demosaic_pixel_blur": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Bayer pixel displacement (R/B channels shifted diagonally, not a blur). 0 = disabled, 1 = natural ~1px sensor offset, higher = stronger.",
                }),
                "noise_strength": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Sensor noise level. 0 = none | 3 = standard | 5 = heavy.",
                }),
                "kernel_motion_blur": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 51.0, "step": 0.1,
                    "tooltip": "Motion blur kernel width (px, box blur — same process as before). 1 = none.",
                }),
                "jpeg_compression": ("INT", {
                    "default": 98, "min": 85, "max": 100, "step": 1,
                    "tooltip": "JPEG quality. 100 = near-lossless, 85 = visible artifacts.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "execute"
    CATEGORY      = "Onyx/Post-Processing"
    DESCRIPTION   = (
        "Onyx Camera Look\n"
        "Complete camera pipeline: sensor noise -> demosaic -> motion blur -> JPEG.\n"
        "Random seed per frame. No external dependencies (numpy + PIL only)."
    )

    def execute(self, *args, **kwargs):
        return _onyx_core()['Onyx_Camera_Look'].execute(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {
    "Onyx_Camera_Look": Onyx_Camera_Look,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Onyx_Camera_Look": "Onyx Camera Look",
}
