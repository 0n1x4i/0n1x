# -*- coding: utf-8 -*-
"""
Onyx Camera Shake — the Candid Snap "shake.py" pass as a node.

Real-photo distortion after the render: camera-shake motion blur + double-image
ghost + soft defocus (+ optional dutch tilt), sized relative to the image WIDTH
so it matches blurry phone references at any resolution. Blur written into the
prompt gets sharpened away by the detail stages or turns the face to mush, so it
is added here, to the finished image.

Six presets (soft, shake_med, shake_med_h, shake_med_v, shake_med_rot, shake_heavy).

`strength` scales lengths, offsets, sigma and tilt (1 = shake.py exactly).
Onyx Studio's `post` output (or Onyx Post Style) can choose the preset per image.
"""

from __future__ import annotations

import math

from ..utils import onyx_post as _onyx_post
from .onyx_render_profile import ensure_profile_ready

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'onyx_camera_shake', __package__)
    return ns


PRESET_NAMES = ["none", "soft", "shake_med", "shake_med_h", "shake_med_v", "shake_med_rot", "shake_heavy"]


def _motion_kernel(*args, **kwargs):
    return _onyx_core()['_motion_kernel'](*args, **kwargs)



def shake_params(*args, **kwargs):
    return _onyx_core()['shake_params'](*args, **kwargs)


def shake(*args, **kwargs):
    return _onyx_core()['shake'](*args, **kwargs)



class OnyxCameraShake:
    """Phone camera shake / softness after rendering (the Candid Snap shake.py presets)."""

    DESCRIPTION = (
        "Phone camera shake after rendering: motion blur, double image, defocus and tilt, width-relative. "
        "The values shown are used as they are; when Onyx Studio (post / post_preset) sends a motion preset "
        "other than 'manual', the preset replaces them for that image ('none' = no shake)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo, hi, st, tip: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": st, "tooltip": tip})  # noqa: E731
        return {
            "required": {
                "images": ("IMAGE",),
                "length": f(0.0, 0.0, 0.05, 0.001, "Motion-blur length, as a fraction of the image width."),
                "angle": f(0.0, 0.0, 180.0, 1.0, "Direction of the shake, in degrees (0 = horizontal)."),
                "ghost_offset": f(0.0, 0.0, 0.05, 0.001, "Double-image offset, as a fraction of the width."),
                "ghost_alpha": f(0.0, 0.0, 0.6, 0.01, "Double-image opacity."),
                "softness": f(0.0, 0.0, 0.005, 0.0001, "Gaussian defocus, as a fraction of the width."),
                "rotate": f(0.0, -15.0, 15.0, 0.5, "Dutch tilt, in degrees."),
                "strength": f(1.0, 0.0, 2.0, 0.05, "Scales everything above."),
            },
            "optional": {
                "preset_override": ("STRING", {"forceInput": True,
                    "tooltip": "A motion preset name as text (e.g. Onyx Studio's post_preset). A preset "
                               "replaces the values above for this image; 'none' = no shake; empty "
                               "(Studio motion 'manual') = the values above."}),
                **_onyx_post.POST_INPUT,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, *args, **kwargs):
        return _onyx_core()['OnyxCameraShake'].apply(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {"OnyxCameraShake": OnyxCameraShake}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxCameraShake": "Onyx Camera Shake"}
