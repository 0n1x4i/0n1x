# -*- coding: utf-8 -*-
"""
Onyx phone-camera effects — original code (Onyx pack).

  Onyx Phone HDR          the "smart HDR" of recent iPhone / Samsung: lifted shadows,
                          compressed highlights (sky detail kept), local contrast and
                          the light over-sharpening halo around edges.
  Onyx Highlight Clip     sensor saturation: lamps and windows burn out, each channel
                          clips on its own (the hue drift at the edge of a burnt area),
                          purple / warm fringing and a little bloom around them.
  Onyx Lens Smudge        a dirty phone lens (fingerprint): milky veiling glare and
                          glow / smear streaks around light sources, strongest at night.
  Onyx Chroma Subsample   the colour resolution loss of a JPEG / social upload
                          (4:2:0, 4:2:2, 4:1:1): luminance kept, colour halved.

Every size is relative to the image WIDTH, so a preset looks the same at any
resolution. All four take the optional `onyx_post` input like the other post nodes.
"""

from __future__ import annotations

import math

from ..utils import onyx_post as _onyx_post
from .onyx_render_profile import ensure_profile_ready

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'onyx_phone_fx', __package__)
    return ns


_LUMA = (0.2126, 0.7152, 0.0722)


def _np(*args, **kwargs):
    return _onyx_core()['_np'](*args, **kwargs)



def _back(*args, **kwargs):
    return _onyx_core()['_back'](*args, **kwargs)



def _luma(*args, **kwargs):
    return _onyx_core()['_luma'](*args, **kwargs)



def _box(*args, **kwargs):
    return _onyx_core()['_box'](*args, **kwargs)



def _guided(*args, **kwargs):
    return _onyx_core()['_guided'](*args, **kwargs)



def _smoothstep(*args, **kwargs):
    return _onyx_core()['_smoothstep'](*args, **kwargs)



def _srgb_to_lin(*args, **kwargs):
    return _onyx_core()['_srgb_to_lin'](*args, **kwargs)



def _lin_to_srgb(*args, **kwargs):
    return _onyx_core()['_lin_to_srgb'](*args, **kwargs)



def _gauss(*args, **kwargs):
    return _onyx_core()['_gauss'](*args, **kwargs)



def _motion(*args, **kwargs):
    return _onyx_core()['_motion'](*args, **kwargs)



# ─────────────────────────────────────────────────────────────────────────────
# Phone HDR
# ─────────────────────────────────────────────────────────────────────────────

def phone_hdr(*args, **kwargs):
    return _onyx_core()['phone_hdr'](*args, **kwargs)



class OnyxPhoneHDR:
    """Smart-HDR look of recent phones: lifted shadows, held highlights, local contrast, edge halos."""

    DESCRIPTION = ("Recent iPhone / Samsung 'smart HDR': shadows lifted, bright sky / windows held back, "
                   "local contrast up and the light over-sharpening halo around edges. Edge-aware (guided "
                   "filter), sizes relative to the image width.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Mix of the whole effect."}),
                "shadow_lift": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Opens the dark areas (faces in backlight, rooms)."}),
                "highlight_compress": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05,
                                                 "tooltip": "Holds back bright areas: sky and window detail."}),
                "local_contrast": ("FLOAT", {"default": 0.3, "min": -0.5, "max": 1.5, "step": 0.05,
                                             "tooltip": "Texture / micro contrast kept on top of the flattened light."}),
                "radius": ("FLOAT", {"default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005,
                                     "tooltip": "Size of the 'light' layer, x image width."}),
                "sharpen_halo": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.5, "step": 0.05,
                                           "tooltip": "Phone over-sharpening: bright / dark halo along edges."}),
                "halo_radius": ("FLOAT", {"default": 1.5, "min": 0.3, "max": 5.0, "step": 0.1,
                                          "tooltip": "Halo width in pixels at 1024 px wide (scales with width)."}),
                "saturation": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 1.6, "step": 0.01}),
            },
            "optional": {**_onyx_post.POST_INPUT},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, *args, **kwargs):
        return _onyx_core()['OnyxPhoneHDR'].apply(self, *args, **kwargs)



# ─────────────────────────────────────────────────────────────────────────────
# Highlight clip
# ─────────────────────────────────────────────────────────────────────────────

_FRINGE = {"purple": (0.55, 0.15, 1.0), "warm": (1.0, 0.55, 0.15), "cyan": (0.15, 0.8, 1.0)}


def highlight_clip(*args, **kwargs):
    return _onyx_core()['highlight_clip'](*args, **kwargs)



class OnyxHighlightClip:
    """Burnt highlights like a real sensor: per-channel clipping, colour drift, fringing, bloom."""

    DESCRIPTION = ("Lamps, windows and flash hot spots burn out like on a phone sensor: each channel clips "
                   "on its own (so the colour drifts at the edge of the burnt area), with purple / warm "
                   "fringing and a little bloom. Only already-bright areas are pushed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.86, "min": 0.3, "max": 0.99, "step": 0.01,
                                        "tooltip": "Brightness from which areas are pushed towards clipping."}),
                "push": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                                   "tooltip": "Over-exposure of those areas, in stops."}),
                "knee": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.3, "step": 0.01,
                                   "tooltip": "0 = hard digital clip, higher = softer roll-off."}),
                "fringe": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "Coloured fringe around burnt areas."}),
                "fringe_color": (list(_FRINGE), {"default": "purple"}),
                "fringe_width": ("FLOAT", {"default": 0.004, "min": 0.001, "max": 0.02, "step": 0.001,
                                           "tooltip": "x image width."}),
                "bloom": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "bloom_radius": ("FLOAT", {"default": 0.02, "min": 0.002, "max": 0.1, "step": 0.002,
                                           "tooltip": "x image width."}),
            },
            "optional": {**_onyx_post.POST_INPUT},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, *args, **kwargs):
        return _onyx_core()['OnyxHighlightClip'].apply(self, *args, **kwargs)



# ─────────────────────────────────────────────────────────────────────────────
# Lens smudge
# ─────────────────────────────────────────────────────────────────────────────

def lens_smudge(*args, **kwargs):
    return _onyx_core()['lens_smudge'](*args, **kwargs)



class OnyxLensSmudge:
    """Dirty / greasy phone lens: veiling glare + patchy glow and smear streaks around lights."""

    DESCRIPTION = ("A fingerprint on the phone lens: milky veiling glare (blacks lifted with the scene's "
                   "light colour) and patchy glow + smear streaks around light sources. With auto_light the "
                   "effect follows how much bright light is in the image (strong at night, subtle by day).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "haze": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "Milky veil over the whole image (lifted blacks)."}),
                "light_threshold": ("FLOAT", {"default": 0.75, "min": 0.3, "max": 0.98, "step": 0.01,
                                              "tooltip": "What counts as a light source."}),
                "glow_radius": ("FLOAT", {"default": 0.04, "min": 0.005, "max": 0.2, "step": 0.005,
                                          "tooltip": "x image width."}),
                "streaks": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05,
                                      "tooltip": "Smear streaks left by the finger."}),
                "streak_angle": ("FLOAT", {"default": 20.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "streak_length": ("FLOAT", {"default": 0.12, "min": 0.02, "max": 0.4, "step": 0.01,
                                            "tooltip": "x image width."}),
                "patchiness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "Uneven grease: the glow varies across the frame."}),
                "auto_light": ("BOOLEAN", {"default": True,
                                           "tooltip": "Scale the effect with the amount of bright light."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {**_onyx_post.POST_INPUT},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, *args, **kwargs):
        return _onyx_core()['OnyxLensSmudge'].apply(self, *args, **kwargs)



# ─────────────────────────────────────────────────────────────────────────────
# Chroma subsampling
# ─────────────────────────────────────────────────────────────────────────────

_SUBSAMPLING = {"4:2:0": (2, 2), "4:2:2": (1, 2), "4:1:1": (1, 4)}


def chroma_subsample(*args, **kwargs):
    return _onyx_core()['chroma_subsample'](*args, **kwargs)



class OnyxChromaSubsample:
    """Colour resolution of a JPEG / social upload: luminance kept, chroma halved."""

    DESCRIPTION = ("The colour loss of JPEG and social uploads: the image is split into brightness + colour "
                   "(YCbCr) and the colour is stored at lower resolution (4:2:0 = Instagram / most JPEGs). "
                   "Red / saturated edges bleed slightly, exactly like a re-uploaded photo.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (list(_SUBSAMPLING), {"default": "4:2:0",
                         "tooltip": "4:2:0 = half colour resolution both ways (JPEG, Instagram); "
                                    "4:2:2 = horizontal only; 4:1:1 = quarter horizontal (old video)."}),
                "upsample": (["bilinear", "nearest"], {"default": "bilinear",
                             "tooltip": "How the decoder rebuilds the colour: bilinear (most apps) or "
                                        "nearest (blocky)."}),
            },
            "optional": {**_onyx_post.POST_INPUT},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, *args, **kwargs):
        return _onyx_core()['OnyxChromaSubsample'].apply(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {
    "OnyxPhoneHDR": OnyxPhoneHDR,
    "OnyxHighlightClip": OnyxHighlightClip,
    "OnyxLensSmudge": OnyxLensSmudge,
    "OnyxChromaSubsample": OnyxChromaSubsample,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxPhoneHDR": "Onyx Phone HDR",
    "OnyxHighlightClip": "Onyx Highlight Clip",
    "OnyxLensSmudge": "Onyx Lens Smudge",
    "OnyxChromaSubsample": "Onyx Chroma Subsample",
}
