# -*- coding: utf-8 -*-
"""
Onyx Realism Injector — micro-shadows, eye moisture and skin oil, added after
the render to take the "plastic" edge off AI skin and eyes.

Same widget names and order as the "RealismInjector" node used in the Krea 4K
workflows (so saved values load unchanged). Original code for the Onyx pack:
no source or licence was found for the node it replaces, so nothing is copied.

Three independent effects, each with its own switch:
  micro shadows   deepens fine crevices (pores, fabric weave, lip lines): the
                  negative part of the luminance detail band, beyond a small
                  threshold, darkens the image. An optional depth map (bright =
                  near) concentrates it on the subject.
  eye moisture    YuNet finds the faces and eyes: a small catchlight in the
                  upper iris, a slightly glossier eye area, and a faint wet
                  line along the lower lid (tear meniscus).
  skin oil        specular sheen on the skin's highlights, strongest on the
                  T-zone (forehead, nose bridge), reduced on the cheeks
                  (side_matte). Without a detected face it falls back to a
                  gentle sheen on every skin-coloured highlight.
global_mix blends the result with the input (0 = off, 1 = full).
"""

from __future__ import annotations

import os

import numpy as np

from ..utils import onyx_post as _onyx_post
from .onyx_render_profile import ensure_profile_ready

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'onyx_realism_injector', __package__)
    return ns


_HERE = os.path.dirname(os.path.abspath(__file__))
_YUNET = os.path.join(_HERE, ".yunet_cache", "face_detection_yunet_2023mar.onnx")
_detectors = {}


def _faces(*args, **kwargs):
    return _onyx_core()['_faces'](*args, **kwargs)



def _gauss(*args, **kwargs):
    return _onyx_core()['_gauss'](*args, **kwargs)



def _add(*args, **kwargs):
    return _onyx_core()['_add'](*args, **kwargs)



def _maxin(*args, **kwargs):
    return _onyx_core()['_maxin'](*args, **kwargs)



def _blur(*args, **kwargs):
    return _onyx_core()['_blur'](*args, **kwargs)



def _skin_mask(*args, **kwargs):
    return _onyx_core()['_skin_mask'](*args, **kwargs)



def inject(*args, **kwargs):
    return _onyx_core()['inject'](*args, **kwargs)



class OnyxRealismInjector:
    """Micro-shadows, eye moisture and skin oil after the render (RealismInjector-compatible widgets)."""

    DESCRIPTION = ("Adds micro-shadows in fine crevices, wet eyes (catchlight, gloss, tear line) and "
                   "an oily sheen on the skin's T-zone. Faces found with YuNet; optional depth map "
                   "(bright = near) focuses the shadows on the subject.")

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo, hi, st: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": st})
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_micro_shadows": ("BOOLEAN", {"default": True}),
                "shadow_strength": f(0.3, 0.0, 1.0, 0.05),
                "shadow_radius": f(2.0, 0.5, 10.0, 0.1),
                "shadow_detail_threshold": f(0.015, 0.0, 0.2, 0.001),
                "enable_eye_moisture": ("BOOLEAN", {"default": True}),
                "moisture_strength": f(0.7, 0.0, 1.0, 0.05),
                "catchlight_intensity": f(0.8, 0.0, 1.0, 0.05),
                "tear_meniscus": f(0.6, 0.0, 1.0, 0.05),
                "enable_skin_oil": ("BOOLEAN", {"default": True}),
                "oil_strength": f(0.8, 0.0, 1.0, 0.05),
                "t_zone_gloss": f(0.7, 0.0, 1.0, 0.05),
                "side_matte": f(0.5, 0.0, 1.0, 0.05),
                "global_mix": f(1.0, 0.0, 1.0, 0.05),
            },
            "optional": {"depth_map": ("IMAGE", {"tooltip": "Optional depth map, bright = near."}),
                         **_onyx_post.POST_INPUT},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, *args, **kwargs):
        return _onyx_core()['OnyxRealismInjector'].apply(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {"OnyxRealismInjector": OnyxRealismInjector}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxRealismInjector": "Onyx Realism Injector"}
