# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""
Onyx Lens Character — the optical signature of a real lens, on GPU.

Same controls as a classic "lens / bokeh" post node, written from scratch for
Onyx in torch (runs on the GPU, whole batch at once):

  * focus map: radial, gradient (top / bottom), flat (no focus blur) or a depth map;
  * out-of-focus blur with a polygon aperture (number of blades), highlight bloom,
    longitudinal CA (red / blue blur slightly larger / smaller), field curvature
    (edge_softness) and cat-eye smearing toward the corners;
  * lateral (transverse) chromatic aberration growing toward the edges;
  * optical vignette (natural fall-off).

With focus_mode = flat and blur_strength = 0 only the lens "character" remains:
lateral CA + vignette, no blur at all.
"""

import math

import torch
import torch.nn.functional as F

from .onyx_render_profile import ensure_profile_ready

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'onyx_lens_character', __package__)
    return ns


_FOCUS_MODES = ["radial", "gradient_top", "gradient_bottom", "depth_input", "flat"]


def _luma(*args, **kwargs):
    return _onyx_core()['_luma'](*args, **kwargs)



def _radius_grid(*args, **kwargs):
    return _onyx_core()['_radius_grid'](*args, **kwargs)



def _gauss(*args, **kwargs):
    return _onyx_core()['_gauss'](*args, **kwargs)



def _aperture(*args, **kwargs):
    return _onyx_core()['_aperture'](*args, **kwargs)



def _conv_kernel(*args, **kwargs):
    return _onyx_core()['_conv_kernel'](*args, **kwargs)



def _scale_from_centre(*args, **kwargs):
    return _onyx_core()['_scale_from_centre'](*args, **kwargs)



class OnyxLensCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "focus_mode": (_FOCUS_MODES, {"default": "flat",
                    "tooltip": "Where the lens is in focus. flat = no focus blur (lens character only)."}),
                "blur_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Out-of-focus blur radius in pixels. 0 = no blur."}),
                "focus_center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_center_y": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Radial: focus centre. Gradients: sharp line. Depth: focus plane (0 near … 1 far)."}),
                "focus_size": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_feather": ("FLOAT", {"default": 0.55, "min": 0.01, "max": 2.0, "step": 0.01}),
                "aperture_blades": ("INT", {"default": 7, "min": 0, "max": 16,
                    "tooltip": "Bokeh shape: 0-2 = round, 3+ = polygon with that many blades."}),
                "cat_eye": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "long_ca": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Longitudinal CA: red / blue fringes in the blurred areas."}),
                "transverse_ca": ("FLOAT", {"default": 0.09, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Lateral CA: colour fringes growing toward the edges of the frame."}),
                "optical_vignette": ("FLOAT", {"default": 0.14, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_softness": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Field curvature: extra softness toward the edges (needs blur_strength > 0)."}),
                "bokeh_highlight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Bloom of bright highlights inside the bokeh."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Swap sharp and blurred areas."}),
            },
            "optional": {"depth_map": ("IMAGE", {"tooltip": "For focus_mode = depth_input."})},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Onyx/Post"
    DESCRIPTION = ("Real-lens character on GPU: focus blur with polygon bokeh, longitudinal and lateral "
                   "chromatic aberration, field curvature, cat-eye and optical vignette.")

    @staticmethod
    def _focus(*args, **kwargs):
        return _onyx_core()['OnyxLensCharacter']._focus(*args, **kwargs)


    def run(self, *args, **kwargs):
        return _onyx_core()['OnyxLensCharacter'].run(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {"OnyxLensCharacter": OnyxLensCharacter}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxLensCharacter": "Onyx Lens Character"}
