# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""
Onyx Depth Colormap.

Drop-in replacement for "Video Depth Anything Output Depth Map" with many more
colormaps. It takes the DEPTHS output of "Video Depth Anything Process" and
returns an IMAGE batch, normalised over the WHOLE clip (same depth = same colour
on every frame, no flicker).

Besides gray / inferno it offers a few matplotlib maps and four "vivid" ramps
(green, magenta, blue, orange) that need no extra library. In a head-swap guide,
a saturated colour that does not exist in the scene is the easiest region marker
for the model to recognise as "replace this".
"""

import numpy as np
import torch

# Rampes vives, sans dependance : (position, (r, g, b)) en 0-1, interpolation lineaire.
_VIVID_STOPS = {
    "onyx_green":   [(0.0, (0.00, 0.10, 0.00)), (0.5, (0.05, 0.60, 0.05)), (1.0, (0.35, 1.00, 0.35))],
    "onyx_magenta": [(0.0, (0.15, 0.00, 0.15)), (0.5, (0.65, 0.00, 0.65)), (1.0, (1.00, 0.35, 1.00))],
    "onyx_blue":    [(0.0, (0.00, 0.00, 0.20)), (0.5, (0.05, 0.25, 0.85)), (1.0, (0.35, 0.65, 1.00))],
    "onyx_orange":  [(0.0, (0.20, 0.05, 0.00)), (0.5, (0.85, 0.35, 0.00)), (1.0, (1.00, 0.70, 0.20))],
}

_MPL_MAPS = ["inferno", "magma", "plasma", "viridis", "cividis", "turbo", "jet",
             "Greens", "Blues", "Reds", "Oranges", "Purples"]

COLORMAPS = ["gray"] + list(_VIVID_STOPS.keys()) + _MPL_MAPS


def _vivid_lut(name):
    stops = _VIVID_STOPS[name]
    xs = np.array([p for p, _ in stops], dtype=np.float32)
    lut = np.zeros((256, 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    for c in range(3):
        ys = np.array([rgb[c] for _, rgb in stops], dtype=np.float32)
        lut[:, c] = np.interp(t, xs, ys)
    return lut


def _mpl_lut(name):
    try:
        import matplotlib
        cmap = matplotlib.colormaps[name]
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(name)
    return np.asarray(cmap(np.linspace(0.0, 1.0, 256))[:, :3], dtype=np.float32)


def _lut_for(name):
    if name in _VIVID_STOPS:
        return _vivid_lut(name)
    return _mpl_lut(name)


class OnyxDepthColormap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depths": ("DEPTHS",),
                "colormap": (COLORMAPS, {"default": "gray",
                    "tooltip": "gray / inferno behave like the original node. onyx_* are saturated "
                               "ramps (no extra library needed). The others come from matplotlib."}),
            },
            "optional": {
                # Ajoute EN DERNIER : les widgets sont positionnels.
                "brightness_floor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.01,
                    "tooltip": "Lifts the darkest depth values so the far parts of the region are not "
                               "pure black (0 = original behaviour)."}),
                "invert": ("BOOLEAN", {"default": False,
                    "tooltip": "Swap near and far (near = dark, far = bright)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "output"
    CATEGORY = "Onyx"
    DESCRIPTION = "Colourise a Video Depth Anything depth map with many more colormaps than the original node."

    def output(self, depths, colormap, brightness_floor=0.0, invert=False):
        depths = np.asarray(depths, dtype=np.float32)
        d_min, d_max = float(depths.min()), float(depths.max())
        norm = (depths - d_min) / (d_max - d_min + 1e-6)
        if invert:
            norm = 1.0 - norm
        if brightness_floor > 0.0:
            norm = brightness_floor + (1.0 - brightness_floor) * norm
        norm = np.clip(norm, 0.0, 1.0)

        if colormap == "gray":
            out = np.repeat(norm[..., None], 3, axis=-1)
        else:
            lut = _lut_for(colormap)
            idx = (norm * 255.0).astype(np.uint8)
            out = lut[idx]
        return (torch.from_numpy(np.ascontiguousarray(out.astype(np.float32))),)
