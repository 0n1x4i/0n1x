# -*- coding: utf-8 -*-
# Onyx — light tools: Curves, Levels, Contrast.
#
# Onyx Curves Adjust and Onyx Levels Adjust are a FORK of "Curves Adjust Pro" and
# "Levels Adjust" from comfyui-machinepainting-nodes, used under the MIT License:
#
#   MIT License — Copyright (c) 2025 MachinePainting
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the Software
#   without restriction, including without limitation the rights to use, copy, modify, merge,
#   publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#   to whom the Software is furnished to do so, subject to the following conditions:
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
#
# Onyx changes: whole batches (the originals process only the first image), curve presets
# applied when the curve is untouched, `onyx_post` input, Post Studio "Light" stages.
# Onyx Contrast is Onyx's own: the contrast step of the realism-enhancor MoreJPEG node
# (PIL ImageEnhance.Contrast), alone, so it can sit with the light tools.
"""Curves / Levels / Contrast — the light tools (first stages of Onyx Post Studio)."""

from __future__ import annotations

import json

import numpy as np
import torch

CURVE_PRESETS = {'none': {'rgb': [[0, 0], [1, 1]], 'red': [[0, 0], [1, 1]], 'green': [[0, 0], [1, 1]], 'blue': [[0, 0], [1, 1]]}, 'auto_enhance': {'rgb': [[0, 0], [0.25, 0.22], [0.5, 0.5], [0.75, 0.78], [1, 1]], 'red': [[0, 0], [1, 1]], 'green': [[0, 0], [1, 1]], 'blue': [[0, 0], [1, 1]]}, 'high_contrast': {'rgb': [[0, 0], [0.25, 0.15], [0.5, 0.5], [0.75, 0.85], [1, 1]], 'red': [[0, 0], [1, 1]], 'green': [[0, 0], [1, 1]], 'blue': [[0, 0], [1, 1]]}, 'low_contrast': {'rgb': [[0, 0.1], [0.5, 0.5], [1, 0.9]], 'red': [[0, 0], [1, 1]], 'green': [[0, 0], [1, 1]], 'blue': [[0, 0], [1, 1]]}, 'warm_tones': {'rgb': [[0, 0], [1, 1]], 'red': [[0, 0], [0.5, 0.55], [1, 1]], 'green': [[0, 0], [1, 1]], 'blue': [[0, 0], [0.5, 0.45], [1, 1]]}, 'cool_tones': {'rgb': [[0, 0], [1, 1]], 'red': [[0, 0], [0.5, 0.45], [1, 1]], 'green': [[0, 0], [1, 1]], 'blue': [[0, 0], [0.5, 0.55], [1, 1]]}, 'vintage_fade': {'rgb': [[0, 0.05], [1, 0.95]], 'red': [[0, 0.02], [0.5, 0.52], [1, 1]], 'green': [[0, 0], [0.5, 0.48], [1, 0.95]], 'blue': [[0, 0.05], [0.5, 0.45], [1, 0.9]]}, 'cinematic': {'rgb': [[0, 0.03], [0.5, 0.5], [1, 0.97]], 'red': [[0, 0], [0.5, 0.48], [1, 1]], 'green': [[0, 0], [0.5, 0.5], [1, 0.98]], 'blue': [[0, 0.05], [0.5, 0.52], [1, 1]]}, 'matte_look': {'rgb': [[0, 0.08], [0.25, 0.28], [0.75, 0.75], [1, 0.92]], 'red': [[0, 0], [1, 1]], 'green': [[0, 0], [1, 1]], 'blue': [[0, 0], [1, 1]]}, 'cross_process': {'rgb': [[0, 0], [1, 1]], 'red': [[0, 0.05], [0.5, 0.45], [1, 1]], 'green': [[0, 0], [0.5, 0.55], [1, 0.95]], 'blue': [[0, 0.1], [0.5, 0.4], [1, 0.9]]}, 'sepia_tone': {'rgb': [[0, 0], [0.5, 0.5], [1, 0.95]], 'red': [[0, 0], [0.5, 0.55], [1, 1]], 'green': [[0, 0], [0.5, 0.48], [1, 0.9]], 'blue': [[0, 0], [0.5, 0.35], [1, 0.8]]}, 'bleach_bypass': {'rgb': [[0, 0], [0.2, 0.12], [0.5, 0.5], [0.8, 0.88], [1, 1]], 'red': [[0, 0], [1, 1]], 'green': [[0, 0], [1, 1]], 'blue': [[0, 0], [0.5, 0.45], [1, 1]]}, 'golden_hour': {'rgb': [[0, 0], [0.5, 0.52], [1, 1]], 'red': [[0, 0], [0.5, 0.55], [1, 1]], 'green': [[0, 0], [0.5, 0.52], [1, 0.98]], 'blue': [[0, 0], [0.5, 0.4], [1, 0.85]]}, 'moonlight': {'rgb': [[0, 0], [0.5, 0.48], [1, 0.95]], 'red': [[0, 0], [0.5, 0.45], [1, 0.9]], 'green': [[0, 0], [0.5, 0.5], [1, 0.95]], 'blue': [[0, 0.05], [0.5, 0.55], [1, 1]]}, 'vibrant_pop': {'rgb': [[0, 0], [0.2, 0.15], [0.5, 0.5], [0.8, 0.85], [1, 1]], 'red': [[0, 0], [0.5, 0.52], [1, 1]], 'green': [[0, 0], [0.5, 0.52], [1, 1]], 'blue': [[0, 0], [0.5, 0.52], [1, 1]]}, 'noir': {'rgb': [[0, 0], [0.15, 0.05], [0.5, 0.5], [0.85, 0.95], [1, 1]], 'red': [[0, 0], [0.5, 0.48], [1, 1]], 'green': [[0, 0], [0.5, 0.48], [1, 1]], 'blue': [[0, 0], [0.5, 0.48], [1, 1]]}, 'sunset_glow': {'rgb': [[0, 0], [1, 1]], 'red': [[0, 0], [0.5, 0.58], [1, 1]], 'green': [[0, 0], [0.5, 0.48], [1, 0.95]], 'blue': [[0, 0], [0.5, 0.38], [1, 0.8]]}, 'forest_green': {'rgb': [[0, 0.02], [0.5, 0.5], [1, 0.98]], 'red': [[0, 0], [0.5, 0.45], [1, 0.95]], 'green': [[0, 0], [0.5, 0.55], [1, 1]], 'blue': [[0, 0], [0.5, 0.45], [1, 0.9]]}}

IDENTITY = {"rgb": [[0, 0], [1, 1]], "red": [[0, 0], [1, 1]], "green": [[0, 0], [1, 1]], "blue": [[0, 0], [1, 1]]}
CHANNELS = ("rgb", "red", "green", "blue")


def _spline_lut(points, num_samples=256):
    """Smooth curve using weighted tangent averaging - no S-curves."""
    if len(points) < 2:
        return np.linspace(0, 1, num_samples)

    points = sorted(points, key=lambda p: p[0])

    # Extend to endpoints
    if points[0][0] > 0.001:
        points.insert(0, [0, points[0][1]])
    if points[-1][0] < 0.999:
        points.append([1, points[-1][1]])

    n = len(points)

    if n == 2:
        lut = np.zeros(num_samples)
        for i in range(num_samples):
            t = i / (num_samples - 1)
            lut[i] = points[0][1] + (points[1][1] - points[0][1]) * t
        return lut

    # Calculate tangents using weighted average
    tangents = []
    for i in range(n):
        if i == 0:
            tangents.append((points[1][1] - points[0][1]) / (points[1][0] - points[0][0]))
        elif i == n - 1:
            tangents.append((points[n-1][1] - points[n-2][1]) / (points[n-1][0] - points[n-2][0]))
        else:
            dx0 = points[i][0] - points[i-1][0]
            dx1 = points[i+1][0] - points[i][0]
            dy0 = points[i][1] - points[i-1][1]
            dy1 = points[i+1][1] - points[i][1]
            s0 = dy0 / dx0
            s1 = dy1 / dx1
            tangents.append((s0 * dx1 + s1 * dx0) / (dx0 + dx1))

    lut = np.zeros(num_samples)

    for i in range(num_samples):
        x = i / (num_samples - 1)

        # Find segment
        seg = 0
        for j in range(n - 1):
            if points[j][0] <= x <= points[j + 1][0]:
                seg = j
                break
            if j == n - 2:
                seg = j

        p1 = points[seg]
        p2 = points[seg + 1]
        h = p2[0] - p1[0]
        t = (x - p1[0]) / h if h > 0.0001 else 0

        m1 = tangents[seg] * h
        m2 = tangents[seg + 1] * h

        t2 = t * t
        t3 = t2 * t

        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2

        y = h00 * p1[1] + h10 * m1 + h01 * p2[1] + h11 * m2

        lut[i] = np.clip(y, 0, 1)

    return lut


def _is_identity(curves: dict) -> bool:
    for ch in CHANNELS:
        pts = curves.get(ch) or []
        if pts and sorted(map(tuple, pts)) != [(0, 0), (1, 1)]:
            return False
    return True


def _mask_for(mask, index, h, w, device):
    if mask is None:
        return None
    m = mask if mask.ndim == 3 else mask.unsqueeze(0)
    m = m[min(index, m.shape[0] - 1)].to(device=device, dtype=torch.float32)
    if m.shape != (h, w):
        m = torch.nn.functional.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return m.clamp(0, 1)


class OnyxCurvesAdjust:
    """Photoshop-style curves (fork of MachinePainting's Curves Adjust Pro, MIT)."""

    DESCRIPTION = ("Curves per channel (RGB, R, G, B) with an interactive editor. Click to add a point, drag to "
                   "move, shift+click to remove. A preset is used while the curve is untouched. "
                   "Fork of Curves Adjust Pro (comfyui-machinepainting-nodes, MIT).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "curve_data": ("STRING", {"default": json.dumps(IDENTITY, separators=(",", ":")), "multiline": False}),
                "preset": (list(CURVE_PRESETS), {"default": "none"}),
            },
            "optional": {
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_curves"
    CATEGORY = "Onyx/Post"

    def apply_curves(self, image, curve_data, preset="none", mask=None, invert_mask=False):
        try:
            curves = json.loads(curve_data) if curve_data else {}
        except (TypeError, ValueError):
            curves = {}
        if _is_identity(curves) and preset != "none":
            curves = CURVE_PRESETS.get(preset, curves)
        luts = {}
        for ch in CHANNELS:
            pts = curves.get(ch) or []
            if len(pts) >= 2 and sorted(map(tuple, pts)) != [(0, 0), (1, 1)]:
                luts[ch] = torch.from_numpy(_spline_lut([list(p) for p in pts]).astype(np.float32))
        if not luts:
            return (image,)
        out = []
        for i in range(image.shape[0]):
            img = image[i, ..., :3].float()
            dev = img.device
            res = img.clone()
            idx = (res * 255).long().clamp(0, 255)
            if "rgb" in luts:
                res = luts["rgb"].to(dev)[idx]
                idx = (res * 255).long().clamp(0, 255)
            for c, ch in enumerate(("red", "green", "blue")):
                if ch in luts:
                    res[..., c] = luts[ch].to(dev)[idx[..., c]]
            m = _mask_for(mask, i, img.shape[0], img.shape[1], dev)
            if m is not None:
                if invert_mask:
                    m = 1.0 - m
                res = img * (1 - m[..., None]) + res * m[..., None]
            res = res.clamp(0, 1)
            if image.shape[-1] > 3:
                res = torch.cat([res, image[i, ..., 3:].float()], -1)
            out.append(res)
        return (torch.stack(out, 0).to(image.dtype),)


class OnyxLevelsAdjust:
    """Black / white point, gamma, output levels (fork of MachinePainting's Levels Adjust, MIT)."""

    DESCRIPTION = ("Input black / white point, gamma and output levels. "
                   "Fork of Levels Adjust (comfyui-machinepainting-nodes, MIT).")

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo, hi, st: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": st, "display": "slider"})  # noqa: E731
        return {"required": {"image": ("IMAGE",)},
                "optional": {"black_point": f(0.0, 0.0, 0.5, 0.01), "white_point": f(1.0, 0.5, 1.0, 0.01),
                             "gamma": f(1.0, 0.1, 3.0, 0.05), "output_black": f(0.0, 0.0, 0.5, 0.01),
                             "output_white": f(1.0, 0.5, 1.0, 0.01)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_levels"
    CATEGORY = "Onyx/Post"

    def adjust_levels(self, image, black_point=0.0, white_point=1.0, gamma=1.0, output_black=0.0, output_white=1.0):
        if (black_point, white_point, gamma, output_black, output_white) == (0.0, 1.0, 1.0, 0.0, 1.0):
            return (image,)
        rgb = image[..., :3].float()
        res = ((rgb - black_point) / max(white_point - black_point, 1e-6)).clamp(0, 1)
        if gamma != 1.0:
            res = res.pow(1.0 / gamma)
        res = (res * (output_white - output_black) + output_black).clamp(0, 1)
        if image.shape[-1] > 3:
            res = torch.cat([res, image[..., 3:].float()], -1)
        return (res.to(image.dtype),)


class OnyxContrast:
    """Global contrast exactly like the realism-enhancor MoreJPEG node (PIL ImageEnhance.Contrast)."""

    DESCRIPTION = ("Global contrast around the image's mean grey, like the contrast setting of the MoreJPEG "
                   "(realism-enhancor) node: 1 = unchanged, below 1 = flatter, above 1 = punchier.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, image, contrast=1.0):
        if abs(contrast - 1.0) < 1e-6:
            return (image,)
        rgb = image[..., :3].float()
        q = (rgb * 255).round()                                   # PIL works on 8-bit
        grey = (0.299 * q[..., 0] + 0.587 * q[..., 1] + 0.114 * q[..., 2]).floor()
        mean = torch.floor(grey.mean(dim=(1, 2), keepdim=True) + 0.5)[..., None]   # per image, like PIL
        res = (mean + contrast * (q - mean)).round().clamp(0, 255) / 255.0
        if image.shape[-1] > 3:
            res = torch.cat([res, image[..., 3:].float()], -1)
        return (res.to(image.dtype),)


NODE_CLASS_MAPPINGS = {"OnyxCurvesAdjust": OnyxCurvesAdjust, "OnyxLevelsAdjust": OnyxLevelsAdjust,
                       "OnyxContrast": OnyxContrast}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxCurvesAdjust": "Onyx Curves Adjust (fork of MachinePainting)",
                              "OnyxLevelsAdjust": "Onyx Levels Adjust (fork of MachinePainting)",
                              "OnyxContrast": "Onyx Contrast"}
