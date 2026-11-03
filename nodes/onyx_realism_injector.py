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

_HERE = os.path.dirname(os.path.abspath(__file__))
_YUNET = os.path.join(_HERE, ".yunet_cache", "face_detection_yunet_2023mar.onnx")
_detectors = {}


def _faces(rgb_u8):
    """YuNet faces: list of (x, y, w, h, landmarks[5][2]). [] when unavailable."""
    try:
        import cv2
        from .nano_banana_aio import _ensure_yunet_model
        if not _ensure_yunet_model(_YUNET):
            return []
        h, w = rgb_u8.shape[:2]
        scale = min(1.0, 1280.0 / max(h, w))
        small = cv2.resize(rgb_u8, (int(w * scale), int(h * scale))) if scale < 1 else rgb_u8
        key = small.shape[:2]
        det = _detectors.get(key)
        if det is None:
            det = cv2.FaceDetectorYN.create(_YUNET, "", (small.shape[1], small.shape[0]), 0.7, 0.3, 50)
            _detectors[key] = det
        _, found = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        out = []
        for f in (found if found is not None else []):
            x, y, fw, fh = (f[:4] / scale).tolist()
            lm = (f[4:14].reshape(5, 2) / scale)
            out.append((x, y, fw, fh, lm))
        return out
    except Exception as e:
        print(f"[Onyx Realism Injector] face detection unavailable: {e}")
        return []


def _gauss(shape, cx, cy, sx, sy):
    """Local gaussian: (slice_y, slice_x, g) covering +-3.5 sigma only, so a 4K
    image does not allocate a full-size grid for a 20 px catchlight."""
    import numpy as np
    h, w = shape
    sx, sy = max(sx, 0.5), max(sy, 0.5)
    x0, x1 = max(0, int(cx - 3.5 * sx)), min(w, int(cx + 3.5 * sx) + 1)
    y0, y1 = max(0, int(cy - 3.5 * sy)), min(h, int(cy + 3.5 * sy) + 1)
    if x0 >= x1 or y0 >= y1:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), np.float32)
    xs = ((np.arange(x0, x1, dtype=np.float32) - cx) / sx) ** 2
    ys = ((np.arange(y0, y1, dtype=np.float32) - cy) / sy) ** 2
    g = np.exp(-(ys[:, None] + xs[None, :]) / 2.0)
    return slice(y0, y1), slice(x0, x1), g


def _add(target, gauss, amount):
    """target[window] += g * amount (amount: scalar, or HxWx3 array sampled in the window)."""
    sy, sx, g = gauss
    if g.size == 0:
        return
    if target.ndim == 3:
        a = amount[sy, sx] if hasattr(amount, "ndim") and amount.ndim == 3 else amount
        target[sy, sx] += g[..., None] * a
    else:
        target[sy, sx] += g * amount


def _maxin(target, gauss, scale=1.0):
    sy, sx, g = gauss
    if g.size:
        target[sy, sx] = np.maximum(target[sy, sx], g * scale)


def _blur(arr, sigma):
    import cv2
    return cv2.GaussianBlur(arr, (0, 0), max(0.3, sigma))


def _skin_mask(rgb):
    import cv2
    import numpy as np
    ycc = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr, cb = ycc[..., 1], ycc[..., 2]
    m = ((cr > 133) & (cr < 175) & (cb > 77) & (cb < 130)).astype(np.float32)
    return _blur(m, 3.0)


def inject(rgb, depth, p):
    """rgb: float32 HxWx3 0-1 (numpy). Returns the processed copy."""
    import numpy as np
    out = rgb.copy()
    h, w = rgb.shape[:2]
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    faces = _faces((rgb * 255).astype(np.uint8)) if (p["enable_eye_moisture"] or p["enable_skin_oil"]) else []

    if p["enable_micro_shadows"] and p["shadow_strength"] > 0:
        radius = max(0.5, p["shadow_radius"] * max(h, w) / 2000.0)
        detail = luma - _blur(luma, radius)
        dark = np.clip(-detail - p["shadow_detail_threshold"], 0, None)
        dark = dark / (dark.max() + 1e-6)
        weight = np.ones_like(luma)
        if depth is not None:
            weight = 0.35 + 0.65 * np.clip(depth, 0, 1)
        out *= (1.0 - 0.35 * p["shadow_strength"] * dark * weight)[..., None]

    if p["enable_eye_moisture"] and faces:
        sharp = None
        for (x, y, fw, fh, lm) in faces:
            r = 0.085 * fw
            if sharp is None:
                sharp = rgb - _blur(rgb, max(0.5, r * 0.15))
            for ex, ey in lm[:2]:
                eye = _gauss((h, w), ex, ey, r * 0.9, r * 0.55)
                _add(out, eye, 0.6 * p["moisture_strength"] * sharp)
                _add(out, eye, 0.04 * p["moisture_strength"])
                _add(out, _gauss((h, w), ex - 0.22 * r, ey - 0.28 * r, r * 0.07, r * 0.07),
                     0.55 * p["catchlight_intensity"])
                _add(out, _gauss((h, w), ex, ey + 0.42 * r, r * 0.55, r * 0.05), 0.07 * p["tear_meniscus"])

    if p["enable_skin_oil"] and p["oil_strength"] > 0:
        skin = _skin_mask(rgb)
        spec = np.clip((luma - 0.5) / 0.5, 0, 1) ** 2
        zone = np.zeros_like(luma)
        if faces:
            for (x, y, fw, fh, lm) in faces:
                (rex, rey), (lex, ley), (nx, ny) = lm[0], lm[1], lm[2]
                mx, my = (rex + lex) / 2, (rey + ley) / 2
                tg = p["t_zone_gloss"]
                _maxin(zone, _gauss((h, w), mx, my - 0.33 * fh, 0.28 * fw, 0.12 * fh), tg)          # forehead
                _maxin(zone, _gauss((h, w), (mx + nx) / 2, (my + ny) / 2, 0.06 * fw, 0.16 * fh), tg)  # nose bridge
                _maxin(zone, _gauss((h, w), nx, ny, 0.07 * fw, 0.06 * fh), tg)                      # nose tip
                side = (1 - p["side_matte"]) * 0.5
                _maxin(zone, _gauss((h, w), rex, ny, 0.12 * fw, 0.10 * fh), side)                  # cheeks
                _maxin(zone, _gauss((h, w), lex, ny, 0.12 * fw, 0.10 * fh), side)
        else:
            zone = np.full_like(luma, 0.4)
        sheen = 0.22 * p["oil_strength"] * skin * spec * zone
        out = out + sheen[..., None] * (1.0 - out) * np.array([1.0, 0.98, 0.95], dtype=np.float32)

    out = np.clip(out, 0, 1)
    mix = float(np.clip(p["global_mix"], 0, 1))
    return rgb * (1 - mix) + out * mix


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

    def apply(self, image, depth_map=None, onyx_post=None, **params):
        ensure_profile_ready()
        params, bypass = _onyx_post.apply("OnyxRealismInjector", params, onyx_post)
        if bypass:
            return (image,)
        import numpy as np
        import torch
        out = []
        for i in range(image.shape[0]):
            rgb = image[i, ..., :3].detach().cpu().numpy().astype(np.float32)
            depth = None
            if depth_map is not None:
                d = depth_map[min(i, depth_map.shape[0] - 1)].detach().cpu().numpy().astype(np.float32)
                d = d.mean(-1) if d.ndim == 3 else d
                if d.shape != rgb.shape[:2]:
                    import cv2
                    d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]))
                depth = d
            out.append(torch.from_numpy(inject(rgb, depth, params)))
        return (torch.stack(out, 0).to(image.dtype),)


NODE_CLASS_MAPPINGS = {"OnyxRealismInjector": OnyxRealismInjector}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxRealismInjector": "Onyx Realism Injector"}
