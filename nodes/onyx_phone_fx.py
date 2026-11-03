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

_LUMA = (0.2126, 0.7152, 0.0722)


def _np(img_t):
    import numpy as np
    return img_t[..., :3].detach().cpu().float().numpy().astype(np.float32)


def _back(arrs, like):
    import numpy as np
    import torch
    out = torch.from_numpy(np.clip(np.stack(arrs, 0), 0.0, 1.0))
    if like.shape[-1] == 4:                         # keep an alpha channel untouched
        out = torch.cat([out, like[..., 3:].detach().cpu().float()], -1)
    return out.to(like.dtype)


def _luma(rgb):
    return rgb[..., 0] * _LUMA[0] + rgb[..., 1] * _LUMA[1] + rgb[..., 2] * _LUMA[2]


def _box(x, r):
    import cv2
    k = max(1, int(r)) * 2 + 1
    return cv2.boxFilter(x, -1, (k, k), borderType=cv2.BORDER_REFLECT)


def _guided(guide, src, r, eps):
    """He et al. guided filter (edge-aware smoothing), box filters only: fast at any size."""
    mean_i, mean_p = _box(guide, r), _box(src, r)
    cov = _box(guide * src, r) - mean_i * mean_p
    var = _box(guide * guide, r) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


def _smoothstep(e0, e1, x):
    import numpy as np
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _srgb_to_lin(x):
    import numpy as np
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(x):
    import numpy as np
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _gauss(x, sigma):
    import cv2
    if sigma <= 0.05:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def _motion(x, length, angle_deg):
    import cv2
    import numpy as np
    n = max(3, int(round(length)) | 1)
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (n, n))
    return cv2.filter2D(x, -1, k / max(k.sum(), 1e-6), borderType=cv2.BORDER_REFLECT)


# ─────────────────────────────────────────────────────────────────────────────
# Phone HDR
# ─────────────────────────────────────────────────────────────────────────────

def phone_hdr(rgb, strength=0.6, shadow_lift=0.4, highlight_compress=0.45, local_contrast=0.3,
              radius=0.02, sharpen_halo=0.3, halo_radius=1.5, saturation=1.05):
    import numpy as np
    h, w = rgb.shape[:2]
    L = np.maximum(_luma(rgb), 1e-4)
    logL = np.log(L)
    r = max(2, int(radius * w))
    base = _guided(logL, logL, r, 0.02)                      # large-scale light, edges kept
    detail = logL - base
    b = np.clip(np.exp(base), 0, 1)
    lifted = b + shadow_lift * 2.2 * b * (1 - b) ** 2        # strongest around b = 1/3
    comp = lifted - highlight_compress * 0.28 * lifted ** 3  # 1 -> 0.72 at full compress
    new_log = np.log(np.maximum(comp, 1e-4)) + detail * (1 + local_contrast)
    L2 = np.exp(new_log)
    if sharpen_halo > 0:                                     # phone over-sharpening: USM halos
        L2 = L2 + sharpen_halo * 1.2 * (L2 - _gauss(L2, max(0.3, halo_radius * w / 1024.0)))
    out = rgb * (L2 / L)[..., None]
    if saturation != 1.0:
        Lo = _luma(out)[..., None]
        out = Lo + (out - Lo) * saturation
    return rgb + (np.clip(out, 0, 1) - rgb) * strength


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

    def apply(self, image, onyx_post=None, **params):
        ensure_profile_ready()
        params, bypass = _onyx_post.apply("OnyxPhoneHDR", params, onyx_post)
        if bypass or params.get("strength", 0) <= 0:
            return (image,)
        arrs = [phone_hdr(a, **params) for a in _np(image)]
        return (_back(arrs, image),)


# ─────────────────────────────────────────────────────────────────────────────
# Highlight clip
# ─────────────────────────────────────────────────────────────────────────────

_FRINGE = {"purple": (0.55, 0.15, 1.0), "warm": (1.0, 0.55, 0.15), "cyan": (0.15, 0.8, 1.0)}


def highlight_clip(rgb, threshold=0.86, push=1.0, knee=0.05, fringe=0.4, fringe_color="purple",
                   fringe_width=0.004, bloom=0.25, bloom_radius=0.02):
    import cv2
    import numpy as np
    h, w = rgb.shape[:2]
    lin = _srgb_to_lin(rgb)
    L = _luma(rgb)
    # Only what is already bright is pushed past the sensor's limit.
    gain = 1.0 + (2.0 ** push - 1.0) * _smoothstep(threshold - 0.08, threshold + 0.04, L)
    lin = lin * gain[..., None]
    # Each channel saturates on its own: the hue drifts towards white at the edge.
    if knee > 0:
        k0 = 1.0 - knee
        lin = np.where(lin > k0, k0 + knee * (1 - np.exp(-(lin - k0) / knee)), lin)
    lin = np.minimum(lin, 1.0)
    clipped = (lin.max(-1) >= 0.985).astype(np.float32)
    if bloom > 0:
        glow = _gauss(clipped, max(1.0, bloom_radius * w))
        lin = lin + bloom * 0.6 * glow[..., None] * (1 - lin)
    if fringe > 0 and clipped.any():
        fw = max(1, int(round(fringe_width * w)))
        ring = _gauss(cv2.dilate(clipped, np.ones((2 * fw + 1, 2 * fw + 1), np.uint8)), fw * 0.7) - clipped
        ring = np.clip(ring, 0, 1)[..., None]
        col = np.array(_FRINGE.get(fringe_color, _FRINGE["purple"]), np.float32)
        lin = lin * (1 - fringe * 0.5 * ring) + fringe * 0.35 * ring * col
    return np.clip(_lin_to_srgb(np.clip(lin, 0, 1)), 0, 1)


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

    def apply(self, image, onyx_post=None, **params):
        ensure_profile_ready()
        params, bypass = _onyx_post.apply("OnyxHighlightClip", params, onyx_post)
        if bypass:
            return (image,)
        arrs = [highlight_clip(a, **params) for a in _np(image)]
        return (_back(arrs, image),)


# ─────────────────────────────────────────────────────────────────────────────
# Lens smudge
# ─────────────────────────────────────────────────────────────────────────────

def lens_smudge(rgb, amount=0.5, haze=0.3, light_threshold=0.75, glow_radius=0.04, streaks=0.35,
                streak_angle=20.0, streak_length=0.12, patchiness=0.5, auto_light=True, seed=0):
    import numpy as np
    h, w = rgb.shape[:2]
    L = _luma(rgb)
    lights = rgb * _smoothstep(light_threshold, min(1.0, light_threshold + 0.2), L)[..., None]
    energy = float(lights.mean())
    scale = amount * (min(1.0, 0.25 + energy * 25.0) if auto_light else 1.0)
    if scale <= 0:
        return rgb
    glow = _gauss(lights, max(1.0, glow_radius * w))
    if streaks > 0:
        glow = glow + streaks * 1.5 * _motion(lights, streak_length * w, streak_angle)
    # Fingerprint: the grease is uneven, so the glow is patchy.
    if patchiness > 0:
        rs = np.random.RandomState(int(seed) & 0x7FFFFFFF)
        small = rs.rand(max(2, h // 64), max(2, w // 64)).astype(np.float32)
        import cv2
        pat = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        pat = _gauss(pat, w * 0.03)
        pat = (pat - pat.min()) / max(1e-6, pat.max() - pat.min())
        glow = glow * (1 - patchiness + patchiness * 1.6 * pat)[..., None]
    # Veiling glare: scattered light lifts the blacks with the scene's light colour.
    veil_col = lights.reshape(-1, 3).mean(0) / max(1e-6, energy) * 0.5 if energy > 1e-5 else np.full(3, 0.5, np.float32)
    veil = np.clip(haze * 0.25 * min(1.0, energy * 15.0 + 0.3), 0, 0.5)
    out = rgb * (1 - veil) + veil * np.clip(veil_col, 0, 1)
    g = np.clip(glow * scale, 0, 1)
    out = 1 - (1 - out) * (1 - g)                                   # screen
    return np.clip(out, 0, 1)


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

    def apply(self, image, onyx_post=None, **params):
        ensure_profile_ready()
        params, bypass = _onyx_post.apply("OnyxLensSmudge", params, onyx_post)
        if bypass or params.get("amount", 0) <= 0:
            return (image,)
        arrs = [lens_smudge(a, **params) for a in _np(image)]
        return (_back(arrs, image),)


# ─────────────────────────────────────────────────────────────────────────────
# Chroma subsampling
# ─────────────────────────────────────────────────────────────────────────────

_SUBSAMPLING = {"4:2:0": (2, 2), "4:2:2": (1, 2), "4:1:1": (1, 4)}


def chroma_subsample(images, mode="4:2:0", upsample="bilinear"):
    """images: torch B,H,W,3 in 0-1. JPEG's YCbCr (BT.601 full range)."""
    import torch
    import torch.nn.functional as F
    fy, fx = _SUBSAMPLING.get(mode, (2, 2))
    x = images[..., :3].float()
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    c = torch.stack([cb, cr], 1)                                        # B,2,H,W
    H, W = c.shape[-2:]
    ph, pw = (-H) % fy, (-W) % fx
    if ph or pw:
        c = F.pad(c, (0, pw, 0, ph), mode="replicate")
    c = F.avg_pool2d(c, (fy, fx))
    if upsample == "nearest":
        c = F.interpolate(c, scale_factor=(fy, fx), mode="nearest")
    else:
        c = F.interpolate(c, scale_factor=(fy, fx), mode="bilinear", align_corners=False)
    c = c[..., :H, :W]
    cb, cr = c[:, 0], c[:, 1]
    out = torch.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], -1).clamp(0, 1)
    if images.shape[-1] == 4:
        out = torch.cat([out, images[..., 3:].float()], -1)
    return out.to(images.dtype)


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

    def apply(self, image, onyx_post=None, **params):
        ensure_profile_ready()
        params, bypass = _onyx_post.apply("OnyxChromaSubsample", params, onyx_post)
        if bypass:
            return (image,)
        return (chroma_subsample(image, **params),)


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
