# -*- coding: utf-8 -*-
"""
Onyx Camera Shake — the Candid Snap "shake.py" pass as a node.

Real-photo distortion after the render: camera-shake motion blur + double-image
ghost + soft defocus (+ optional dutch tilt), sized relative to the image WIDTH
so it matches blurry phone references at any resolution. Blur written into the
prompt gets sharpened away by the detail stages or turns the face to mush, so it
is added here, to the finished image.

Same presets, values and operation order as shake.py
(rotate -> ghost -> motion blur -> gaussian softness):

               length  angle  ghost_off  ghost_a  sigma   rotate
  soft         0.000     0     0.000      0.00    0.0012    0
  shake_med    0.010    15     0.008      0.20    0.0008    0
  shake_med_h  0.012     0     0.012      0.25    0.0008    0
  shake_med_v  0.010    75     0.007      0.18    0.0008    0
  shake_med_rot 0.012   60     0.008      0.20    0.0008    8
  shake_heavy  0.022    35     0.015      0.30    0.0015    0

`strength` scales lengths, offsets, sigma and tilt (1 = shake.py exactly).
Onyx Studio's `post` output (or Onyx Post Style) can choose the preset per image.
"""

from __future__ import annotations

import math

from ..utils import onyx_post as _onyx_post
from .onyx_render_profile import ensure_profile_ready

PRESETS = {
    #                length  angle  ghost_off ghost_a  sigma   rotate
    "soft":          (0.000,   0,   0.000,   0.00,  0.0012,  0),
    "shake_med":     (0.010,  15,   0.008,   0.20,  0.0008,  0),
    "shake_med_h":   (0.012,   0,   0.012,   0.25,  0.0008,  0),
    "shake_med_v":   (0.010,  75,   0.007,   0.18,  0.0008,  0),
    "shake_med_rot": (0.012,  60,   0.008,   0.20,  0.0008,  8),
    "shake_heavy":   (0.022,  35,   0.015,   0.30,  0.0015,  0),
}
PRESET_NAMES = ["none", *PRESETS]


def _motion_kernel(length, angle_deg):
    import cv2
    import numpy as np
    n = max(3, int(round(length)) | 1)
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (n, n))
    return k / k.sum()


def shake(img, preset: str, strength: float = 1.0):
    """img: float32 HxWx3 in 0-255 (numpy). Same maths as shake.py.run()."""
    import cv2
    import numpy as np
    if preset not in PRESETS or strength <= 0:
        return img
    h, w = img.shape[:2]
    length, ang, goff, ga, sigma, rot = PRESETS[preset]
    length, goff, sigma, rot = length * strength, goff * strength, sigma * strength, rot * strength
    ga = min(0.6, ga * strength) if strength > 1 else ga * strength
    if rot:
        t = math.radians(abs(rot))
        s = max((w * math.cos(t) + h * math.sin(t)) / w, (w * math.sin(t) + h * math.cos(t)) / h)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rot, s)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if ga:
        dx = int(round(goff * w * math.cos(math.radians(ang))))
        dy = int(round(-goff * w * math.sin(math.radians(ang))))
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        ghost = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        img = (1 - ga) * img + ga * ghost
    if length:
        img = cv2.filter2D(img, -1, _motion_kernel(length * w, ang), borderType=cv2.BORDER_REFLECT)
    if sigma:
        img = cv2.GaussianBlur(img, (0, 0), sigma * w)
    return np.clip(img, 0, 255)


class OnyxCameraShake:
    """Phone camera shake / softness after rendering (the Candid Snap shake.py presets)."""

    DESCRIPTION = (
        "shake.py as a node: soft, shake_med (diagonal), shake_med_h (side-to-side + double image), "
        "shake_med_v, shake_med_rot (+ tilted frame), shake_heavy. Width-relative, so it looks the same at "
        "any resolution. Wire Onyx Studio's `post` (or Onyx Post Style) to choose the preset per image."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (PRESET_NAMES, {"default": "shake_med"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                       "tooltip": "1 = shake.py exactly."}),
            },
            "optional": {
                "preset_override": ("STRING", {"forceInput": True,
                    "tooltip": "A preset name as text (e.g. Onyx Studio's post_preset). Not empty = "
                               "replaces the preset above; empty = the preset above is used; "
                               "'none' = no shake for this image."}),
                **_onyx_post.POST_INPUT,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, images, preset, strength, preset_override=None, onyx_post=None):
        ensure_profile_ready()
        values, bypass = _onyx_post.apply("OnyxCameraShake", {"preset": preset, "strength": strength}, onyx_post)
        if bypass:
            return (images,)
        chosen = (preset_override or "").strip().lower() or values["preset"]
        if chosen not in PRESET_NAMES:
            print(f"⚠️  [Onyx Camera Shake] unknown preset '{chosen}' — using '{values['preset']}'.")
            chosen = values["preset"]
        if chosen == "none":
            return (images,)
        import numpy as np
        import torch
        out = []
        for i in range(images.shape[0]):
            arr = images[i, ..., :3].detach().cpu().numpy().astype(np.float32) * 255.0
            out.append(torch.from_numpy(shake(arr, chosen, float(values["strength"])) / 255.0))
        return (torch.stack(out, 0).to(images.dtype),)


NODE_CLASS_MAPPINGS = {"OnyxCameraShake": OnyxCameraShake}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxCameraShake": "Onyx Camera Shake"}
