# -*- coding: utf-8 -*-
"""
Onyx Camera Shake — the "shake" post-process of the Candid Snap method, as a node.

Why after the render and not in the prompt: the workflow's detail stages
re-sharpen the face, so blur written into the prompt is either cleaned away or
turns the face to mush. Real hand-shake blur is added to the finished image.

Presets (the names the Candid Snap skill writes in its NOTES line, so the
Studio's `post_preset` output can drive this node directly):
  soft           slightly soft / hazy overall, no smear
  shake_med      diagonal hand-shake smear, eyes and lips still readable
  shake_med_h    side-to-side smear with a faint double image
  shake_med_v    up-and-down smear
  shake_med_rot  smear plus a slightly tilted frame
  shake_heavy    strong smear, flash-in-the-dark style, face barely readable
  none           pass-through

Lengths are relative to the image size, so a preset looks the same at 1K and 4K.
Batches (video frames) get a slightly different random shake per frame from `seed`.
"""

from __future__ import annotations

import math
import random

from .onyx_render_profile import ensure_profile_ready

PRESETS = ["none", "soft", "shake_med", "shake_med_h", "shake_med_v", "shake_med_rot", "shake_heavy"]

# length = blur length as a fraction of the long side; angle in degrees (0 = horizontal);
# ghost = (strength, offset fraction) of the faint double image; tilt = frame rotation (deg)
_SPECS = {
    "soft":          {"length": 0.0,   "angle": 0,   "ghost": (0.0, 0.0),   "tilt": 0.0, "soft": 1.0, "haze": 0.06},
    "shake_med":     {"length": 0.010, "angle": 35,  "ghost": (0.0, 0.0),   "tilt": 0.0, "soft": 0.0, "haze": 0.0},
    "shake_med_h":   {"length": 0.012, "angle": 0,   "ghost": (0.22, 0.010), "tilt": 0.0, "soft": 0.0, "haze": 0.0},
    "shake_med_v":   {"length": 0.011, "angle": 90,  "ghost": (0.0, 0.0),   "tilt": 0.0, "soft": 0.0, "haze": 0.0},
    "shake_med_rot": {"length": 0.010, "angle": 20,  "ghost": (0.0, 0.0),   "tilt": 2.2, "soft": 0.0, "haze": 0.0},
    "shake_heavy":   {"length": 0.032, "angle": 25,  "ghost": (0.30, 0.016), "tilt": 1.0, "soft": 0.0, "haze": 0.03},
}


def _line_kernel(length_px: float, angle_deg: float, device):
    import torch
    n = max(1, int(round(length_px)))
    size = n if n % 2 else n + 1
    k = torch.zeros((size, size), device=device)
    c = (size - 1) / 2
    a = math.radians(angle_deg)
    steps = max(2, size * 3)
    for i in range(steps):
        t = i / (steps - 1) - 0.5
        x = c + t * (size - 1) * math.cos(a)
        y = c - t * (size - 1) * math.sin(a)
        k[int(round(y)), int(round(x))] += 1.0
    return k / k.sum()


def _gauss_kernel(sigma: float, device):
    import torch
    r = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def _conv(img, kernel):
    """img [1,C,H,W], kernel [k,k] -> same size, reflect padding, per channel."""
    import torch.nn.functional as F
    k = kernel.shape[0]
    p = k // 2
    x = F.pad(img, (p, p, p, p), mode="reflect" if p < min(img.shape[-2:]) else "replicate")
    w = kernel[None, None].repeat(img.shape[1], 1, 1, 1)
    return F.conv2d(x, w, groups=img.shape[1])


def _rotate(img, deg):
    """Rotate and zoom just enough that no empty corner shows (a real tilted phone)."""
    import torch
    import torch.nn.functional as F
    a = math.radians(deg)
    h, w = img.shape[-2:]
    zoom = abs(math.cos(a)) + abs(math.sin(a)) * max(h / w, w / h)
    cos, sin = math.cos(a) / zoom, math.sin(a) / zoom
    theta = torch.tensor([[cos, -sin * h / w, 0.0], [sin * w / h, cos, 0.0]],
                         device=img.device, dtype=img.dtype)[None]
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def shake_frame(img, preset: str, strength: float, rng: random.Random):
    """img [H,W,C] float 0-1 -> same."""
    import torch
    spec = _SPECS.get(preset)
    if spec is None or strength <= 0:
        return img
    x = img.permute(2, 0, 1)[None]
    long_side = max(x.shape[-2:])
    if spec["tilt"]:
        x = _rotate(x, spec["tilt"] * strength * rng.choice((-1, 1)) * rng.uniform(0.8, 1.2))
    if spec["length"]:
        length = spec["length"] * long_side * strength * rng.uniform(0.85, 1.15)
        angle = spec["angle"] + rng.uniform(-8, 8)
        x = _conv(x, _line_kernel(length, angle, x.device))
    if spec["ghost"][0]:
        amount, off = spec["ghost"]
        dx = int(round(off * long_side * strength * math.cos(math.radians(spec["angle"]))))
        dy = int(round(-off * long_side * strength * math.sin(math.radians(spec["angle"]))))
        ghost = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        x = x * (1 - amount * min(strength, 1.5)) + ghost * amount * min(strength, 1.5)
    if spec["soft"]:
        x = _conv(x, _gauss_kernel(max(0.4, spec["soft"] * long_side / 1400 * strength), x.device))
    if spec["haze"]:
        h = spec["haze"] * strength
        x = x * (1 - h) + h * x.mean(dim=(-2, -1), keepdim=True).clamp(0.35, 0.65)
    return x[0].permute(1, 2, 0).clamp(0, 1)


class OnyxCameraShake:
    """Add phone hand-shake / softness after the render (Candid Snap 'post' presets)."""

    DESCRIPTION = (
        "Phone camera shake after rendering: soft, shake_med (diagonal), shake_med_h (side-to-side "
        "+ faint double image), shake_med_v, shake_med_rot (+ tilted frame), shake_heavy. Connect "
        "Onyx Studio's post_preset to `preset_override` to use the preset the skill chose for "
        "this image."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (PRESETS, {"default": "shake_med"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "preset_override": ("STRING", {"forceInput": True,
                    "tooltip": "From Onyx Studio `post_preset`. When connected and not empty, it "
                               "replaces the preset above ('none' = no shake for this image)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, images, preset, strength, seed, preset_override=None):
        ensure_profile_ready()
        chosen = (preset_override or "").strip().lower() or preset
        if chosen not in PRESETS:
            print(f"⚠️  [Onyx Camera Shake] unknown preset '{chosen}' — using '{preset}'.")
            chosen = preset
        if chosen == "none":
            return (images,)
        import torch
        rng = random.Random(seed)
        out = [shake_frame(images[i], chosen, float(strength), rng) for i in range(images.shape[0])]
        return (torch.stack(out, 0),)


NODE_CLASS_MAPPINGS = {"OnyxCameraShake": OnyxCameraShake}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxCameraShake": "Onyx Camera Shake"}
