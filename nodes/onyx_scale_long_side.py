# -*- coding: utf-8 -*-
"""
ComfyUI node - Onyx Scale Long Side (MP grid).

Scales an image so its LONG side hits a chosen value, then lands the short side
on a resolution that Onyx Resolution (MP) actually produces for that aspect
ratio — cropping the few extra pixels equally from both edges.

Why not just "scale to 3840 and round": Onyx Resolution (MP) snaps both sides to
its grid (multiples of 32) and picks the pair from a megapixel budget, so for a
given long side only a couple of short sides ever come out of it (9:16 at 3840 ->
2144 or 2176). Feeding a 3840x2201 image into a pipeline built around those
values breaks the "same resolution everywhere" contract. This node uses the very
same _snap_to_budget table, so the two nodes can never disagree.
"""

import functools
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .onyx_render_profile import ensure_profile_ready
from .onyx_resolution_mp import _ASPECT_PRESETS, _nearest_ratio_label, _snap_to_budget

_MP_MIN, _MP_MAX, _MP_STEP = 0.05, 20.0, 0.01      # widget range of Onyx Resolution (MP)
_AUTO = "auto (nearest to image)"
_PRESETS = [k for k, v in _ASPECT_PRESETS.items() if v is not None]
_METHODS = ["lanczos", "bicubic", "bilinear", "area"]


@functools.lru_cache(maxsize=64)
def _grid(label: str, multiple: int) -> dict:
    """{long_side: (short sides...)} exactly as Onyx Resolution (MP) produces them.

    Walks the MP widget range at its 0.01 step and records every (w, h) pair the
    resolution node can output for this ratio. One long side usually maps to 2-3
    short sides (the MP node optimises the pixel budget, not the ratio), which is
    why the final choice is made against the image, not here.
    """
    w0, h0 = _ASPECT_PRESETS[label]
    ar = w0 / h0
    out = {}
    n = int(round((_MP_MAX - _MP_MIN) / _MP_STEP))
    for i in range(n + 1):
        mp = round(_MP_MIN + i * _MP_STEP, 2)
        w, h = _snap_to_budget(mp, ar, multiple)
        out.setdefault(max(w, h), set()).add(min(w, h))
    return {k: tuple(sorted(v)) for k, v in out.items()}


@functools.lru_cache(maxsize=1)
def _long_side_choices() -> list:
    """Every long side Onyx Resolution (MP) can output at /32, any ratio."""
    sides = set()
    for label in _PRESETS:
        sides.update(_grid(label, 32).keys())
    return [str(s) for s in sorted(sides)]


def _resize(images: torch.Tensor, w: int, h: int, method: str) -> torch.Tensor:
    """[B,H,W,C] float -> [B,h,w,C]."""
    if method == "lanczos":
        out = []
        for img in images:
            arr = (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
            pil = Image.fromarray(arr, mode=mode).resize((w, h), Image.Resampling.LANCZOS)
            out.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))
        return torch.stack(out).to(images.device)
    x = images.movedim(-1, 1)
    if method == "area":
        x = F.interpolate(x, size=(h, w), mode="area")
    else:
        x = F.interpolate(x, size=(h, w), mode=method, align_corners=False, antialias=True)
    return x.movedim(1, -1).clamp(0.0, 1.0)


class OnyxScaleLongSide:

    @classmethod
    def INPUT_TYPES(cls):
        choices = _long_side_choices()
        return {
            "required": {
                "image": ("IMAGE",),
                "long_side": (choices, {
                    "default": "3840" if "3840" in choices else choices[len(choices) // 2],
                    "tooltip": "Target for the LONGEST side of the image. Only values that "
                               "Onyx Resolution (MP) can output are listed (multiples of 32).",
                }),
                "aspect_ratio": ([_AUTO] + _PRESETS, {
                    "default": _AUTO,
                    "tooltip": "Which Onyx Resolution (MP) ratio the output must match.\n"
                               "auto: the named ratio closest to the image (a 1080x1920 "
                               "photo -> 9:16). Force one only if auto picks the wrong one: "
                               "the further it is from the image, the more gets cropped.",
                }),
                "multiple_of": ("INT", {
                    "default": 32, "min": 8, "max": 128, "step": 8,
                    "tooltip": "Same grid as Onyx Resolution (MP). Keep 32 to get the exact "
                               "same resolutions.",
                }),
                "method": (_METHODS, {
                    "default": "lanczos",
                    "tooltip": "lanczos: sharpest, best for upscaling a photo. area: best "
                               "for a strong downscale. bicubic / bilinear: faster on big batches.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "aspect_ratio")
    FUNCTION = "scale"
    CATEGORY = "Onyx/Utils"
    DESCRIPTION = ("Scale an image by its longest side to a resolution of the Onyx Resolution "
                   "(MP) grid, then center-crop the short side to the matching grid value.")

    def scale(self, image, long_side, aspect_ratio, multiple_of, method):
        ensure_profile_ready()
        L = int(long_side)
        mult = int(multiple_of)
        src_h, src_w = int(image.shape[1]), int(image.shape[2])
        portrait = src_h > src_w
        src_long, src_short = max(src_w, src_h), min(src_w, src_h)

        label = _nearest_ratio_label(src_w / src_h) if aspect_ratio == _AUTO else aspect_ratio
        pw, ph = _ASPECT_PRESETS[label]
        # Une image paysage ne peut pas prendre un ratio portrait (et l'inverse) :
        # on garde la famille du preset mais dans l'orientation de l'image.
        if pw != ph and (pw < ph) != portrait and src_w != src_h:
            flipped = f"{ph}:{pw}"
            if flipped in _ASPECT_PRESETS:
                label = flipped

        grid = _grid(label, mult)
        if L not in grid:
            near = min(grid, key=lambda k: abs(k - L))
            print(f"[Onyx Scale Long Side] {L} is not on the {label} grid at /{mult} — "
                  f"using {near}.")
            L = near

        # Petit cote : parmi ceux que Resolution (MP) donne pour ce grand cote, le
        # plus proche de ce que l'image donnerait a l'echelle -> le moins de crop.
        # A egalite, le plus petit (on recadre plutot que d'agrandir).
        scaled_short = src_short * L / src_long
        S = min(grid[L], key=lambda s: (abs(s - scaled_short), s))

        out_w, out_h = (S, L) if portrait else (L, S)
        if src_w == src_h and label != "1:1":
            out_w, out_h = (L, S) if _ASPECT_PRESETS[label][0] >= _ASPECT_PRESETS[label][1] else (S, L)

        # Couvrir la cible puis recadrer au centre : aucune bande noire, aucune
        # deformation. Le facteur est celui qui fait tenir les DEUX cotes.
        k = max(out_w / src_w, out_h / src_h)
        rw, rh = max(out_w, round(src_w * k)), max(out_h, round(src_h * k))
        img = _resize(image, rw, rh, method)
        left = (rw - out_w) // 2
        top = (rh - out_h) // 2
        img = img[:, top:top + out_h, left:left + out_w, :].contiguous()

        crop_x, crop_y = rw - out_w, rh - out_h
        print(f"📐 [Onyx Scale Long Side] {src_w}x{src_h} -> scaled {rw}x{rh} -> "
              f"{out_w}x{out_h} ({label}, /{mult}) | cropped "
              f"{crop_x}px wide ({left}|{crop_x - left}), {crop_y}px high ({top}|{crop_y - top})"
              + (" | WARNING: large crop, check the aspect ratio" if max(crop_x / rw, crop_y / rh) > 0.05 else ""))
        return (img, out_w, out_h, label)


NODE_CLASS_MAPPINGS = {"OnyxScaleLongSide": OnyxScaleLongSide}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxScaleLongSide": "Onyx Scale Long Side (MP grid)"}
