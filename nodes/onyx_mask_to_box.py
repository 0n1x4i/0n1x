# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""
Onyx Mask To Box.

Replaces a holey, flickering SAM3 video mask by one clean, stable shape per frame:
a padded rectangle or ellipse around the detected region. Holes (mouth, hair
strands) cannot exist any more, and the shape cannot flicker because the box
coordinates are median-filtered then averaged over neighbouring frames.

It costs almost nothing: per frame it only looks for the first/last non-empty
row and column, so there is no dilate / blur / closing involved.
"""

import numpy as np
import torch


def _fill_missing(v, valid):
    """Hold the last valid value forward, then the first valid value backward."""
    out = v.copy()
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out, False
    last = out[idx[0]]
    for i in range(len(out)):
        if valid[i]:
            last = out[i]
        else:
            out[i] = last
    return out, True


def _smooth(v, radius):
    if radius <= 0 or len(v) < 3:
        return v
    n = len(v)
    pad = np.pad(v, (radius, radius), mode="edge")
    # mediane glissante (supprime les frames aberrantes), puis moyenne glissante
    stack = np.stack([pad[i:i + n] for i in range(2 * radius + 1)], axis=0)
    med = np.median(stack, axis=0)
    pad2 = np.pad(med, (radius, radius), mode="edge")
    k = np.ones(2 * radius + 1, dtype=np.float32) / (2 * radius + 1)
    return np.convolve(pad2, k, mode="valid")


class OnyxMaskToBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "shape": (["ellipse", "rectangle"], {"default": "ellipse",
                    "tooltip": "ellipse = rounder, covers a head without grabbing the box corners. "
                               "rectangle = covers everything inside the box."}),
                "padding_percent": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Grows the box by this % of its own size on every side."}),
                "smoothing_frames": ("INT", {"default": 3, "min": 0, "max": 30,
                    "tooltip": "Temporal radius. Box coordinates are median-filtered then averaged over "
                               "+/- this many frames: no flicker, no single-frame outliers. 0 = off."}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "single_box_for_clip": ("BOOLEAN", {"default": False,
                    "tooltip": "Use ONE box covering all frames (most stable, but large if the head moves a lot)."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "Onyx"
    DESCRIPTION = "Turns a SAM3 video mask into a stable padded box/ellipse per frame. Fast, no blur."

    def process(self, masks, shape="ellipse", padding_percent=12.0, smoothing_frames=3,
                threshold=0.5, single_box_for_clip=False):
        m = masks
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.dim() == 4:
            m = m[..., 0] if m.shape[-1] == 1 else m.amax(dim=1)
        T, H, W = m.shape

        binm = m > threshold
        rows = binm.any(dim=2).cpu()   # [T,H]
        cols = binm.any(dim=1).cpu()   # [T,W]
        ys = torch.arange(H)
        xs = torch.arange(W)
        y0 = torch.where(rows, ys, torch.full_like(ys, H)).amin(dim=1).numpy().astype(np.float32)
        y1 = torch.where(rows, ys, torch.full_like(ys, -1)).amax(dim=1).numpy().astype(np.float32)
        x0 = torch.where(cols, xs, torch.full_like(xs, W)).amin(dim=1).numpy().astype(np.float32)
        x1 = torch.where(cols, xs, torch.full_like(xs, -1)).amax(dim=1).numpy().astype(np.float32)
        valid = y1 >= 0

        if not valid.any():
            return (torch.zeros((T, H, W), dtype=torch.float32),)

        coords = []
        for v in (y0, y1, x0, x1):
            f, _ = _fill_missing(v, valid)
            # frames de debut sans detection : on prend la premiere valeur valide
            first = int(np.where(valid)[0][0])
            f[:first] = f[first]
            coords.append(f)
        y0, y1, x0, x1 = coords

        if single_box_for_clip:
            y0[:] = y0.min(); x0[:] = x0.min(); y1[:] = y1.max(); x1[:] = x1.max()
        else:
            y0, y1, x0, x1 = (_smooth(v, int(smoothing_frames)) for v in (y0, y1, x0, x1))

        # marge
        bh = np.maximum(y1 - y0 + 1, 1.0)
        bw = np.maximum(x1 - x0 + 1, 1.0)
        pad = padding_percent / 100.0
        y0 = np.clip(y0 - bh * pad, 0, H - 1); y1 = np.clip(y1 + bh * pad, 0, H - 1)
        x0 = np.clip(x0 - bw * pad, 0, W - 1); x1 = np.clip(x1 + bw * pad, 0, W - 1)

        out = torch.zeros((T, H, W), dtype=torch.float32)
        yy = torch.arange(H, dtype=torch.float32).view(1, H, 1)
        xx = torch.arange(W, dtype=torch.float32).view(1, 1, W)
        step = 16  # chunks : memoire bornee quel que soit le clip
        for s in range(0, T, step):
            e = min(T, s + step)
            Y0 = torch.from_numpy(y0[s:e]).view(-1, 1, 1); Y1 = torch.from_numpy(y1[s:e]).view(-1, 1, 1)
            X0 = torch.from_numpy(x0[s:e]).view(-1, 1, 1); X1 = torch.from_numpy(x1[s:e]).view(-1, 1, 1)
            if shape == "rectangle":
                sel = (yy >= Y0) & (yy <= Y1) & (xx >= X0) & (xx <= X1)
            else:
                cy = (Y0 + Y1) / 2; cx = (X0 + X1) / 2
                ry = ((Y1 - Y0) / 2 * 1.15).clamp(min=1.0); rx = ((X1 - X0) / 2 * 1.15).clamp(min=1.0)
                sel = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
            out[s:e] = sel.float()
        return (out,)
