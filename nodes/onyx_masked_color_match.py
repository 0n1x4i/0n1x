# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""
Onyx Masked Color Match.

Generated video (H3, turbo samplers...) often comes back slightly darker or
tinted compared with the source. When the subject is pasted back on the
original frames, the regenerated background that stays visible around the new
silhouette then shows up as a darker "halo".

This node measures, on every frame, the per-channel mean and spread of the
BACKGROUND (pixels outside `mask`) in both the generated and the source frames,
and applies the affine correction that makes the generated background match the
source. The correction is smoothed over time so it cannot flicker, and is
applied to the whole generated frame (subject included), so the subject keeps
the same relation to its surroundings.
"""

import torch


class OnyxMaskedColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Generated frames to correct."}),
                "reference": ("IMAGE", {"tooltip": "Original source frames (same count and size)."}),
                "mask": ("MASK", {"tooltip": "Subject region to IGNORE when measuring (white = subject)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temporal_radius": ("INT", {"default": 8, "min": 0, "max": 60,
                    "tooltip": "Smooth the correction over +/- this many frames (no flicker)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match"
    CATEGORY = "Onyx"
    DESCRIPTION = "Match the generated background colour/brightness to the source, outside the subject mask."

    @staticmethod
    def _smooth(x, r):
        if r <= 0 or x.shape[0] < 2:
            return x
        T = x.shape[0]
        pad = torch.cat([x[:1].expand(r, *x.shape[1:]), x, x[-1:].expand(r, *x.shape[1:])], 0)
        out = torch.stack([pad[i:i + T] for i in range(2 * r + 1)], 0).mean(0)
        return out

    def match(self, image, reference, mask, strength=1.0, temporal_radius=8):
        T = min(image.shape[0], reference.shape[0])
        img = image[:T].float()
        ref = reference[:T].float()
        if ref.shape[1:3] != img.shape[1:3]:
            ref = torch.nn.functional.interpolate(ref.permute(0, 3, 1, 2), size=img.shape[1:3],
                                                  mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[0] < T:
            m = torch.cat([m, m[-1:].expand(T - m.shape[0], *m.shape[1:])], 0)
        m = m[:T].float()
        if m.shape[1:3] != img.shape[1:3]:
            m = torch.nn.functional.interpolate(m.unsqueeze(1), size=img.shape[1:3], mode="nearest").squeeze(1)
        bg = (m < 0.5).float().unsqueeze(-1)                     # [T,H,W,1]
        n = bg.sum(dim=(1, 2)).clamp(min=1.0)                    # [T,1]

        def stats(x):
            mean = (x * bg).sum(dim=(1, 2)) / n                  # [T,C]
            var = (((x - mean[:, None, None, :]) ** 2) * bg).sum(dim=(1, 2)) / n
            return mean, var.clamp(min=1e-6).sqrt()

        mg, sg = stats(img)
        mr, sr = stats(ref)
        gain = (sr / sg).clamp(0.8, 1.25)
        offset = mr - gain * mg
        gain = self._smooth(gain, int(temporal_radius))
        offset = self._smooth(offset, int(temporal_radius))
        s = float(strength)
        gain = 1.0 + (gain - 1.0) * s
        offset = offset * s
        out = (img * gain[:, None, None, :] + offset[:, None, None, :]).clamp(0.0, 1.0)
        if image.shape[0] > T:
            out = torch.cat([out, image[T:].float()], 0)
        return (out.to(image.dtype),)
