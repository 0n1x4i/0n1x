# -*- coding: utf-8 -*-
# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""
Onyx H3 Guide Exclude Zone — make part of an Add Guide clip "transparent".

MiniMax H3 does not paste a guide clip into the video: Add Guide appends the
clip's latent as extra rows (2x2 latent patches = 32x32 pixels) that sit on the
same positions as the target video, and the target reads them through joint
attention. This node patches the model's attention so that the target can no
longer read the guide rows inside a mask (e.g. the head): those patches are
dropped from the keys/values. Everything outside the mask is guided exactly as
before; inside it, the model only has the reference image, the prompt and the
audio to go on.

Only guide segments as long as the target (the full guide clip) are masked;
short anchor clips are left alone. It replaces any other attention override on
the model (Sparse Attention, ...): put it last, or leave those out.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

LOG = logging.getLogger("onyx.h3_guide_exclude")
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _latent_frame_spans(vt):
    """Pixel frame span [a, b) covered by each latent frame of a vt-frame latent."""
    spans, start = [], 0
    for k in range(vt):
        n = _FRAME_PER_TOKEN[k % 5]
        spans.append((start, start + n))
        start += n
    return spans


def _token_mask(mask, vt, th, tw, dilate):
    """[F, H, W] pixel mask -> [vt * th * tw] bool, True = guide patch to hide."""
    m = mask.float()
    if m.dim() == 2:
        m = m[None]
    f = m.shape[0]
    pooled = F.adaptive_max_pool2d(m[:, None], (th, tw))[:, 0] > 0.5           # [F, th, tw]
    if dilate > 0:
        k = 2 * dilate + 1
        pooled = F.max_pool2d(pooled[:, None].float(), k, stride=1, padding=dilate)[:, 0] > 0.5
    spans = _latent_frame_spans(vt)
    total = spans[-1][1]
    out = torch.zeros(vt, th, tw, dtype=torch.bool)
    for i, (a, b) in enumerate(spans):
        # map the latent's pixel span onto the mask's frames (the mask may have another length)
        lo = min(f - 1, int(a * f / max(1, total)))
        hi = max(lo + 1, min(f, int(round(b * f / max(1, total)))))
        out[i] = pooled[lo:hi].any(dim=0)
    return out.reshape(-1)


class _ExcludeOverride:
    """optimized_attention_override: drop the masked guide rows from k / v."""

    def __init__(self, mask, dilate):
        self.mask = mask.detach().cpu()
        self.dilate = int(dilate)
        self._cache = {}          # id(layout) -> keep index (LongTensor) or None

    def _keep(self, layout, device):
        key = id(layout)
        if key in self._cache:
            keep = self._cache[key]
            return None if keep is None else keep.to(device)
        keep = None
        try:
            _, latent_t, lat_h, lat_w, _ = layout.signature
            th, tw = lat_h // 2, lat_w // 2
            rows = th * tw
            drop = torch.zeros(layout.seq_len, dtype=torch.bool)
            hidden = 0
            for a, b, kind in layout.segments:
                if kind != "cond" or (b - a) % rows:
                    continue
                vt = (b - a) // rows
                if vt < latent_t:          # short anchor clip: keep it whole
                    continue
                tok = _token_mask(self.mask, vt, th, tw, self.dilate)
                drop[a:b] = tok
                hidden += int(tok.sum())
            if hidden:
                keep = torch.nonzero(~drop).flatten()
                print(f"🫥 [Onyx H3 Guide Exclude] {hidden} guide patches hidden "
                      f"({100.0 * hidden / max(1, int((drop | ~drop).sum())):.1f}% of the sequence)")
            else:
                print("⚠️  [Onyx H3 Guide Exclude] no full-length guide clip found, or the mask is empty: "
                      "nothing hidden.")
        except Exception as e:
            LOG.warning("[Onyx H3 Guide Exclude] could not build the token mask (%s): guide left intact.", e)
        self._cache[key] = keep
        return None if keep is None else keep.to(device)

    def __call__(self, func, q, k, v, heads, *args, **kwargs):
        layout = (kwargs.get("transformer_options") or {}).get("minimax_h3_layout")
        if layout is not None and k.shape[-2] == layout.seq_len:
            keep = self._keep(layout, k.device)
            if keep is not None:
                k = k.index_select(-2, keep)
                v = v.index_select(-2, keep)
        return func(q, k, v, heads, *args, **kwargs)


class OnyxH3GuideExcludeZone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The MiniMax H3 model (after its LoRAs)."}),
                "mask": ("MASK", {"tooltip": "Zone to hide from the guide, one mask per guide frame (e.g. a SAM3 "
                                             "'head' track). White = the video cannot see the guide there."}),
                "dilate_patches": ("INT", {"default": 1, "min": 0, "max": 8,
                                           "tooltip": "Grow the hidden zone by N patches (1 patch = 32 px) so the "
                                                      "edge of the head is hidden too."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Onyx/H3"
    DESCRIPTION = ("Makes a zone of the Add Guide clip transparent: inside the mask the video can no longer read "
                   "the guide (attention-level), outside it the guide works as usual.")

    def patch(self, model, mask, dilate_patches=1):
        m = model.clone()
        to = m.model_options.setdefault("transformer_options", {})
        if to.get("optimized_attention_override") is not None:
            print("⚠️  [Onyx H3 Guide Exclude] another attention override (e.g. Sparse Attention) was replaced.")
        to["optimized_attention_override"] = _ExcludeOverride(mask, dilate_patches)
        return (m,)


NODE_CLASS_MAPPINGS = {"OnyxH3GuideExcludeZone": OnyxH3GuideExcludeZone}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxH3GuideExcludeZone": "Onyx H3 Guide Exclude Zone"}
