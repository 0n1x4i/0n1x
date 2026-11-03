# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""
Onyx Lens Character — the optical signature of a real lens, on GPU.

Same controls as a classic "lens / bokeh" post node, written from scratch for
Onyx in torch (runs on the GPU, whole batch at once):

  * focus map: radial, gradient_top or flat (no focus blur);
  * out-of-focus blur with a polygon aperture (number of blades), highlight bloom,
    longitudinal CA (red / blue blur slightly larger / smaller), field curvature
    (edge_softness) and cat-eye smearing toward the corners;
  * lateral (transverse) chromatic aberration growing toward the edges;
  * optical vignette (natural fall-off).

With focus_mode = flat and blur_strength = 0 only the lens "character" remains:
lateral CA + vignette, no blur at all.
"""

import math

import torch
import torch.nn.functional as F

from .onyx_render_profile import ensure_profile_ready

_FOCUS_MODES = ["radial", "gradient_top", "flat"]


def _luma(x):  # x: [B,3,H,W]
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def _radius_grid(b, h, w, device):
    """Normalised distance to the frame centre: 0 at the centre, 1 at the corners."""
    y = torch.linspace(-0.5, 0.5, h, device=device).view(1, 1, h, 1)
    x = torch.linspace(-0.5, 0.5, w, device=device).view(1, 1, 1, w)
    return (torch.sqrt(x * x + y * y) / math.sqrt(0.5)).expand(b, 1, h, w)


def _gauss(x, sigma):
    if sigma <= 0.05:
        return x
    r = max(1, int(math.ceil(sigma * 3)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t * t) / (2 * sigma * sigma))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def _aperture(radius, blades, device, dtype):
    """Normalised aperture kernel: a disc, or a regular polygon with `blades` sides."""
    r = max(1, int(round(radius)))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    dist = torch.sqrt(xx * xx + yy * yy) / r
    if blades >= 3:
        half = math.pi / blades
        ang = torch.remainder(torch.atan2(yy, xx), 2 * half) - half
        edge = math.cos(half) / torch.clamp(torch.cos(ang), min=1e-3)
        k = (dist <= edge).to(dtype)
    else:
        k = (dist <= 1.0).to(dtype)
    return k / k.sum().clamp(min=1e-6)


def _conv_kernel(x, k):
    c = x.shape[1]
    p = k.shape[-1] // 2
    x = F.pad(x, (p, p, p, p), mode="reflect")
    return F.conv2d(x, k.view(1, 1, *k.shape).repeat(c, 1, 1, 1), groups=c)


def _scale_from_centre(ch, s):
    """Resample one channel [B,1,H,W] magnified by s around the centre (lateral CA)."""
    b, _, h, w = ch.shape
    y = torch.linspace(-1, 1, h, device=ch.device, dtype=ch.dtype) / s
    x = torch.linspace(-1, 1, w, device=ch.device, dtype=ch.dtype) / s
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((gx, gy), -1).unsqueeze(0).expand(b, h, w, 2)
    return F.grid_sample(ch, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


class OnyxLensCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "focus_mode": (_FOCUS_MODES, {"default": "flat",
                    "tooltip": "Where the lens is in focus. flat = no focus blur (lens character only)."}),
                "blur_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Out-of-focus blur radius in pixels. 0 = no blur."}),
                "focus_center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_center_y": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Radial: focus centre. gradient_top: the sharp line."}),
                "focus_size": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_feather": ("FLOAT", {"default": 0.55, "min": 0.01, "max": 2.0, "step": 0.01}),
                "aperture_blades": ("INT", {"default": 7, "min": 0, "max": 16,
                    "tooltip": "Bokeh shape: 0-2 = round, 3+ = polygon with that many blades."}),
                "cat_eye": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "long_ca": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Longitudinal CA: red / blue fringes in the blurred areas."}),
                "transverse_ca": ("FLOAT", {"default": 0.09, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Lateral CA: colour fringes growing toward the edges of the frame."}),
                "optical_vignette": ("FLOAT", {"default": 0.14, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_softness": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Field curvature: extra softness toward the edges (needs blur_strength > 0)."}),
                "bokeh_highlight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Bloom of bright highlights inside the bokeh."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Swap sharp and blurred areas."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Onyx/Post"
    DESCRIPTION = ("Real-lens character on GPU: focus blur with polygon bokeh, longitudinal and lateral "
                   "chromatic aberration, field curvature, cat-eye and optical vignette.")

    # ── focus map: 0 = sharp, 1 = max blur ─────────────────────────────────
    @staticmethod
    def _focus(b, h, w, mode, cx, cy, size, feather, invert, device, dtype):
        if mode == "flat":
            amt = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
        else:
            y = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
            x = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
            if mode == "gradient_top":
                amt = (y - cy).abs()
            else:  # radial
                amt = ((torch.sqrt((x - cx) ** 2 + (y - cy) ** 2) - size) / max(feather, 1e-3))
            amt = amt.clamp(0, 1)
        if invert:
            amt = 1.0 - amt
        return _gauss(amt, max(1.0, min(h, w) * 0.01)).clamp(0, 1)

    def run(self, image, focus_mode, blur_strength, focus_center_x, focus_center_y, focus_size,
            focus_feather, aperture_blades, cat_eye, long_ca, transverse_ca, optical_vignette,
            edge_softness, bokeh_highlight, invert):
        ensure_profile_ready()
        import comfy.model_management as mm
        device = mm.get_torch_device()
        out = []
        for i in range(image.shape[0]):           # one frame at a time: big kernels stay in VRAM
            src = image[i:i + 1].to(device=device, dtype=torch.float32)
            alpha = src[..., 3:] if src.shape[-1] > 3 else None
            x = src[..., :3].movedim(-1, 1)        # [1,3,H,W]
            _, _, h, w = x.shape
            fmap = self._focus(1, h, w, focus_mode, focus_center_x, focus_center_y, focus_size,
                               focus_feather, invert, device, x.dtype)
            redge = _radius_grid(1, h, w, device)

            if edge_softness > 0:
                fmap = (fmap + edge_softness * 0.5 * redge ** 2 * (1 - fmap)).clamp(0, 1)

            res = x
            if blur_strength > 0.1:
                srcb = x
                if bokeh_highlight > 0:
                    hl = ((_luma(x) - 0.72) / 0.28).clamp(0, 1)
                    srcb = x + hl * bokeh_highlight
                radius = min(40.0, max(2.0, float(blur_strength)))
                blades = int(aperture_blades)
                disk = _conv_kernel(srcb, _aperture(radius, blades, device, x.dtype))
                if long_ca > 0:
                    kr = _aperture(radius * (1 + 0.12 * long_ca), blades, device, x.dtype)
                    kb = _aperture(max(1.0, radius * (1 - 0.12 * long_ca)), blades, device, x.dtype)
                    disk = torch.cat([_conv_kernel(srcb[:, 0:1], kr), disk[:, 1:2],
                                      _conv_kernel(srcb[:, 2:3], kb)], 1)
                mid = _gauss(srcb, max(0.5, radius * 0.4))
                # 0 → sharp, 0.5 → soft gaussian, 1 → full aperture bokeh
                t = fmap * 2.0
                lo = (t.clamp(0, 1))
                hi = ((t - 1.0).clamp(0, 1))
                res = x * (1 - lo) + mid * lo
                res = res * (1 - hi) + disk * hi
                res = res.clamp(0, 1)
                if cat_eye > 0:
                    smear = _gauss(res, max(0.5, radius * 0.5))
                    wgt = cat_eye * 0.4 * redge ** 2 * fmap
                    res = res * (1 - wgt) + smear * wgt

            if transverse_ca > 0:
                s = transverse_ca * 0.004
                res = torch.cat([_scale_from_centre(res[:, 0:1], 1.0 + s), res[:, 1:2],
                                 _scale_from_centre(res[:, 2:3], 1.0 - s)], 1)

            if optical_vignette > 0:
                res = res * (1.0 - optical_vignette * redge ** 2)

            res = res.clamp(0, 1).movedim(1, -1)
            if alpha is not None:
                res = torch.cat([res, alpha], -1)
            out.append(res.to(device=image.device, dtype=image.dtype))
        return (torch.cat(out, 0),)


NODE_CLASS_MAPPINGS = {"OnyxLensCharacter": OnyxLensCharacter}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxLensCharacter": "Onyx Lens Character"}
