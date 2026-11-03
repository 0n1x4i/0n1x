# -*- coding: utf-8 -*-
"""
Onyx More JPEG — contrast / sharpness / JPEG re-encode finishing pass.

Same widget names and order as the "MoreJPEG" node used in the Krea 4K
workflows (so their saved values load unchanged), written from scratch for
the Onyx pack: the original repository publishes no licence, so none of its
code is used here.

Order of operations, per pass:
  pixelate -> blur -> noise -> posterize -> contrast -> sharpness -> effects
  (grayscale, invert, emboss, edge enhance) -> JPEG re-encode (when < 100)
`passes` repeats the whole chain (generation loss, like re-sharing a photo).

Recipes set the main values in one click ("custom" = the widgets as typed):
  lite          light re-share: JPEG 82, mild sharpening
  noise         sensor-like noise + JPEG 90
  artifact_hell heavy blocking: JPEG 8, 4 passes, posterize 5
  pro_plus      crisp finish: JPEG 95, sharpness 1.25, contrast 1.04
"""

from __future__ import annotations

import io

from ..utils import onyx_post as _onyx_post
from .onyx_render_profile import ensure_profile_ready

RECIPES = {
    "custom": {},
    "lite": {"jpeg_quality": 82, "passes": 1, "noise_scale": 0.0, "sharpness": 1.1, "contrast": 1.0},
    "noise": {"jpeg_quality": 90, "passes": 1, "noise_scale": 6.0},
    "artifact_hell": {"jpeg_quality": 8, "passes": 4, "posterize_bits": 5, "noise_scale": 4.0},
    "pro_plus": {"jpeg_quality": 95, "passes": 1, "sharpness": 1.25, "contrast": 1.04},
}


def _process(pil, p, rng):
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    img = pil
    for _ in range(max(1, int(p["passes"]))):
        if p["pixelate_box"] > 1:
            w, h = img.size
            b = int(p["pixelate_box"])
            img = img.resize((max(1, w // b), max(1, h // b)), Image.NEAREST).resize((w, h), Image.NEAREST)
        if p["blur_radius"] > 0:
            img = img.filter(ImageFilter.GaussianBlur(p["blur_radius"]))
        if p["noise_scale"] > 0:
            arr = np.asarray(img, dtype=np.float32)
            arr = arr + rng.normal(0.0, p["noise_scale"], arr.shape).astype(np.float32)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if int(p["posterize_bits"]) < 8:
            img = ImageOps.posterize(img, int(p["posterize_bits"]))
        if p["contrast"] != 1.0:
            img = ImageEnhance.Contrast(img).enhance(p["contrast"])
        if p["sharpness"] != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(p["sharpness"])
        if p["grayscale"]:
            img = ImageOps.grayscale(img).convert("RGB")
        if p["invert"]:
            img = ImageOps.invert(img)
        if p["emboss"]:
            img = img.filter(ImageFilter.EMBOSS)
        if p["edge_enhance"]:
            img = img.filter(ImageFilter.EDGE_ENHANCE)
        q = int(p["jpeg_quality"])
        if q < 100:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=max(1, q))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
    return img


class OnyxMoreJPEG:
    """Contrast, sharpness, noise and JPEG re-encode finish (MoreJPEG-compatible widgets)."""

    DESCRIPTION = ("Finishing pass: contrast, sharpness, noise, posterize, blur, pixelate and real JPEG "
                   "re-encoding, repeated `passes` times. Widget-compatible with MoreJPEG.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "recipe": (list(RECIPES), {"default": "custom"}),
                "jpeg_quality": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1,
                                         "tooltip": "100 = no re-encode."}),
                "passes": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "noise_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0,
                                          "tooltip": "Gaussian noise, in 0-255 levels."}),
                "sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                        "tooltip": "1 = unchanged."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                       "tooltip": "1 = unchanged."}),
                "posterize_bits": ("INT", {"default": 8, "min": 1, "max": 8, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "pixelate_box": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
                "grayscale": ("BOOLEAN", {"default": False}),
                "emboss": ("BOOLEAN", {"default": False}),
                "edge_enhance": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": dict(_onyx_post.POST_INPUT),
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Onyx/Post"

    def apply(self, image, onyx_post=None, **params):
        ensure_profile_ready()
        params, bypass = _onyx_post.apply("OnyxMoreJPEG", params, onyx_post)
        if bypass:
            return (image,)
        import numpy as np
        import torch
        from PIL import Image
        p = dict(params)
        p.update(RECIPES.get(p.get("recipe", "custom"), {}))
        rng = np.random.default_rng(int(p.get("seed", 0)) % (2 ** 32))
        out = []
        for i in range(image.shape[0]):
            arr = (image[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            res = _process(Image.fromarray(arr[..., :3]).convert("RGB"), p, rng)
            out.append(torch.from_numpy(np.asarray(res, dtype=np.float32) / 255.0))
        return (torch.stack(out, 0),)


NODE_CLASS_MAPPINGS = {"OnyxMoreJPEG": OnyxMoreJPEG}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxMoreJPEG": "Onyx More JPEG"}
