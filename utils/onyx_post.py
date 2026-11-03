# -*- coding: utf-8 -*-
"""
Onyx Post — one choice upstream, every post-processing node follows.

Onyx Studio (or the small "Onyx Post Style" node) outputs an ONYX_POST value.
Every Onyx post-processing node has an optional `onyx_post` input. Wired, and
with a look other than "none", it OVERRIDES that node's widgets with the
preset's values for that node; the widgets typed on the node are used for
anything the preset does not set. "none" = the node's own settings, untouched.

    ONYX_POST = {"look": "amateur", "motion": "shake_med",
                 "nodes": {"Onyx_Renoise": {...}, "OnyxCameraShake": {...}, ...}}

Looks (the whole chain at once):
  crispy   the "Onyx Krea 4K Crispy" workflow as tuned today (reference)
  clean    sharp, almost no grain, no phone degradation
  amateur  everyday phone photo: light sensor noise, mild NR, JPEG 88-92
  night    low light: heavy phone noise + NR smear, cooler, bloom on lights
  flash    direct phone flash: harder contrast, shine on skin, light falloff
  golden   warm low sun: warmer white balance, soft glow

Motion (independent of the look, drives Onyx Camera Shake):
  none · auto (the preset the skill wrote in its NOTES line) · soft ·
  shake_med · shake_med_h · shake_med_v · shake_med_rot · shake_heavy

A node key absent from a look = that node keeps its own settings. The special
key "_bypass": True makes the node return its input unchanged (e.g. no Camera
Look for "clean").

Presets are data, not code: the Worker can publish more in the registry
("post_looks" block, same shape as LOOKS), merged at startup.
"""

from __future__ import annotations

import copy

POST_TYPE = "ONYX_POST"
MOTIONS = ["none", "auto", "soft", "shake_med", "shake_med_h", "shake_med_v", "shake_med_rot", "shake_heavy"]

# Base = the current "Onyx Krea 4K Crispy" workflow values, node by node.
_CRISPY = {
    "OnyxFameGridColorFinish": {"color_strength": 0.05, "sharpen_strength": 0.0},
    "OnyxRealismInjector": {
        "enable_micro_shadows": False, "shadow_strength": 0.3, "shadow_radius": 2.0,
        "shadow_detail_threshold": 0.015, "enable_eye_moisture": False, "moisture_strength": 0.7,
        "catchlight_intensity": 0.8, "tear_meniscus": 0.6, "enable_skin_oil": True,
        "oil_strength": 0.8, "t_zone_gloss": 0.7, "side_matte": 0.5, "global_mix": 1.0},
    "OnyxPostProcessSuite": {
        "enable_upscale": False, "enable_levels": True, "exposure": -0.05, "gamma": 1.05,
        "brightness": 0.0, "contrast": 0.98, "saturation": 1.09, "vibrance": -0.04,
        "enable_color_wheels": True, "gamma_r": 0.994, "gamma_g": 0.994, "gamma_b": 1.015,
        "gain_r": 1.0, "gain_g": 0.997, "gain_b": 1.007, "enable_temp_tint": False,
        "enable_sharpen": False, "enable_small_glow": True, "small_glow_intensity": 0.0,
        "enable_large_glow": True, "large_glow_intensity": 0.075, "large_glow_radius": 50.0,
        "large_glow_threshold": 0.3, "enable_glare": False, "enable_chromatic_aberration": False,
        "enable_vignette": False, "enable_radial_blur": False, "enable_film_grain": False,
        "enable_lens_distortion": False},
    "Onyx_Camera_Look": {"enabled": True, "demosaic_pixel_blur": 0.2, "noise_strength": 0.0,
                         "kernel_motion_blur": 1.8, "jpeg_compression": 100},
    "Onyx_Renoise": {"iso_preset": "ISO 400", "grain_size": 1.0, "look": "Manual", "look_strength": 0.7,
                     "fine_grain": 0.001, "grain_color": 0.25, "mottle": 0.001, "chroma_blotch": 0.006,
                     "blotch_size": 3.5, "purple_green": 0.55, "shadow_weight": 0.8, "nr_smear": 0.0,
                     "chroma_bleed": 1.0, "color_degrade": 0.2, "jpeg_quality": 90},
    "OnyxMoreJPEG": {"recipe": "custom", "jpeg_quality": 100, "passes": 1, "noise_scale": 0.0,
                     "sharpness": 1.0, "contrast": 0.96, "posterize_bits": 8, "blur_radius": 0.0,
                     "pixelate_box": 1},
}


def _derive(base: dict, changes: dict) -> dict:
    out = copy.deepcopy(base)
    for node, values in changes.items():
        if values is None:
            out.pop(node, None)
        else:
            out.setdefault(node, {}).update(values)
    return out


LOOKS = {
    "none": {},
    "crispy": _CRISPY,
    "clean": _derive(_CRISPY, {
        "OnyxFameGridColorFinish": {"color_strength": 0.05, "sharpen_strength": 0.1},
        "OnyxRealismInjector": {"oil_strength": 0.5, "t_zone_gloss": 0.5},
        "OnyxPostProcessSuite": {"contrast": 1.0, "saturation": 1.04, "large_glow_intensity": 0.04},
        "Onyx_Camera_Look": {"_bypass": True},
        "Onyx_Renoise": {"iso_preset": "ISO 100 (Clean)", "look_strength": 0.4, "chroma_blotch": 0.002,
                         "color_degrade": 0.05, "jpeg_quality": 0},
        "OnyxMoreJPEG": {"contrast": 1.0},
    }),
    "amateur": _derive(_CRISPY, {
        "OnyxFameGridColorFinish": {"color_strength": 0.1},
        "OnyxPostProcessSuite": {"contrast": 0.97, "saturation": 1.03, "small_glow_intensity": 0.0,
                                 "large_glow_intensity": 0.05, "enable_vignette": True,
                                 "vignette_strength": 0.08, "vignette_radius": 2.7, "vignette_softness": 4.0},
        "Onyx_Camera_Look": {"demosaic_pixel_blur": 0.6, "noise_strength": 1.0, "kernel_motion_blur": 2.5,
                             "jpeg_compression": 92},
        "Onyx_Renoise": {"iso_preset": "ISO 200", "look": "Phone indoor (subtle)", "look_strength": 0.8},
        "OnyxMoreJPEG": {"jpeg_quality": 88, "contrast": 1.02},
    }),
    "night": _derive(_CRISPY, {
        "OnyxRealismInjector": {"oil_strength": 0.6},
        "OnyxPostProcessSuite": {"exposure": -0.12, "gamma": 1.08, "saturation": 0.95, "vibrance": -0.08,
                                 "enable_temp_tint": True, "temperature": -8.0, "tint": 0.0,
                                 "large_glow_intensity": 0.12, "large_glow_threshold": 0.25},
        "Onyx_Camera_Look": {"demosaic_pixel_blur": 0.8, "noise_strength": 2.0, "kernel_motion_blur": 2.0,
                             "jpeg_compression": 90},
        "Onyx_Renoise": {"iso_preset": "ISO 800", "look": "Phone night (heavy)", "look_strength": 0.9},
        "OnyxMoreJPEG": {"jpeg_quality": 85, "contrast": 1.0},
    }),
    "flash": _derive(_CRISPY, {
        "OnyxRealismInjector": {"oil_strength": 1.0, "t_zone_gloss": 0.9, "side_matte": 0.35},
        "OnyxPostProcessSuite": {"exposure": 0.03, "contrast": 1.06, "saturation": 1.04,
                                 "enable_vignette": True, "vignette_strength": 0.22, "vignette_radius": 1.9,
                                 "vignette_softness": 3.0, "large_glow_intensity": 0.05},
        "Onyx_Camera_Look": {"demosaic_pixel_blur": 0.5, "noise_strength": 1.2, "kernel_motion_blur": 1.5,
                             "jpeg_compression": 92},
        "Onyx_Renoise": {"iso_preset": "ISO 400", "look": "Phone low light (iPhone indoor)", "look_strength": 0.6},
        "OnyxMoreJPEG": {"jpeg_quality": 88, "contrast": 1.03},
    }),
    "golden": _derive(_CRISPY, {
        "OnyxPostProcessSuite": {"saturation": 1.08, "vibrance": 0.04, "enable_temp_tint": True,
                                 "temperature": 10.0, "tint": 2.0, "large_glow_intensity": 0.12,
                                 "large_glow_threshold": 0.35},
        "Onyx_Camera_Look": {"demosaic_pixel_blur": 0.4, "noise_strength": 0.6, "jpeg_compression": 95},
        "Onyx_Renoise": {"iso_preset": "ISO 200", "look_strength": 0.5},
    }),
}

LOOK_INFO = {
    "none": "Each node keeps its own settings.",
    "crispy": "The 4K Crispy workflow as tuned: clean detail, light grain, skin shine.",
    "clean": "Sharp and clean: almost no grain, no phone degradation.",
    "amateur": "Everyday phone photo: light noise, mild noise-reduction smear, JPEG 88-92.",
    "night": "Low light: heavy phone noise, cooler white balance, bloom on lights.",
    "flash": "Direct phone flash: harder contrast, shine on skin, light falloff at the edges.",
    "golden": "Warm low sun: warmer white balance, soft glow.",
}


def looks() -> dict:
    """Built-in looks + the ones published by the Worker (registry 'post_looks')."""
    out = dict(LOOKS)
    try:
        from . import onyx_registry as _reg
        extra = _reg.registry().get("post_looks") or {}
        for name, nodes in extra.items():
            if isinstance(nodes, dict):
                base = out.get(nodes.get("_base", "crispy"), _CRISPY)
                out[name] = _derive(base, {k: v for k, v in nodes.items() if k != "_base"})
    except Exception:
        pass
    return out


def make(look: str = "none", motion: str = "none", auto_motion: str = "", overrides: dict | None = None) -> dict:
    """Build the ONYX_POST value. `auto_motion` = the preset a skill chose (NOTES)."""
    all_looks = looks()
    nodes = copy.deepcopy(all_looks.get(look, {}))
    m = (auto_motion or "none") if motion == "auto" else motion
    if m and m != "none":
        nodes.setdefault("OnyxCameraShake", {})["preset"] = m
    elif motion != "none" or look != "none":
        nodes.setdefault("OnyxCameraShake", {})["preset"] = "none"
    for node, values in (overrides or {}).items():
        nodes.setdefault(node, {}).update(values or {})
    return {"look": look, "motion": m or "none", "nodes": nodes}


def settings_for(node_key: str, post) -> dict:
    """The overriding settings for one node, {} when nothing applies."""
    if not isinstance(post, dict):
        return {}
    if post.get("look", "none") == "none" and post.get("motion", "none") == "none":
        return {}
    return dict((post.get("nodes") or {}).get(node_key) or {})


def apply(node_key: str, kwargs: dict, post) -> tuple:
    """-> (kwargs with the preset applied, bypass?). Only keys the node already
    has (its widgets) are replaced, so a preset can never inject a bad argument."""
    s = settings_for(node_key, post)
    if not s:
        return kwargs, False
    bypass = bool(s.pop("_bypass", False))
    out = dict(kwargs)
    for k, v in s.items():
        if k in out:
            out[k] = v
    changed = [k for k in s if k in kwargs and kwargs[k] != s[k]]
    if changed or bypass:
        print(f"🎛️  [Onyx Post] {node_key}: look '{post.get('look')}'"
              + (" → bypassed" if bypass else f" → {len(changed)} setting(s) overridden"))
    return out, bypass


POST_INPUT = {"onyx_post": (POST_TYPE, {"tooltip": "From Onyx Studio (post output) or Onyx Post Style. "
                                                   "A look other than 'none' overrides this node's settings."})}


def wire(cls, node_key: str | None = None):
    """Give an existing post-processing node class the `onyx_post` input, without
    touching its own code: INPUT_TYPES gets the optional socket, and the node's
    FUNCTION applies the look's values to its keyword arguments before running
    (or returns the image unchanged when the look bypasses it)."""
    key = node_key or cls.__name__
    if getattr(cls, "_onyx_post_wired", False):
        return cls
    orig_types = cls.INPUT_TYPES.__func__ if hasattr(cls.INPUT_TYPES, "__func__") else cls.INPUT_TYPES
    fname = cls.FUNCTION
    orig_fn = getattr(cls, fname)

    def INPUT_TYPES(c):
        spec = dict(orig_types(c))
        opt = dict(spec.get("optional") or {})
        opt.update(POST_INPUT)
        spec["optional"] = opt
        return spec

    def _image_name(spec):
        for name, s in (spec.get("required") or {}).items():
            if isinstance(s, (tuple, list)) and s and s[0] == "IMAGE":
                return name
        return "image"

    def run(self, *args, onyx_post=None, **kwargs):
        kwargs, bypass = apply(key, kwargs, onyx_post)
        if bypass:
            name = _image_name(orig_types(cls))
            return (kwargs[name] if name in kwargs else args[0],)
        return orig_fn(self, *args, **kwargs)

    cls.INPUT_TYPES = classmethod(INPUT_TYPES)
    setattr(cls, fname, run)
    cls._onyx_post_wired = True
    return cls
