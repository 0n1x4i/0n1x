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

Motion (independent of the look, drives Onyx Camera Shake):
  none (Camera Shake switched off) · manual (Camera Shake keeps its own
  settings) · auto (the preset the skill wrote in its NOTES line) · soft ·
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
MOTIONS = ["none", "manual", "auto", "soft", "shake_med", "shake_med_h", "shake_med_v", "shake_med_rot", "shake_heavy"]

# The look values (crispy, clean, amateur, night, flash) are protected data:
# they come from the Worker (onyx_data_core, LOOKS). Only "none" ships with the pack.
def _derive(base: dict, changes: dict) -> dict:
    out = copy.deepcopy(base)
    for node, values in changes.items():
        if values is None:
            out.pop(node, None)
        else:
            out.setdefault(node, {}).update(values)
    return out


LOOKS = {"none": {}}


LOOK_INFO = {
    "none": "Each node keeps its own settings.",
    "default": "Onyx default: the Krea 4K Crispy chain as used today (colour finish, skin, grade, camera, lens, grain, contrast); everything else off.",
    "crispy": "The 4K Crispy workflow as tuned: clean detail, light grain, skin shine.",
    "clean": "Sharp and clean: almost no grain, no phone degradation.",
    "amateur": "Everyday phone photo: light noise, mild noise-reduction smear, JPEG 88-92.",
    "night": "Low light: heavy phone noise, cooler white balance, bloom on lights.",
    "flash": "Direct phone flash: harder contrast, shine on skin, light falloff at the edges.",
}


def look_names() -> list:
    """Names for dropdowns: always the known ones (saved workflows stay valid offline) + any extra."""
    return list(dict.fromkeys([*LOOK_INFO, *looks()]))


def looks(wait: float = 0.0) -> dict:
    """Built-in looks + the ones published by the Worker (onyx_data_core LOOKS, registry 'post_looks').
    `wait`: seconds to wait for a Worker fetch in flight (0 = what is cached now)."""
    out = dict(LOOKS)
    try:
        from . import onyx_remote_data as _rd
        for name, nodes in (_rd.get("LOOKS", {}, wait=wait) or {}).items():
            if isinstance(nodes, dict):
                out[name] = copy.deepcopy(nodes)
    except Exception:
        pass
    try:
        from . import onyx_registry as _reg
        extra = _reg.registry().get("post_looks") or {}
        for name, nodes in extra.items():
            if isinstance(nodes, dict):
                base = out.get(nodes.get("_base", "crispy"), {})
                out[name] = _derive(base, {k: v for k, v in nodes.items() if k != "_base"})
    except Exception:
        pass
    return out


def make(look: str = "none", motion: str = "none", auto_motion: str = "", overrides: dict | None = None) -> dict:
    """Build the ONYX_POST value. `auto_motion` = the preset a skill chose (NOTES)."""
    all_looks = looks(wait=15.0)
    nodes = copy.deepcopy(all_looks.get(look, {}))
    # none   = no camera shake (the Camera Shake node is switched off)
    # manual = the Camera Shake node keeps its own settings
    # auto   = the preset the skill wrote in its NOTES (none when it wrote none)
    m = (auto_motion or "none") if motion == "auto" else (motion or "none")
    if m == "manual":
        nodes.pop("OnyxCameraShake", None)
    else:
        nodes.setdefault("OnyxCameraShake", {})["preset"] = m
    for node, values in (overrides or {}).items():
        nodes.setdefault(node, {}).update(values or {})
    return {"look": look, "motion": m or "none", "nodes": nodes}


def settings_for(node_key: str, post) -> dict:
    """The overriding settings for one node, {} when nothing applies."""
    if not isinstance(post, dict):
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
        print(f"🎛️  [Onyx Post] {node_key}: look '{post.get('look')}', motion '{post.get('motion')}'"
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
