# -*- coding: utf-8 -*-
"""
Onyx Post Studio ✦ — the whole post-processing chain in ONE node.

The chain of the "Onyx Krea 4K Crispy" workflow, in its order, plus the camera
shake at the very end (motion blur goes last: it smears everything before it):

  1 Color finish   OnyxFameGridColorFinish   colour pop + light sharpen, on the clean render
  2 Skin & eyes    OnyxRealismInjector       skin oil / T-zone, micro shadows, eye moisture
  3 Grade & optics OnyxPostProcessSuite      levels, colour wheels, glow, glare, vignette...
  4 Camera sensor  Onyx_Camera_Look          demosaic softness, sensor noise, lens blur, JPEG
  5 Grain & noise  Onyx_Renoise              ISO grain, chroma blotches, NR smear
  6 JPEG & contrast OnyxMoreJPEG             final contrast / JPEG pass
  7 Motion         OnyxCameraShake           shake.py presets (none by default)

Every stage RUNS THE EXISTING NODE (same code, protected code still comes from
the Worker); this node only stores the settings and calls them in order. The
settings live in the hidden `post_state` widget (JSON): {stages: {key: {on, v}}}.
Defaults = each node's defaults with the Crispy workflow values on top.

`onyx_post` (Onyx Studio's `post` / Onyx Post Style) still works: a look other
than none overrides the stages' settings at run time, exactly as on the nodes.

Preview: the last image this node processed (or an image dropped on the panel)
is kept in memory, and the panel can run the chain on it to show before / after.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import os
import threading
import time
from collections import OrderedDict

from ..utils import onyx_post as _onyx_post
from .onyx_render_profile import ensure_profile_ready

STAGES = [
    # ── the scene: colour, skin, grade (the "photo" before the camera) ──────────
    {"key": "color", "group": "scene", "short": "Color", "cls": "OnyxFameGridColorFinish", "title": "Color finish", "icon": "🎨",
     "info": "Colour pop + light sharpen. First, on the clean render."},
    {"key": "skin", "group": "scene", "short": "Skin", "cls": "OnyxRealismInjector", "title": "Skin & eyes", "icon": "💧",
     "info": "Skin oil and T-zone shine, micro shadows, eye moisture (face-aware)."},
    {"key": "skintone", "group": "scene", "short": "Skin tone", "cls": "Onyx_Skin_Tone", "title": "Skin tone", "icon": "🧴",
     "info": "Pulls skin hues towards a natural tone and evens them out."},
    {"key": "dehaze", "group": "scene", "short": "Dehaze", "cls": "Onyx_Dehaze", "title": "Dehaze", "icon": "🌁",
     "info": "Removes haze / flat contrast (dark channel)."},
    {"key": "clarity", "group": "scene", "short": "Clarity", "cls": "Onyx_Clarity", "title": "Clarity", "icon": "🔍",
     "info": "Midtone local contrast, edge-aware."},
    {"key": "grade", "group": "scene", "short": "Grade", "cls": "OnyxPostProcessSuite", "title": "Grade & optics", "icon": "🎚️",
     "info": "Levels, colour wheels, temperature, glow, glare, chromatic aberration, vignette, lens."},
    {"key": "lut", "group": "scene", "short": "LUT", "cls": "Onyx_Apply_LUT", "title": "LUT", "icon": "🎞️",
     "info": "A .cube LUT from the pack's luts/ folder — the final colour grade."},
    # ── the phone: lens, sensor, processing, file ─────────────────────────────
    {"key": "smudge", "group": "phone", "short": "Smudge", "cls": "OnyxLensSmudge", "title": "Lens smudge", "icon": "👆",
     "info": "Fingerprint on the lens: milky veil, patchy glow and smear streaks around lights."},
    {"key": "clip", "group": "phone", "short": "Clip", "cls": "OnyxHighlightClip", "title": "Highlight clip", "icon": "💡",
     "info": "Lamps and windows burn out like on a sensor: per-channel clip, colour drift, fringe, bloom."},
    {"key": "camera", "group": "phone", "short": "Camera", "cls": "Onyx_Camera_Look", "title": "Camera sensor", "icon": "📷",
     "info": "Phone sensor: demosaic softness, sensor noise, lens blur, JPEG."},
    {"key": "hdr", "group": "phone", "short": "HDR", "cls": "OnyxPhoneHDR", "title": "Phone HDR", "icon": "📱",
     "info": "The phone's smart HDR: lifted shadows, held highlights, local contrast, sharpening halos."},
    {"key": "grain", "group": "phone", "short": "Grain", "cls": "Onyx_Renoise", "title": "Grain & noise", "icon": "🌫️",
     "info": "ISO grain, chroma blotches, purple/green noise, noise-reduction smear."},
    {"key": "jpeg", "group": "phone", "short": "JPEG", "cls": "OnyxMoreJPEG", "title": "JPEG & contrast", "icon": "🗜️",
     "info": "Final contrast and JPEG pass."},
    {"key": "lens", "group": "phone", "short": "Lens", "cls": "OnyxLensCharacter", "title": "Lens character", "icon": "🔭",
     "info": "Real-lens signature: lateral / longitudinal chromatic aberration, optical vignette, optional "
             "focus blur with polygon bokeh (focus_mode flat + blur 0 = CA + vignette only)."},
    {"key": "motion", "group": "phone", "short": "Motion", "cls": "OnyxCameraShake", "title": "Motion", "icon": "🌀",
     "info": "Camera shake (shake.py): it smears everything before it. "
             "Studio's motion (post input) picks the preset per image."},
    # ── after the photo exists: the upload ────────────────────────────────────
    {"key": "chroma", "group": "upload", "short": "Chroma", "cls": "OnyxChromaSubsample", "title": "Chroma subsample", "icon": "📤",
     "info": "Colour resolution of a JPEG / Instagram upload (4:2:0). After everything: it happens to the finished file."},
]
STAGE_KEYS = [s["key"] for s in STAGES]
# New stages start switched off: the Crispy chain is unchanged until you turn them on.
_DEFAULT_OFF = {"skintone", "dehaze", "clarity", "lut", "lens", "smudge", "clip", "hdr", "chroma"}
# Extra node inputs a stage receives from the Post Studio's own optional inputs.
_STAGE_INPUTS = {"skin": {"depth_map": "depth_map"}
                 "motion": {"motion_override": "preset_override"}}
_NEEDS: dict = {}                       # stage -> input it cannot run without (skipped with a note)
_HIDDEN_PARAMS = {"enabled"}            # Camera Look's own on/off = the stage switch
_MAIN_PARAMS = {"global_mix"}           # never part of an enable_* group
_DEFAULT_OVERRIDES = {"motion": {"preset": "none"}}
_WIDGET_TYPES = {"INT", "FLOAT", "BOOLEAN", "STRING"}


# ─────────────────────────────────────────────────────────────────────────────
# Stage classes (resolved from the pack's mappings at run time: Renoise and
# Camera Look are loaded by the pack's __init__, from the Worker)
# ─────────────────────────────────────────────────────────────────────────────

def _mappings() -> dict:
    pkg = __package__.rsplit(".", 1)[0] if __package__ else ""
    try:
        return getattr(importlib.import_module(pkg), "NODE_CLASS_MAPPINGS", {}) or {}
    except Exception:
        return {}


def stage_class(key: str):
    stage = next((s for s in STAGES if s["key"] == key), None)
    return _mappings().get(stage["cls"]) if stage else None


def _image_param(spec: dict) -> str:
    for name, s in (spec.get("required") or {}).items():
        if isinstance(s, (tuple, list)) and s and s[0] == "IMAGE":
            return name
    return "image"


def _params(cls) -> list:
    """The widgets of a node class, in order, as plain dicts for the UI."""
    try:
        spec = cls.INPUT_TYPES()
    except Exception:
        return []
    out = []
    for section in ("required", "optional"):
        for name, s in (spec.get(section) or {}).items():
            if not isinstance(s, (tuple, list)) or not s or name in _HIDDEN_PARAMS:
                continue
            kind, opts = s[0], (s[1] if len(s) > 1 and isinstance(s[1], dict) else {})
            if opts.get("forceInput"):
                continue
            if isinstance(kind, (list, tuple)):
                options = [str(o) for o in kind]
                default = opts.get("default", options[0] if options else "")
                out.append({"name": name, "type": "COMBO", "options": options, "default": default,
                            "tooltip": opts.get("tooltip", "")})
            elif kind in _WIDGET_TYPES:
                p = {"name": name, "type": kind, "default": opts.get("default"),
                     "tooltip": opts.get("tooltip", "")}
                for k in ("min", "max", "step"):
                    if k in opts:
                        p[k] = opts[k]
                if p["default"] is None:
                    p["default"] = {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False, "STRING": ""}[kind]
                out.append(p)
    # enable_xxx switches open a group; everything after it belongs to it.
    group = "main"
    for p in out:
        if p["type"] == "BOOLEAN" and p["name"].startswith("enable_"):
            group = p["name"][len("enable_"):]
            p["group"], p["switch"] = group, True
        elif p["name"] in _MAIN_PARAMS:
            p["group"] = "main"
        else:
            p["group"] = group
    return out


def defaults(key: str, cls=None) -> dict:
    """Each node's own defaults, with the Crispy workflow values on top."""
    cls = cls or stage_class(key)
    if cls is None:
        return {}
    vals = {p["name"]: p["default"] for p in _params(cls)}
    stage = next(s for s in STAGES if s["key"] == key)
    crispy = _onyx_post.looks().get("crispy", {}).get(stage["cls"], {})
    vals.update({k: v for k, v in crispy.items() if k in vals})
    vals.update({k: v for k, v in _DEFAULT_OVERRIDES.get(key, {}).items() if k in vals})
    if key == "lut":                    # preselect the pack's natural LUT when it is there
        for p in _params(cls):
            if p["name"] == "lut_file":
                natural = [o for o in p["options"] if "natural" in o.lower()]
                if natural:
                    vals["lut_file"] = natural[0]
    return vals


def spec() -> dict:
    """Everything the panel needs: stages + their widgets, looks, presets."""
    stages = []
    for s in STAGES:
        cls = stage_class(s["key"])
        params = _params(cls) if cls is not None else []
        dflt = defaults(s["key"], cls) if cls is not None else {}
        for p in params:
            p["default"] = dflt.get(p["name"], p["default"])
        stages.append({**s, "available": cls is not None, "params": params,
                       "default_on": s["key"] not in _DEFAULT_OFF})
    looks = {}
    for name, nodes in _onyx_post.looks().items():
        if name == "none":
            continue
        looks[name] = {s["key"]: dict(nodes.get(s["cls"], {})) for s in STAGES if s["cls"] in nodes}
    return {"stages": stages, "looks": looks, "look_info": dict(_onyx_post.LOOK_INFO),
            "presets": presets()}


def _coerce(p: dict, v):
    try:
        if p["type"] == "INT":
            return int(v)
        if p["type"] == "FLOAT":
            return float(v)
        if p["type"] == "BOOLEAN":
            return bool(v)
        if p["type"] == "COMBO":
            return v if str(v) in p["options"] else p["default"]
        return str(v)
    except (TypeError, ValueError):
        return p["default"]


def stage_values(key: str, saved: dict, cls=None) -> dict:
    """defaults + the panel's values (only known widgets, type-checked)."""
    cls = cls or stage_class(key)
    vals = defaults(key, cls)
    params = {p["name"]: p for p in _params(cls)} if cls is not None else {}
    for name, v in (saved or {}).items():
        if name in params:
            vals[name] = _coerce(params[name], v)
    return vals


# ─────────────────────────────────────────────────────────────────────────────
# The chain
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(res):
    if isinstance(res, dict):
        res = res.get("result", ())
    return res[0]


def run_chain(images, state: dict, onyx_post=None, depth_map=None, motion_override=None,
              preview: bool = False, progress=None) -> tuple:
    """-> (images, [(stage key, seconds or error text)]). Stages switched off are
    skipped. In a preview a failing stage is skipped and reported; in a real run
    it raises, like the separate node would."""
    stages_state = (state or {}).get("stages") or {}
    timings = []
    for s in STAGES:
        st = stages_state.get(s["key"]) or {}
        on = st.get("on", s["key"] not in _DEFAULT_OFF)
        if progress:
            progress(s["key"])
        if not on:
            continue
        cls = stage_class(s["key"])
        if cls is None:
            print(f"⚠️  [Onyx Post Studio] {s['title']}: node not available — stage skipped.")
            continue
        vals = stage_values(s["key"], st.get("v") or {}, cls)
        if preview and s["key"] == "grade":
            vals["enable_upscale"] = False                 # previews stay fast
        try:
            spec_ = cls.INPUT_TYPES()
        except Exception:
            spec_ = {}
        kwargs = {_image_param(spec_): images, **vals}
        if "enabled" in ((spec_.get("required") or {}) | (spec_.get("optional") or {})):
            kwargs["enabled"] = True
        if onyx_post is not None:
            kwargs["onyx_post"] = onyx_post
        extras = {"depth_map": depth_map, "motion_override": motion_override}
        need = _NEEDS.get(s["key"])
        if need and extras.get(need) is None:
            msg = f"needs the `{need}` input"
            print(f"⚠️  [Onyx Post Studio] {s['title']}: {msg} — stage skipped.")
            timings.append((s["key"], msg))
            continue
        for src, dst in _STAGE_INPUTS.get(s["key"], {}).items():
            if extras.get(src) is not None:
                kwargs[dst] = extras[src]
        t0 = time.time()
        try:
            images = _unwrap(getattr(cls(), cls.FUNCTION)(**kwargs))
        except Exception as e:
            if not preview:
                raise RuntimeError(f"Onyx Post Studio — {s['title']}: {e}") from e
            timings.append((s["key"], str(e)[:300]))
            continue
        timings.append((s["key"], round(time.time() - t0, 3)))
    return images, timings


# ─────────────────────────────────────────────────────────────────────────────
# Presets (10 slots) — stored next to Onyx Studio's
# ─────────────────────────────────────────────────────────────────────────────

PRESET_SLOTS = 10
_lock = threading.Lock()


def _store_path() -> str:
    from .onyx_studio import studio_dir
    return os.path.join(studio_dir(), "post_presets.json")


def presets() -> dict:
    try:
        with open(_store_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_presets(data: dict) -> None:
    path = _store_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def save_preset(slot: int, name: str, state: dict) -> dict:
    if not 1 <= int(slot) <= PRESET_SLOTS:
        raise ValueError(f"slot must be 1-{PRESET_SLOTS}")
    with _lock:
        data = presets()
        data[str(int(slot))] = {"name": (name or f"Preset {slot}").strip()[:40],
                                "stages": (state or {}).get("stages") or {}, "saved": time.time()}
        _write_presets(data)
        return data


def delete_preset(slot: int) -> dict:
    with _lock:
        data = presets()
        data.pop(str(int(slot)), None)
        _write_presets(data)
        return data


# ─────────────────────────────────────────────────────────────────────────────
# Preview source (last processed image per node, or a dropped one)
# ─────────────────────────────────────────────────────────────────────────────

_SOURCES: "OrderedDict[str, dict]" = OrderedDict()
_MAX_SOURCES = 3


def remember_source(node_id, images, onyx_post=None, motion_override=None) -> None:
    if node_id is None or images is None:
        return
    import numpy as np
    arr = (images[0, ..., :3].detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    _SOURCES[str(node_id)] = {"img": arr, "post": onyx_post, "motion": motion_override}
    _SOURCES.move_to_end(str(node_id))
    while len(_SOURCES) > _MAX_SOURCES:
        _SOURCES.popitem(last=False)


def set_source_from_bytes(node_id, data: bytes) -> tuple:
    from PIL import Image
    import numpy as np
    im = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8).copy()
    prev = _SOURCES.get(str(node_id), {})
    _SOURCES[str(node_id)] = {**prev, "img": arr}
    _SOURCES.move_to_end(str(node_id))
    while len(_SOURCES) > _MAX_SOURCES:
        _SOURCES.popitem(last=False)
    return arr.shape[1], arr.shape[0]


def _view(arr, mode: str, cx: float, cy: float, size: int):
    """fit: whole image, long side <= size. crop: a size x size window at 100%."""
    from PIL import Image
    h, w = arr.shape[:2]
    if mode == "crop" and (w > size or h > size):
        cw, ch = min(w, size), min(h, size)
        x0 = int(round(min(max(cx * w - cw / 2, 0), w - cw)))
        y0 = int(round(min(max(cy * h - ch / 2, 0), h - ch)))
        return arr[y0:y0 + ch, x0:x0 + cw]
    scale = min(1.0, size / max(w, h))
    if scale < 1.0:
        im = Image.fromarray(arr).resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        import numpy as np
        return np.asarray(im)
    return arr


def _jpeg_b64(arr) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_preview(node_id, state: dict, mode: str = "fit", cx: float = 0.5, cy: float = 0.5,
                   size: int = 1024) -> dict:
    src = _SOURCES.get(str(node_id))
    if not src:
        raise LookupError("No image yet: run the workflow once, or drop an image on the preview.")
    import numpy as np
    import torch
    view = _view(src["img"], mode, float(cx), float(cy), int(size))
    t = torch.from_numpy(view.astype(np.float32) / 255.0)[None]
    out, timings = run_chain(t, state, onyx_post=src.get("post"), preview=True,
                             motion_override=src.get("motion"))
    after = (out[0, ..., :3].detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    h, w = src["img"].shape[:2]
    return {"before": _jpeg_b64(np.ascontiguousarray(view)), "after": _jpeg_b64(after),
            "timings": timings, "source_size": [w, h], "view_size": [view.shape[1], view.shape[0]],
            "post_look": (src.get("post") or {}).get("look", "none") if isinstance(src.get("post"), dict) else "none"}


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class OnyxPostStudio:
    """All the Onyx post-processing nodes in one, in the Crispy workflow's order."""

    DESCRIPTION = (
        "The whole post chain in one node, in order: color finish → skin & eyes → grade & optics → camera "
        "sensor → grain & noise → JPEG & contrast → motion (last). One tab per stage, on/off per stage, "
        "before/after preview, 10 presets. Runs the same code as the separate nodes. Wire Onyx Studio's "
        "`post` to apply its look and motion."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "post_state": ("STRING", {"default": "{}", "socketless": True}),
            },
            "optional": {
                **_onyx_post.POST_INPUT,
                "depth_map": ("IMAGE", {"tooltip": "Optional depth map for Skin & eyes (bright = near)."}),
                "motion_override": ("STRING", {"forceInput": True,
                    "tooltip": "A Camera Shake preset name (e.g. Onyx Studio's post_preset). 'none' = no "
                               "shake, empty = the Motion tab's preset."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Onyx/Post"

    def run(self, image, post_state="{}", onyx_post=None, depth_map=None, motion_override=None, unique_id=None):
        ensure_profile_ready()
        try:
            state = json.loads(post_state or "{}")
        except ValueError:
            state = {}
        try:
            remember_source(unique_id, image, onyx_post, motion_override)
        except Exception as e:
            print(f"⚠️  [Onyx Post Studio] preview source not kept: {e}")
        bar = None
        try:
            import comfy.utils
            bar = comfy.utils.ProgressBar(len(STAGES))
        except Exception:
            pass
        out, timings = run_chain(image, state, onyx_post=onyx_post, depth_map=depth_map,
                                 motion_override=motion_override,
                                 progress=(lambda _k: bar.update(1)) if bar else None)
        if timings:
            print("🎛️  [Onyx Post Studio] " + " → ".join(f"{k} {t:.2f}s" for k, t in timings
                                                       if isinstance(t, float)))
        return (out,)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    inst = PromptServer.instance
    if getattr(inst, "_onyx_post_studio_routes", False):
        return
    inst._onyx_post_studio_routes = True
    import asyncio

    def ok(**kw):
        return web.json_response({"ok": True, **kw})

    def err(e, status=400):
        return web.json_response({"ok": False, "error": str(e)}, status=status)

    @inst.routes.get("/onyx/post_studio/spec")
    async def _spec(_r):
        try:
            return ok(**await asyncio.get_running_loop().run_in_executor(None, spec))
        except Exception as e:
            return err(e, 500)

    @inst.routes.post("/onyx/post_studio/preview")
    async def _preview(request):
        try:
            b = await request.json()
            res = await asyncio.get_running_loop().run_in_executor(
                None, lambda: render_preview(b.get("node_id"), b.get("state") or {}, b.get("mode", "fit"),
                                             b.get("cx", 0.5), b.get("cy", 0.5), int(b.get("size", 1024))))
            return ok(**res)
        except LookupError as e:
            return err(e, 404)
        except Exception as e:
            return err(e, 500)

    @inst.routes.post("/onyx/post_studio/source")
    async def _source(request):
        try:
            b = await request.json()
            data = b.get("data_url", "")
            raw = base64.b64decode(data.split(",", 1)[1] if "," in data else data)
            w, h = await asyncio.get_running_loop().run_in_executor(
                None, lambda: set_source_from_bytes(b.get("node_id"), raw))
            return ok(size=[w, h])
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/post_studio/preset/save")
    async def _preset_save(request):
        try:
            b = await request.json()
            return ok(presets=save_preset(int(b.get("slot", 0)), b.get("name", ""), b.get("state") or {}))
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/post_studio/preset/delete")
    async def _preset_delete(request):
        try:
            b = await request.json()
            return ok(presets=delete_preset(int(b.get("slot", 0))))
        except Exception as e:
            return err(e)


NODE_CLASS_MAPPINGS = {"OnyxPostStudio": OnyxPostStudio}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxPostStudio": "Onyx Post Studio ✦"}
