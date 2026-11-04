# -*- coding: utf-8 -*-
"""
Onyx Studio — one node for the whole image-creation front end.

    media batch (images + videos)  ─┐
    character (LoRAs, trigger, locks)├─► prompt ─► LoRAs on MODEL/CLIP ─► CONDITIONING
    prompt source + skill + LLM     ─┘

Prompt sources (the "mode"):
  manual    the text typed in the node ({trigger} = the character's trigger line).
  gallery   an entry of the Onyx Prompt Gallery library.
  idea      an idea sentence -> the chosen skill + LLM writes the prompt.
  recreate  the current batch image (or the `image` input) -> skill + LLM
            rewrites it as a prompt for YOUR character. With Queue All, every
            item of the batch gets its own prompt.

Skills are prompt-writing rulebooks used as the LLM's system prompt:
  * files you import (.txt / .md) — e.g. CANDID-SNAP-MASTER.txt;
  * your Grok Prompt Generator presets (used as the instruction sent with the image);
  * skills served by the Worker (bundle "prompt_skills_core", if published).
Imported skills live in ComfyUI/user/<user>/onyx_studio/skills/, never in the
pack's folder or its git repository: they are your work, not shipped code.

A skill can carry a CONFIG block (lines like `TRIGGER_LINE = ...`, `BODY = ...`,
`RATIO = ...`, `OUTPUT = ...`, as in the Candid Snap master). The node rewrites
those lines from the selected character before each call, forces OUTPUT = chat,
and replaces the skill's locked sentence (e.g. the hair lock) with the
character's own. The answer's code block becomes the prompt, its NOTES line
becomes `notes` (+ `ratio`). `post_preset` = the motion chosen (or the skill's in auto); 'none' switches the
Camera Shake off, empty (motion 'manual') = the shake node keeps its own settings.

LLM back-ends: Ollama (local), Qwen3-VL local (transformers), Grok, OpenRouter,
Gemini — see utils/onyx_llm.py. API keys are stored on this computer
(user folder, keys.json), never in the workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid

from .onyx_render_profile import ensure_profile_ready
from . import onyx_conditioning_cache as _cond_cache
from ..utils import onyx_llm as _llm
from ..utils import onyx_registry as _reg
from ..utils.onyx_http import OnyxAPIError
from ..utils import onyx_post as _onyx_post

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('studio_bundle', 'onyx_studio', __package__)
    return ns


MODES = ["manual", "idea", "recreate"]   # "gallery" kept readable for old workflows
_lock = threading.RLock()
_HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Storage (user folder, outside the pack)
# ─────────────────────────────────────────────────────────────────────────────

def studio_dir() -> str:
    base = None
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
        base = os.path.join(base, "default") if os.path.isdir(os.path.join(base, "default")) else base
    except Exception:
        pass
    path = os.path.join(base, "onyx_studio") if base else os.path.join(os.path.dirname(_HERE), "_studio")
    os.makedirs(os.path.join(path, "skills"), exist_ok=True)
    return path


def _read(name: str, default):
    try:
        with open(os.path.join(studio_dir(), name), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write(name: str, data) -> None:
    path = os.path.join(studio_dir(), name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ── keys ─────────────────────────────────────────────────────────────────────

def saved_key(provider: str) -> str:
    return (_read("keys.json", {}) or {}).get(provider, "")


def keys_status() -> dict:
    keys = _read("keys.json", {}) or {}
    return {p: bool(keys.get(p)) for p in _llm.NEEDS_KEY}


def save_key(provider: str, key: str) -> None:
    with _lock:
        keys = _read("keys.json", {}) or {}
        if key.strip():
            keys[provider] = key.strip()
        else:
            keys.pop(provider, None)
        _write("keys.json", keys)


# ── characters ───────────────────────────────────────────────────────────────

def characters() -> list:
    data = _read("characters.json", [])
    return data if isinstance(data, list) else []


def character(char_id: str) -> dict:
    for c in characters():
        if c.get("id") == char_id:
            return c
    return {}


def save_character(c: dict) -> dict:
    """A character = its LoRA(s) and its trigger line. Nothing else is needed:
    the skill and the LoRA own the rest."""
    with _lock:
        items = characters()
        c = {
            "id": c.get("id") or uuid.uuid4().hex[:10],
            "name": str(c.get("name") or "Character")[:60],
            "trigger_line": str(c.get("trigger_line") or "")[:300],
            "loras": [
                {"lora": str(r.get("lora") or ""), "sm": float(r.get("sm", 1.0)),
                 "sc": float(r.get("sc", r.get("sm", 1.0))), "on": bool(r.get("on", True))}
                for r in (c.get("loras") or []) if isinstance(r, dict) and r.get("lora")
            ],
        }
        items = [x for x in items if x.get("id") != c["id"]] + [c]
        _write("characters.json", items)
        return c


def lora_trigger(*args, **kwargs):
    return _onyx_core()['lora_trigger'](*args, **kwargs)



def character_trigger(*args, **kwargs):
    return _onyx_core()['character_trigger'](*args, **kwargs)



def delete_character(char_id: str) -> None:
    with _lock:
        _write("characters.json", [c for c in characters() if c.get("id") != char_id])


# ── presets (10 slots, every setting of the node) ─────────────────────────────

PRESET_SLOTS = 10
_VOLATILE = {"tab", "gallery_q", "gallery_sel", "preview_media", "_galleryLoaded", "images", "run_token"}


def presets() -> dict:
    data = _read("master_presets.json", {})
    return data if isinstance(data, dict) else {}


def save_preset(slot: int, name: str, state: dict, widgets: dict) -> dict:
    if not 1 <= int(slot) <= PRESET_SLOTS:
        raise ValueError(f"slot must be 1-{PRESET_SLOTS}")
    with _lock:
        data = presets()
        data[str(int(slot))] = {
            "name": (name or f"Preset {slot}").strip()[:40],
            "state": {k: v for k, v in (state or {}).items() if k not in _VOLATILE},
            "widgets": {k: v for k, v in (widgets or {}).items() if k != "batch_data"},
            "saved": time.time(),
        }
        _write("master_presets.json", data)
        return data[str(int(slot))]


def delete_preset(slot: int) -> None:
    with _lock:
        data = presets()
        data.pop(str(int(slot)), None)
        _write("master_presets.json", data)


# ── history ──────────────────────────────────────────────────────────────────

def add_history(entry: dict) -> None:
    with _lock:
        hist = _read("history.json", [])
        hist = ([{**entry, "ts": time.time()}] + (hist if isinstance(hist, list) else []))[:60]
        _write("history.json", hist)


# ─────────────────────────────────────────────────────────────────────────────
# Skills
# ─────────────────────────────────────────────────────────────────────────────

_HAIR_RE = re.compile(r'HAIR LOCK\s+"([^"]{20,400})"')
_remote_cache = {"at": 0.0, "skills": {}}


def _skill_meta() -> dict:
    meta = _read("skills.json", {})
    return meta if isinstance(meta, dict) else {}


def _safe_filename(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9 ._()-]+", "_", name).strip(" ._") or "skill"
    if not base.lower().endswith((".txt", ".md")):
        base += ".txt"
    return base[:90]


def import_skill(name: str, text: str) -> dict:
    if not text.strip():
        raise ValueError("empty skill")
    fname = _safe_filename(name)
    with _lock:
        with open(os.path.join(studio_dir(), "skills", fname), "w", encoding="utf-8") as fh:
            fh.write(text)
        meta = _skill_meta()
        m = _HAIR_RE.search(text)
        meta[fname] = {
            "name": os.path.splitext(fname)[0],
            "locked_sentence": " ".join(m.group(1).split()) if m
                               else meta.get(fname, {}).get("locked_sentence", ""),
        }
        _write("skills.json", meta)
    return {"id": "file:" + fname, **meta[fname]}


def update_skill_meta(skill_id: str, name: str | None, locked_sentence: str | None) -> None:
    if not skill_id.startswith("file:"):
        raise ValueError("only imported skills can be edited")
    fname = skill_id[5:]
    with _lock:
        meta = _skill_meta()
        m = meta.setdefault(fname, {"name": os.path.splitext(fname)[0], "locked_sentence": ""})
        if name is not None:
            m["name"] = name.strip()[:80] or m["name"]
        if locked_sentence is not None:
            m["locked_sentence"] = locked_sentence.strip()
        _write("skills.json", meta)


def delete_skill(skill_id: str) -> None:
    if not skill_id.startswith("file:"):
        raise ValueError("only imported skills can be deleted")
    fname = os.path.basename(skill_id[5:])
    with _lock:
        try:
            os.remove(os.path.join(studio_dir(), "skills", fname))
        except OSError:
            pass
        meta = _skill_meta()
        meta.pop(fname, None)
        _write("skills.json", meta)


def _grok_presets() -> dict:
    """{"1": preset, ...} — string slots (the Grok node's loader returns int keys)."""
    try:
        from .grok_prompt import _load_presets
        return {str(k): v for k, v in (_load_presets() or {}).items()}
    except Exception:
        return {}




def list_presets() -> list:
    """Grok Prompt Generator presets, for recreate mode (image -> prompt)."""
    out = []
    for slot, p in sorted(_grok_presets().items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 99):
        text = (p or {}).get("prompt", "")
        if not text.strip():
            continue
        out.append({"slot": str(slot), "name": (p.get("name") or text.strip().splitlines()[0])[:70],
                    "skill": p.get("skill", ""), "model": p.get("model", ""),
                    "embedded_trigger": preset_trigger(p)})
    return out


def preset_trigger(*args, **kwargs):
    return _onyx_core()['preset_trigger'](*args, **kwargs)



def _worker_skills() -> dict:
    """Skills published by the Worker (bundle prompt_skills_core, dict SKILLS).
    Optional: no token / no bundle = none, silently."""
    if time.time() - _remote_cache["at"] < 300:
        return _remote_cache["skills"]
    skills = {}
    try:
        from .onyx_remote_exec import load_remote
        ns = load_remote("prompt_skills_core")
        raw = ns.get("SKILLS") or {}
        skills = {str(k): v for k, v in raw.items() if isinstance(v, dict) and v.get("text")}
    except Exception:
        skills = {}
    _remote_cache.update(at=time.time(), skills=skills)
    return skills


def list_skills(include_worker: bool = True) -> list:
    out = []
    folder = os.path.join(studio_dir(), "skills")
    meta = _skill_meta()
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".txt", ".md")):
            m = meta.get(fname, {})
            out.append({"id": "file:" + fname, "name": m.get("name") or os.path.splitext(fname)[0],
                        "kind": "system", "source": "imported",
                        "locked_sentence": m.get("locked_sentence", "")})
    if include_worker:
        for sid, s in _worker_skills().items():
            out.append({"id": "worker:" + sid, "name": s.get("name") or sid,
                        "kind": s.get("kind", "system"), "source": "worker",
                        "locked_sentence": s.get("locked_sentence", "")})
    return out


def resolve_skill(ref: str) -> str:
    """A skill id from a loose reference: an exact id, or a word found in an imported
    skill's name or file name ("candid" -> file:CANDID-SNAP-MASTER.txt)."""
    ref = (ref or "").strip()
    if not ref or ref.startswith(("file:", "worker:")):
        return ref
    low = ref.lower()
    for s in list_skills(include_worker=True):     # imported files first, then the Worker's
        if low in s["id"].lower() or low in s["name"].lower():
            return s["id"]
    return ""


def skill_text(skill_id: str) -> tuple:
    """-> (text, kind, meta)."""
    if skill_id.startswith("file:"):
        fname = os.path.basename(skill_id[5:])
        with open(os.path.join(studio_dir(), "skills", fname), encoding="utf-8") as fh:
            return fh.read(), "system", _skill_meta().get(fname, {})
    if skill_id.startswith("grok:"):
        p = _grok_presets().get(skill_id[5:]) or {}
        return p.get("prompt", ""), "instruction", p
    if skill_id.startswith("worker:"):
        s = _worker_skills().get(skill_id[7:]) or {}
        return s.get("text", ""), s.get("kind", "system"), s
    raise ValueError(f"unknown skill '{skill_id}'")


def _loose(*args, **kwargs):
    return _onyx_core()['_loose'](*args, **kwargs)



def render_skill(*args, **kwargs):
    return _onyx_core()['render_skill'](*args, **kwargs)



# ─────────────────────────────────────────────────────────────────────────────
# Prompt generation
# ─────────────────────────────────────────────────────────────────────────────

def _gallery_text(*args, **kwargs):
    return _onyx_core()['_gallery_text'](*args, **kwargs)



def _apply_trigger(*args, **kwargs):
    return _onyx_core()['_apply_trigger'](*args, **kwargs)



def _key_for(provider: str, keys: dict) -> str:
    return (keys.get(provider) or "").strip() or saved_key(provider)


def _llm_common(*args, **kwargs):
    return _onyx_core()['_llm_common'](*args, **kwargs)



def _chat(*args, **kwargs):
    return _onyx_core()['_chat'](*args, **kwargs)



def generate(*args, **kwargs):
    return _onyx_core()['generate'](*args, **kwargs)



# ─────────────────────────────────────────────────────────────────────────────
# The node
# ─────────────────────────────────────────────────────────────────────────────

# Last reference image seen by each Studio node (node id -> PIL): the Generate button uses it in Recreate.
_LAST_REF: dict = {}


def _tensor_to_pil(t):
    import numpy as np
    from PIL import Image
    arr = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


# ── Recreate images loaded in the panel (no loader node needed) ─────────────────
# Uploaded to ComfyUI's input folder (onyx_studio/...). The list lives in studio_state:
#   images: [name, ...] · image_pick: "sequential" | "selected" · preview_media: selected name
_SEQ: dict = {}          # node id -> {"sig": (...), "next": int}


def _input_path(name: str) -> str:
    import folder_paths
    return folder_paths.get_annotated_filepath(name)


def _load_input_image(name: str):
    from PIL import Image, ImageOps
    with Image.open(_input_path(name)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def panel_image(state: dict, unique_id, advance: bool = True):
    """(name, PIL) of the panel image this run uses, or (None, None)."""
    names = [n for n in (state.get("images") or []) if isinstance(n, str) and n]
    if not names:
        return None, None
    if state.get("image_pick") == "selected" and state.get("preview_media") in names:
        name = state["preview_media"]
    else:
        key = str(unique_id)
        sig = (tuple(names), str(state.get("run_token") or ""))
        st = _SEQ.get(key)
        if st is None or st["sig"] != sig:
            start = names.index(state["preview_media"]) if state.get("preview_media") in names else 0
            st = {"sig": sig, "next": start}
        name = names[st["next"] % len(names)]
        if advance:
            st["next"] = (st["next"] + 1) % len(names)
        _SEQ[key] = st
    try:
        return name, _load_input_image(name)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Onyx Studio: cannot read the image '{name}' ({e}). Remove it from the list.") from None


def _interruptible(fn, label: str = "LLM"):
    """Runs a blocking call (LLM / API request) in a thread so ComfyUI's Cancel (red cross) works:
    the run stops at once; the abandoned request finishes or times out on its own."""
    import threading as _th
    try:
        import comfy.model_management as _mm
    except Exception:  # noqa: BLE001
        return fn()
    box = {}

    def work():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e
    t = _th.Thread(target=work, daemon=True, name=f"onyx-studio-{label}")
    t.start()
    started, told = time.time(), False
    while t.is_alive():
        t.join(0.25)
        if _mm.processing_interrupted():
            print(f"⏹️  [Onyx Studio] cancelled while waiting for the {label}.")
            _mm.throw_exception_if_processing_interrupted()
        if not told and time.time() - started > 20:
            told = True
            print(f"⏳ [Onyx Studio] still waiting for the {label}… (Cancel works)")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _pil_to_tensor(pil):
    import numpy as np
    import torch
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0)[None]


def _placeholder():
    import torch
    return torch.zeros((1, 64, 64, 3))


class OnyxStudio:
    """Character LoRAs + prompt writing (manual, gallery, idea, recreate) + resolution + post look in one node."""

    DESCRIPTION = (
        "The creation hub: character LoRA + trigger (applied to MODEL / CLIP), the prompt — typed "
        "(optionally improved by a skill), sent from the Prompt Gallery, written from an idea by a skill, or "
        "recreated from the `image` / `video` input with your Grok presets — the resolution and the latent, "
        "and `post` (ONYX_POST), the look every Onyx post-processing node follows. LLM: Ollama, local "
        "Qwen3-VL, Grok, OpenRouter or Gemini (keys saved in Settings, on this computer)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"studio_state": ("STRING", {"default": "{}", "socketless": True})},
            "optional": {
                "model": ("MODEL", {"tooltip": "Diffusion model: the character's LoRAs are applied to it."}),
                "clip": ("CLIP", {"lazy": True,
                                  "tooltip": "Text encoder: LoRAs applied, then the prompt is encoded here. With "
                                             "the encode cache on (Settings) and a fixed prompt, it is not even "
                                             "loaded once the prompt has been encoded (unless the clip output "
                                             "is used)."}),
                "image": ("IMAGE", {"tooltip": "Reference image (e.g. from an Onyx Batch Loader): Recreate "
                                               "writes a prompt from it; 'from image' resolution follows it."}),
                "video": ("IMAGE", {"tooltip": "Reference video frames: used like `image` (its first frame) "
                                               "when no image is connected."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True,
                                 "tooltip": "Varies the LLM answer between runs (sent to the provider). "
                                            "Also resolves {a|b} choices in Grok presets."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "onyx_graph": "PROMPT"},
        }

    # Outputs are linked by index: the media outputs (image, video, audio, video_fps) are gone
    # with the Media tab — media now comes from a separate loader node.
    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "STRING",
                    "STRING", "STRING", "STRING", _onyx_post.POST_TYPE, "INT", "INT", "LATENT", "IMAGE")
    RETURN_NAMES = ("model", "clip", "positive", "negative", "prompt",
                    "notes", "post_preset", "ratio", "post", "width", "height", "latent", "image")
    # Shown on hover over each output dot (the compact panel has no output labels).
    OUTPUT_TOOLTIPS = (
        "model — the diffusion model with the character's LoRAs applied → KSampler / guider model.",
        "clip — the text encoder with the LoRAs applied (only if another node needs to encode text).",
        "positive — the encoded prompt → sampler / guider positive.",
        "negative — empty (zeroed) negative → sampler / guider negative.",
        "prompt — the final prompt text (e.g. → Prompt Saver, Show Text).",
        "notes — the skill's NOTES line (motion, ratio…), as text.",
        "post_preset — the camera motion preset name → Onyx Camera Shake preset_override "
        "(empty when motion is manual).",
        "ratio — the aspect ratio used, as text (e.g. 9:16).",
        "post — the post look + motion (ONYX_POST) → onyx_post of Onyx Post Studio and every Onyx post node.",
        "width — image width in pixels.",
        "height — image height in pixels.",
        "latent — empty latent at width × height (batch size from Format) → sampler latent_image.",
        "image — the source image of this run (the connected image, else the one loaded in the panel) → "
        "VAE Encode for image-to-image, or any image input.",
    )
    FUNCTION = "run"
    CATEGORY = "Onyx"

    @classmethod
    def IS_CHANGED(cls, studio_state="{}", seed=0, **kwargs):
        try:
            state = json.loads(studio_state or "{}")
        except ValueError:
            return float("nan")
        if state.get("mode") in ("idea", "recreate"):
            return float("nan")                     # new LLM answer / next batch item every run
        # Characters and skills live in their own files: editing them must re-run the node.
        char = json.dumps(character(state.get("character_id", "")), sort_keys=True)
        skill_mtime = ""
        sid = state.get("skill") or ""
        if sid.startswith("file:"):
            try:
                skill_mtime = str(os.path.getmtime(os.path.join(studio_dir(), "skills", os.path.basename(sid[5:]))))
            except OSError:
                pass
        return hashlib.sha256(f"{studio_state}|{seed}|{char}|{skill_mtime}".encode()).hexdigest()

    def check_lazy_status(self, studio_state="{}", unique_id=None, onyx_graph=None, clip=None, seed=0, **_):
        """The text encoder is needed unless the encode cache already holds this
        (fixed) prompt and nothing downstream uses the clip output."""
        if clip is not None or not _input_linked(onyx_graph, unique_id, "clip"):
            return []
        if _output_used(onyx_graph, unique_id, 1):          # 1 = the `clip` output
            return ["clip"]
        try:
            state = json.loads(studio_state or "{}")
        except ValueError:
            return ["clip"]
        text = _cacheable_prompt(state, int(seed or 0))
        if text is None or _cond_cache.lookup(text, "studio", _encode_key(state, onyx_graph, unique_id)) is None:
            return ["clip"]
        return []

    def run(self, studio_state="{}", unique_id=None, model=None, clip=None, image=None, video=None,
            seed=0, onyx_graph=None, **_legacy):
        ensure_profile_ready()
        try:
            state = json.loads(studio_state or "{}")
        except ValueError:
            state = {}

        # 1. Reference: the image input, else the first frame of the video input, else the
        #    images loaded in the panel (Recreate), one per run.
        ref = image if image is not None else (video[:1] if video is not None else None)
        media = None
        if ref is None and state.get("mode") == "recreate":
            media, pil_media = panel_image(state, unique_id)
            if pil_media is not None:
                ref = _pil_to_tensor(pil_media)
                print(f"🖼️  [Onyx Studio] source image: {media}")
        if ref is not None:
            _LAST_REF[str(unique_id)] = _tensor_to_pil(ref[0])     # the Generate button reuses it

        # 2. Prompt.
        pil = _tensor_to_pil(ref[0]) if ref is not None else None
        try:
            # API keys: the ones saved in Settings (on this computer, outside the pack).
            res = _interruptible(lambda: generate(state, image=pil, keys={}, seed=int(seed or 0)),
                                 f"writer model ({state.get('provider', 'ollama')})")
        except OnyxAPIError as e:
            raise RuntimeError(str(e)) from None
        prompt = res["prompt"]
        add_history({"prompt": prompt, "notes": res["notes"], "mode": res["mode"], "skill": res["skill"],
                     "preset": res.get("preset", "")})
        print(f"🎬 [Onyx Studio] {res['mode']} prompt ({len(prompt)} chars)"
              + (f" · NOTES: {res['notes']}" if res["notes"] else ""))

        # 3. Character LoRAs on MODEL / CLIP.
        rows = _lora_rows(state)
        if model is not None and rows:
            from .onyx_lora_trigger_stack import OnyxLoraTriggerStack
            model, clip, _, report, _ = OnyxLoraTriggerStack().apply(model, prompt, json.dumps(rows), clip)
        # 4. Encode.
        #    Fixed prompt (manual / gallery without a skill rewrite) + cache on: encoded
        #    once, then read from disk. A prompt that changes every run is never cached.
        positive = negative = None
        cached_text = _cacheable_prompt(state, int(seed or 0))
        cache_on = cached_text is not None and cached_text == prompt
        enc_key = _encode_key(state, onyx_graph, unique_id) if cache_on else ""
        if cache_on:
            positive = _cond_cache.lookup(prompt, "studio", enc_key)
            if positive is not None:
                print("♻️  [Onyx Studio] prompt encoding read from cache"
                      + (" — text encoder not loaded." if clip is None else "."))
        if positive is None and clip is not None:
            tokens = clip.tokenize(prompt)
            positive = clip.encode_from_tokens_scheduled(tokens)
            if cache_on:
                _cond_cache.store(prompt, "studio", positive, enc_key)
        if positive is not None:
            negative = _zero_out(positive)

        post = _onyx_post.make(state.get("post_look", "default"), state.get("post_motion", "none"),
                               auto_motion=res.get("post") or "")
        res["post_look"], res["post_motion"] = post["look"], post["motion"]
        try:
            width, height, ratio_label = resolution(state, res.get("ratio", ""), ref)
        except Exception as e:
            print(f"⚠️  [Onyx Studio] resolution: {e} — 1024x1024 used.")
            width, height, ratio_label = 1024, 1024, state.get("ratio", "1:1")
        res["width"], res["height"] = width, height
        batch = max(1, int((state.get("res") or {}).get("batch", 1) or 1))
        latent = empty_latent(width, height, batch)
        print(f"📐 [Onyx Studio] {width} x {height} ({ratio_label})")
        if media:
            res["media"] = media
        _notify(unique_id, res)
        return (model, clip, positive, negative, prompt, res["notes"],
                "" if post["motion"] == "manual" else post["motion"], ratio_label, post, width, height, latent,
                ref if ref is not None else _placeholder())


RES_ASPECTS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "5:4", "4:5", "from image (input)"]


def resolution(state: dict, skill_ratio: str, image=None):
    """(width, height, ratio label) with Onyx Resolution (MP)'s own engine.
    The ratio is the node's, or the skill's NOTES ratio when 'follow the skill'."""
    res = state.get("res") or {}
    ratio = state.get("ratio") or "9:16"
    if res.get("follow_skill") and skill_ratio in RES_ASPECTS:
        ratio = skill_ratio
    if ratio == "from image (input)" and image is None:
        ratio = "1:1"
    from .onyx_resolution_mp import OnyxResolutionMP
    out = OnyxResolutionMP().compute(megapixels=float(res.get("mp", 1.0)), aspect_ratio=ratio,
                                     multiple_of=int(res.get("mult", 32)), image=image)
    if isinstance(out, dict):
        out = out.get("result", ())
    return int(out[0]), int(out[1]), str(out[2] if len(out) > 2 else ratio)


def empty_latent(width: int, height: int, batch: int = 1) -> dict:
    """The same empty latent as ComfyUI's Empty Latent Image (4 channels, /8). The
    sampler adapts an all-zero latent to the model's own channel count (Krea 2,
    Flux, Qwen...), exactly as it does for the core node."""
    try:
        import nodes as _comfy_nodes
        out = _comfy_nodes.EmptyLatentImage().generate(width, height, batch)[0]
        if isinstance(out, dict) and hasattr(out.get("samples"), "shape"):
            return out
    except Exception:
        pass
    import torch
    return {"samples": torch.zeros([batch, 4, height // 8, width // 8])}


def _lora_rows(state: dict) -> list:
    """Character LoRAs + extra LoRAs, as rows for the LoRA Trigger Stack."""
    char = character(state.get("character_id", ""))
    rows = [{"lora": r["lora"], "mode": "ON", "sm": r.get("sm", 1.0), "sc": r.get("sc", 1.0)}
            for r in (char.get("loras") or []) if r.get("on", True)]
    rows += [{"lora": r["lora"], "mode": "ON", "sm": r.get("sm", 1.0), "sc": r.get("sc", 1.0)}
             for r in (state.get("extra_loras") or []) if r.get("lora") and r.get("on", True)]
    return rows


def _cacheable_prompt(state: dict, seed: int = 0):
    """The prompt when it is the same every run (manual / gallery, no skill
    rewrite) and the encode cache is on; None otherwise (nothing is cached)."""
    if not state.get("cond_cache", True) or state.get("mode", "manual") not in ("manual", "gallery"):
        return None
    if state.get("enhance") and state.get("skill"):
        return None                                   # an LLM rewrite: changes every run
    try:
        return generate(state, image=None, keys={}, seed=seed)["prompt"] or None
    except Exception:
        return None


def _encode_key(state: dict, graph, unique_id) -> str:
    """What makes an encoding unique besides the text: the encoder chain in the
    workflow + the LoRAs this node puts on CLIP (name + clip strength)."""
    loras = [[r["lora"], r.get("sc", 1.0)] for r in _lora_rows(state)]
    fp = _cond_cache.encoder_fingerprint(graph, unique_id, "clip")
    return hashlib.sha256(json.dumps([fp, loras], default=str).encode()).hexdigest()[:16]


def _input_linked(graph, unique_id, name: str) -> bool:
    if not isinstance(graph, dict) or unique_id is None:
        return True                                   # unknown graph: behave as before
    v = ((graph.get(str(unique_id)) or {}).get("inputs") or {}).get(name)
    return isinstance(v, list) and len(v) == 2


def _output_used(graph, unique_id, index: int) -> bool:
    if not isinstance(graph, dict) or unique_id is None:
        return True
    want = str(unique_id)
    for node in graph.values():
        for v in ((node or {}).get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) == want and v[1] == index:
                return True
    return False


def _zero_out(cond):
    """Same as ComfyUI's ConditioningZeroOut: the negative of a cfg-1 workflow."""
    import torch
    out = []
    for c, d in cond:
        d = dict(d)
        if d.get("pooled_output") is not None:
            d["pooled_output"] = torch.zeros_like(d["pooled_output"])
        out.append([torch.zeros_like(c), d])
    return out


def _notify(unique_id, res):
    if unique_id is None:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("onyx.studio.result", {"node": str(unique_id), **res})
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Routes (UI)
# ─────────────────────────────────────────────────────────────────────────────

def _lora_files() -> list:
    try:
        import folder_paths
        return folder_paths.get_filename_list("loras")
    except Exception:
        return []


def _pool_image(filename: str):
    from PIL import Image
    from .onyx_image_batch_loader import _ui_first_frame, _ui_is_video, _ui_pool_dir
    path = os.path.join(_ui_pool_dir(), os.path.basename(filename))
    if not os.path.isfile(path):
        raise RuntimeError("media file not found in the batch pool")
    return _ui_first_frame(path) if _ui_is_video(path) else Image.open(path).convert("RGB")


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    inst = PromptServer.instance
    if getattr(inst, "_onyx_studio_routes", False):
        return
    inst._onyx_studio_routes = True
    import asyncio

    def ok(**kw):
        return web.json_response({"ok": True, **kw})

    def err(e, status=400):
        return web.json_response({"ok": False, "error": str(e)}, status=status)

    @inst.routes.get("/onyx/studio/bootstrap")
    async def _bootstrap(_r):
        loop = asyncio.get_running_loop()
        skills = await loop.run_in_executor(None, list_skills)
        return ok(skills=skills, presets=list_presets(), characters=characters(), keys=keys_status(),
                  post_looks=[{"id": k, "info": _onyx_post.LOOK_INFO.get(k, "")} for k in _onyx_post.look_names()],
                  post_motions=_onyx_post.MOTIONS,
                  history=_read("history.json", [])[:30], loras=_lora_files(),
                  providers=[{"id": p, "label": _llm.PROVIDER_LABELS[p]} for p in _llm.PROVIDERS],
                  models={"xai": _reg.model_ids("grok", vision=True, include_deprecated=False),
                          "gemini": _reg.model_ids("gemini_text", include_deprecated=False),
                          "vertex": _reg.model_ids("gemini_text", include_deprecated=False)},
                  vertex_folder=saved_key("vertex") or _llm.DEFAULT_VERTEX_FOLDER,
                  modes=MODES, studio_dir=studio_dir(), presets_saved=presets(), res_aspects=RES_ASPECTS)

    @inst.routes.get("/onyx/studio/encode_cache")
    async def _encode_cache_stats(request):
        try:
            return ok(**_cond_cache.stats())
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/studio/encode_cache/clear")
    async def _encode_cache_clear(request):
        try:
            removed = _cond_cache.clear()
            return ok(removed=removed, **_cond_cache.stats())
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/studio/skill/import")
    async def _skill_import(request):
        try:
            body = await request.json()
            return ok(skill=import_skill(body.get("name") or "skill.txt", body.get("text") or ""))
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/studio/skill/meta")
    async def _skill_meta_route(request):
        try:
            body = await request.json()
            update_skill_meta(body.get("id", ""), body.get("name"), body.get("locked_sentence"))
            return ok()
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/studio/skill/delete")
    async def _skill_delete(request):
        try:
            delete_skill((await request.json()).get("id", ""))
            return ok()
        except Exception as e:
            return err(e)

    @inst.routes.get("/onyx/studio/skill/text")
    async def _skill_text(request):
        try:
            text, kind, meta = skill_text(request.query.get("id", ""))
            return ok(text=text, kind=kind)
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/studio/character/save")
    async def _char_save(request):
        try:
            return ok(character=save_character((await request.json()).get("character") or {}))
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/studio/character/delete")
    async def _char_delete(request):
        delete_character((await request.json()).get("id", ""))
        return ok()

    @inst.routes.get("/onyx/studio/presets")
    async def _presets(_r):
        return ok(presets=presets(), slots=PRESET_SLOTS)

    @inst.routes.post("/onyx/studio/preset/save")
    async def _preset_save(request):
        try:
            b = await request.json()
            return ok(preset=save_preset(int(b.get("slot", 0)), b.get("name", ""), b.get("state") or {},
                                         b.get("widgets") or {}), presets=presets())
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/studio/preset/delete")
    async def _preset_delete(request):
        delete_preset(int((await request.json()).get("slot", 0)))
        return ok(presets=presets())

    @inst.routes.get("/onyx/studio/lora_trigger")
    async def _lora_trigger(request):
        name = request.query.get("name", "")
        trig = await asyncio.get_running_loop().run_in_executor(None, lora_trigger, name)
        return ok(trigger=trig)

    @inst.routes.post("/onyx/studio/key")
    async def _key(request):
        body = await request.json()
        provider = body.get("provider", "")
        if provider not in _llm.NEEDS_KEY:
            return err("unknown provider")
        save_key(provider, body.get("key", ""))
        return ok(keys=keys_status())

    @inst.routes.get("/onyx/studio/vertex_keys")
    async def _vertex_keys(request):
        folder = request.query.get("folder") or saved_key("vertex") or _llm.DEFAULT_VERTEX_FOLDER
        keys = await asyncio.get_running_loop().run_in_executor(None, _llm.vertex_keys, folder)
        return ok(folder=folder, keys=[{"file": os.path.basename(f), "project": p} for f, p in keys])

    @inst.routes.get("/onyx/studio/ollama_models")
    async def _ollama(request):
        url = request.query.get("url") or _llm.DEFAULT_OLLAMA_URL
        try:
            models = await asyncio.get_running_loop().run_in_executor(None, _llm.ollama_models, url)
            return ok(models=models)
        except Exception as e:
            return err(e, 200)

    @inst.routes.get("/onyx/studio/history")
    async def _history(_r):
        return ok(history=_read("history.json", [])[:60])

    @inst.routes.post("/onyx/studio/preview")
    async def _preview(request):
        """Write one prompt now, without queueing: the Generate button."""
        try:
            body = await request.json()
            ensure_profile_ready()
            state = body.get("state") or {}

            def work():
                pil = None
                if state.get("mode") == "recreate" and body.get("media"):
                    pil = _load_input_image(str(body["media"]))      # an image loaded in the panel
                if pil is None:
                    pil = _LAST_REF.get(str(body.get("node_id") or ""))
                if state.get("mode") == "recreate" and pil is None:
                    raise RuntimeError("Recreate: load images in the Prompt tab, or connect an image to the "
                                       "`image` (or `video`) input and run the workflow once.")
                return generate(state, image=pil, seed=int(body.get("seed") or 0))
            res = await asyncio.get_running_loop().run_in_executor(None, work)
            add_history({"prompt": res["prompt"], "notes": res["notes"], "mode": res["mode"],
                         "skill": res["skill"], "preset": res.get("preset", ""), "preview": True})
            return ok(**res)
        except Exception as e:
            return err(e, 200)


NODE_CLASS_MAPPINGS = {"OnyxStudio": OnyxStudio}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxStudio": "Onyx Studio ✦"}
