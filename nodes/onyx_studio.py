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
becomes `notes` (+ `ratio`, `post_preset` for the Camera Shake node).

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
from ..utils import onyx_llm as _llm
from ..utils import onyx_registry as _reg
from ..utils.onyx_http import OnyxAPIError

MODES = ["manual", "gallery", "idea", "recreate"]
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
    with _lock:
        items = characters()
        c = {
            "id": c.get("id") or uuid.uuid4().hex[:10],
            "name": str(c.get("name") or "Character")[:60],
            "trigger_line": str(c.get("trigger_line") or "")[:300],
            "body": str(c.get("body") or "")[:600],
            "hair_lock": str(c.get("hair_lock") or "")[:600],
            "extra_rules": str(c.get("extra_rules") or "")[:3000],
            "loras": [
                {"lora": str(r.get("lora") or ""), "sm": float(r.get("sm", 1.0)), "sc": float(r.get("sc", r.get("sm", 1.0))),
                 "on": bool(r.get("on", True))}
                for r in (c.get("loras") or []) if isinstance(r, dict) and r.get("lora")
            ],
        }
        items = [x for x in items if x.get("id") != c["id"]] + [c]
        _write("characters.json", items)
        return c


def delete_character(char_id: str) -> None:
    with _lock:
        _write("characters.json", [c for c in characters() if c.get("id") != char_id])


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
_CONFIG_KEYS = ("TRIGGER_LINE", "BODY", "RATIO", "OUTPUT")
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
    try:
        from .grok_prompt import _load_presets
        return _load_presets() or {}
    except Exception:
        return {}


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
    for slot, p in sorted(_grok_presets().items(), key=lambda kv: str(kv[0])):
        text = (p or {}).get("prompt", "")
        if text.strip():
            first = text.strip().splitlines()[0][:48]
            out.append({"id": f"grok:{slot}", "name": f"Grok preset {slot} · {first}", "kind": "instruction",
                        "source": "grok", "locked_sentence": ""})
    if include_worker:
        for sid, s in _worker_skills().items():
            out.append({"id": "worker:" + sid, "name": s.get("name") or sid,
                        "kind": s.get("kind", "system"), "source": "worker",
                        "locked_sentence": s.get("locked_sentence", "")})
    return out


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


def _loose(sentence: str) -> str:
    """Regex for `sentence` that also matches it wrapped over several lines."""
    return r"\s+".join(re.escape(w) for w in sentence.split())


def render_skill(text: str, meta: dict, char: dict, ratio: str = "") -> str:
    """Fill the skill's CONFIG lines and swap its character-specific text for the
    selected character's: the original trigger line and body (also inside the
    examples, which otherwise pull the model back to the skill author's
    character) and the locked sentence (e.g. the hair lock)."""
    values = {
        "TRIGGER_LINE": char.get("trigger_line", ""),
        "BODY": char.get("body", ""),
        "RATIO": ratio,
        "OUTPUT": "chat",
    }
    for key in ("TRIGGER_LINE", "BODY"):
        new = (values[key] or "").strip()
        old = re.search(rf"^{key}\s*=\s*(.+?)\s*$", text, flags=re.M)
        if new and old and old.group(1).strip() and old.group(1).strip() != new:
            text = re.sub(_loose(old.group(1).strip()), lambda m, v=new: v, text)
    for key in _CONFIG_KEYS:
        val = (values.get(key) or "").strip()
        if val:
            text = re.sub(rf"^({key}\s*=)[^\n]*$", lambda m, v=val: f"{m.group(1)} {v}", text, flags=re.M)
    locked = " ".join((meta or {}).get("locked_sentence", "").split())
    hair = char.get("hair_lock", "").strip()
    if locked and hair and locked != hair:
        text = re.sub(_loose(locked), lambda m: hair, text)
    if char.get("extra_rules", "").strip():
        text += "\n\nCHARACTER RULES (override anything above)\n" + char["extra_rules"].strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Prompt generation
# ─────────────────────────────────────────────────────────────────────────────

def _gallery_text(entry_id: str, fallback: str) -> str:
    try:
        from .onyx_prompt_library import _load_index
        for e in _load_index():
            if str(e.get("id")) == str(entry_id):
                return e.get("prompt") or fallback
    except Exception:
        pass
    return fallback


def _apply_trigger(prompt: str, char: dict, force: bool, from_llm: bool = False) -> str:
    trig = (char.get("trigger_line") or "").strip()
    prompt = prompt.replace("{trigger}", trig).strip()
    if force and trig and not prompt.lower().startswith(trig.lower()):
        lines = prompt.splitlines()
        first = lines[0].strip() if lines else ""
        # An LLM answer that starts with ANOTHER trigger line (short, no full stop,
        # alone on its line): replace it instead of stacking two trigger lines.
        if from_llm and len(lines) > 1 and first and len(first) < 80 and not first.endswith("."):
            lines[0] = trig
            prompt = "\n".join(lines)
        else:
            prompt = f"{trig}\n{prompt}"
    return prompt.strip()


def _key_for(provider: str, keys: dict) -> str:
    return (keys.get(provider) or "").strip() or saved_key(provider)


def generate(state: dict, image=None, keys: dict | None = None, seed: int = 0) -> dict:
    """The prompt for one run. `image` is a PIL image (recreate mode)."""
    keys = keys or {}
    mode = state.get("mode", "manual")
    char = character(state.get("character_id", ""))
    force_trigger = bool(state.get("force_trigger", True))
    result = {"prompt": "", "notes": "", "ratio": "", "post": "", "mode": mode, "skill": ""}

    if mode == "manual":
        result["prompt"] = _apply_trigger(state.get("manual", ""), char, force_trigger)
        return result
    if mode == "gallery":
        g = state.get("gallery") or {}
        result["prompt"] = _apply_trigger(_gallery_text(g.get("id", ""), g.get("text", "")), char, force_trigger)
        return result

    skill_id = state.get("skill") or ""
    if not skill_id:
        raise RuntimeError("Onyx Studio: pick a skill (Prompt tab) for idea / recreate mode.")
    text, kind, meta = skill_text(skill_id)
    result["skill"] = skill_id
    provider = state.get("provider", "ollama")
    model = state.get("model", "")
    temperature = float(state.get("temperature", 0.7))
    max_tokens = int(state.get("max_tokens", 2048))
    extra = (state.get("extra") or "").strip()

    if mode == "recreate":
        if image is None:
            raise RuntimeError("Onyx Studio: recreate mode needs an image — add media to the batch "
                               "(Media tab) or connect the `image` input.")
        images = [image]
        request = "Recreate this reference photo."
    else:
        images = []
        request = (state.get("idea") or "").strip()
        if not request:
            raise RuntimeError("Onyx Studio: idea mode — write the idea in the Prompt tab.")
    if extra:
        request += f"\n{extra}"

    if kind == "instruction":                    # Grok preset style: the instruction goes with the image
        try:
            from .grok_prompt import _resolve_choices
            text, _ = _resolve_choices(text, seed)
        except Exception:
            pass
        system, user = "", f"{text.strip()}\n\n{request}"
    else:
        system = render_skill(text, meta, char, state.get("ratio", ""))
        user = (f"{request}\n\n(Answer the request directly. Do not reply READY. "
                f"One prompt, in one code block, then the NOTES line.)")

    common = dict(api_key=_key_for(provider, keys), temperature=temperature, max_tokens=max_tokens,
                  seed=seed or None, ollama_url=state.get("ollama_url") or _llm.DEFAULT_OLLAMA_URL)
    answer = _llm.chat(provider, model, system=system, user=user, images=images, **common)
    parsed = _llm.parse_answer(answer)

    if state.get("double_check") and kind != "instruction" and parsed["prompt"]:
        check = _llm.chat(provider, model, system=system,
                          user=("check this\n" + parsed["prompt"] +
                                "\n\nList nothing: return only the fixed prompt in one code block, "
                                "then the NOTES line."), images=[], **common)
        fixed = _llm.parse_answer(check)
        if fixed["prompt"]:
            fixed["notes"] = fixed["notes"] or parsed["notes"]
            fixed["ratio"] = fixed["ratio"] or parsed["ratio"]
            fixed["post"] = fixed["post"] or parsed["post"]
            parsed = fixed

    if kind == "instruction":
        trig = (meta or {}).get("trigger_word", "").strip()
        if trig and not parsed["prompt"].lower().startswith(trig.lower()):
            parsed["prompt"] = f"{trig} {parsed['prompt']}"
    parsed["prompt"] = _apply_trigger(parsed["prompt"], char, force_trigger, from_llm=True)
    result.update(parsed)
    if not result["prompt"]:
        raise RuntimeError("Onyx Studio: the model returned no prompt. Raw answer: " + answer[:300])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# The node
# ─────────────────────────────────────────────────────────────────────────────

def _batch_spec() -> dict:
    from .onyx_image_batch_loader import OnyxImageBatchLoader
    return OnyxImageBatchLoader.INPUT_TYPES()


def _tensor_to_pil(t):
    import numpy as np
    from PIL import Image
    arr = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def _placeholder():
    import torch
    return torch.zeros((1, 64, 64, 3))


class OnyxStudio:
    """Media batch + character LoRAs + prompt writing (manual, gallery, idea, recreate) in one node."""

    DESCRIPTION = (
        "The creation hub: image + video batch, character LoRAs (applied to MODEL / CLIP), and the "
        "prompt — typed, picked from the Prompt Gallery, or written by a skill through Ollama, local "
        "Qwen3-VL, Grok, OpenRouter or Gemini (idea mode, or recreate mode from each batch image). "
        "Outputs the patched MODEL / CLIP, positive and zeroed negative conditioning, the prompt, the "
        "media, and the skill's notes (ratio, post preset for Onyx Camera Shake)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = _batch_spec()
        required = dict(spec["required"])          # batch_data FIRST: the media panel relies on it
        required["studio_state"] = ("STRING", {"default": "{}", "socketless": True})
        optional = dict(spec.get("optional", {}))
        optional.update({
            "model": ("MODEL", {"tooltip": "Diffusion model: the character's LoRAs are applied to it."}),
            "clip": ("CLIP", {"tooltip": "Text encoder: LoRAs applied, then the prompt is encoded here."}),
            "image": ("IMAGE", {"tooltip": "Optional reference image. Overrides the batch item for "
                                           "recreate mode."}),
            "grok_api_key": ("STRING", {"forceInput": True}),
            "gemini_api_key": ("STRING", {"forceInput": True}),
            "openrouter_api_key": ("STRING", {"forceInput": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True,
                             "tooltip": "Varies the LLM answer between runs (sent to the provider). "
                                        "Also resolves {a|b} choices in Grok presets."}),
        })
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "STRING",
                    "IMAGE", "IMAGE", "AUDIO", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "positive", "negative", "prompt",
                    "image", "video", "audio", "video_fps", "notes", "post_preset", "ratio")
    FUNCTION = "run"
    CATEGORY = "Onyx"

    @classmethod
    def IS_CHANGED(cls, studio_state="{}", batch_data="{}", seed=0, **kwargs):
        try:
            state = json.loads(studio_state or "{}")
            batch = json.loads(batch_data or "{}")
        except ValueError:
            return float("nan")
        if state.get("mode") in ("idea", "recreate") or (batch.get("order") or []):
            return float("nan")                     # new LLM answer / next batch item every run
        return hashlib.sha256(f"{studio_state}|{seed}".encode()).hexdigest()

    def run(self, studio_state="{}", unique_id=None, model=None, clip=None, image=None,
            grok_api_key=None, gemini_api_key=None, openrouter_api_key=None, seed=0, **batch_kwargs):
        ensure_profile_ready()
        try:
            state = json.loads(studio_state or "{}")
        except ValueError:
            state = {}

        # 1. Media: the next batch item (same engine as the Batch Loader node).
        out_image, out_video, audio, fps = None, None, None, 0.0
        try:
            has_media = bool(json.loads(batch_kwargs.get("batch_data") or "{}").get("order"))
        except ValueError:
            has_media = False
        if has_media:
            from .onyx_image_batch_loader import OnyxImageBatchLoader
            res = OnyxImageBatchLoader().load_next(unique_id=unique_id, **batch_kwargs)
            if isinstance(res, dict):
                res = res.get("result", ())
            out_image, out_video, fps = res[0], res[1], float(res[2] or 0.0)
            audio = res[5] if len(res) > 5 else None
        ref = image if image is not None else out_image
        if ref is not None and ref.shape[1] <= 64 and out_video is not None and out_video.shape[0] > 1:
            ref = out_video[:1]                       # a video item: its first frame is the reference

        # 2. Prompt.
        keys = {"xai": grok_api_key or "", "gemini": gemini_api_key or "", "openrouter": openrouter_api_key or ""}
        pil = _tensor_to_pil(ref[0]) if ref is not None else None
        try:
            res = generate(state, image=pil, keys=keys, seed=int(seed or 0))
        except OnyxAPIError as e:
            raise RuntimeError(str(e)) from None
        prompt = res["prompt"]
        add_history({"prompt": prompt, "notes": res["notes"], "mode": res["mode"], "skill": res["skill"]})
        print(f"🎬 [Onyx Studio] {res['mode']} prompt ({len(prompt)} chars)"
              + (f" · NOTES: {res['notes']}" if res["notes"] else ""))

        # 3. Character LoRAs on MODEL / CLIP.
        char = character(state.get("character_id", ""))
        rows = [{"lora": r["lora"], "mode": "ON", "sm": r.get("sm", 1.0), "sc": r.get("sc", 1.0)}
                for r in (char.get("loras") or []) if r.get("on", True)]
        rows += [{"lora": r["lora"], "mode": "ON", "sm": r.get("sm", 1.0), "sc": r.get("sc", 1.0)}
                 for r in (state.get("extra_loras") or []) if r.get("lora") and r.get("on", True)]
        if model is not None and rows:
            from .onyx_lora_trigger_stack import OnyxLoraTriggerStack
            model, clip, _, report, _ = OnyxLoraTriggerStack().apply(model, prompt, json.dumps(rows), clip)
        # 4. Encode.
        positive = negative = None
        if clip is not None:
            tokens = clip.tokenize(prompt)
            positive = clip.encode_from_tokens_scheduled(tokens)
            negative = _zero_out(positive)

        _notify(unique_id, res)
        return (model, clip, positive, negative, prompt,
                out_image if out_image is not None else (image if image is not None else _placeholder()),
                out_video if out_video is not None else _placeholder(),
                audio, fps, res["notes"], res["post"] or "none", res["ratio"] or state.get("ratio", ""))


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
        return ok(skills=skills, characters=characters(), keys=keys_status(),
                  history=_read("history.json", [])[:30], loras=_lora_files(),
                  providers=[{"id": p, "label": _llm.PROVIDER_LABELS[p]} for p in _llm.PROVIDERS],
                  models={"xai": _reg.model_ids("grok", vision=True, include_deprecated=False),
                          "gemini": _reg.model_ids("gemini_text", include_deprecated=False)},
                  modes=MODES, studio_dir=studio_dir())

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

    @inst.routes.post("/onyx/studio/key")
    async def _key(request):
        body = await request.json()
        provider = body.get("provider", "")
        if provider not in _llm.NEEDS_KEY:
            return err("unknown provider")
        save_key(provider, body.get("key", ""))
        return ok(keys=keys_status())

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
                pil = _pool_image(body["media"]) if body.get("media") else None
                return generate(state, image=pil, seed=int(body.get("seed") or 0))
            res = await asyncio.get_running_loop().run_in_executor(None, work)
            add_history({"prompt": res["prompt"], "notes": res["notes"], "mode": res["mode"],
                         "skill": res["skill"], "preview": True})
            return ok(**res)
        except Exception as e:
            return err(e, 200)


NODE_CLASS_MAPPINGS = {"OnyxStudio": OnyxStudio}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxStudio": "Onyx Studio ✦"}
