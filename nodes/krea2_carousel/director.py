# -*- coding: utf-8 -*-
"""Local visual prompt director for up to three routed image references."""

import base64
import io as bytes_io
import json
import re
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image


from . import local_vlm

from ..onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('krea2_bundle', 'director', __package__)
    return ns



MODES = [
    "Photography",
    "Photo Enhance",
    "Architecture",
    "Character",
    "Product",
    "Image Edit",
    "Style Transfer",
    "Dataset Caption",
    "Video",
    "Custom",
]

TARGETS = [
    "Generic",
    "Krea 2",
    "FLUX.2 Klein",
    "Z-Image",
    "Qwen Image",
    "MiniMax",
    "LTX 2.5",
]

CREATIVITY = ["Strict", "Balanced", "Creative", "Dice"]

DIRECTORS = [
    "General Director",
    "Prompt Enhancer",
    "Reverse Engineer",
    "Surgical Edit",
    "Face / Identity Analyst",
    "Subject Appearance Analyst",
    "Reference Composer",
    "Photography Director",
    "Smartphone Realism",
    "Arm's-Length Selfie",
    "Mirror Selfie",
    "First-Person POV",
    "Fashion Editorial",
    "Vintage / Analog",
    "Intimate Portrait",
    "Krea 2 High Detail",
    "Krea 2 Smartphone Realism",
    "Krea 2 Pose Lock",
    "Video Director",
    "MiniMax H3 Director",
    "Architecture Director",
    "Character Director",
    "Product Director",
    "Style Transfer Director",
    "Dataset Caption Director",
    "Maximum Detail Director",
    "Custom",
]

REFERENCE_KEYS = [
    ("subject", "Subject / appearance"),
    ("face_identity", "Face / identity"),
    ("body_proportions", "Body / proportions"),
    ("outfit", "Outfit"),
    ("pose", "Pose"),
    ("composition", "Composition"),
    ("camera", "Camera"),
    ("scene_environment", "Scene / environment"),
    ("lighting", "Lighting"),
    ("colors", "Colors"),
    ("mood_style", "Mood / style"),
    ("materials", "Materials"),
]

REFERENCE_OPTIONS = ["Auto", "Image 1", "Image 2", "Image 3", "Blend"]


MAP_PRESETS = {
    "Manual": None,
    "Identity 1 / Scene 2": {
        "subject": "Image 1",
        "face_identity": "Image 1",
        "body_proportions": "Image 1",
        "outfit": "Image 1",
        "pose": "Image 1",
        "composition": "Image 2",
        "camera": "Image 2",
        "scene_environment": "Image 2",
        "lighting": "Image 2",
        "colors": "Image 2",
        "mood_style": "Image 2",
        "materials": "Image 1",
    },
    "Outfit from 3": {
        "subject": "Image 1",
        "face_identity": "Image 1",
        "body_proportions": "Image 1",
        "outfit": "Image 3",
        "pose": "Image 2",
        "composition": "Image 2",
        "camera": "Image 2",
        "scene_environment": "Image 2",
        "lighting": "Image 2",
        "colors": "Image 2",
        "mood_style": "Image 2",
        "materials": "Image 3",
    },
    "Pose lock (Krea)": {
        "subject": "Image 1",
        "face_identity": "Image 1",
        "body_proportions": "Image 1",
        "outfit": "Image 1",
        "pose": "Image 2",
        "composition": "Image 2",
        "camera": "Image 2",
        "scene_environment": "Image 2",
        "lighting": "Image 2",
        "colors": "Image 2",
        "mood_style": "Image 2",
        "materials": "Image 1",
    },
    "Style only from 2": {
        "subject": "Image 1",
        "face_identity": "Image 1",
        "body_proportions": "Image 1",
        "outfit": "Image 1",
        "pose": "Image 1",
        "composition": "Image 1",
        "camera": "Image 1",
        "scene_environment": "Image 1",
        "lighting": "Image 2",
        "colors": "Image 2",
        "mood_style": "Image 2",
        "materials": "Image 2",
    },
}

# Targets that do NOT use a negative prompt (e.g. Krea 2 Turbo).
TARGETS_WITHOUT_NEGATIVE = {"Krea 2"}


PROMPT_LANGUAGES = {
    "English": "Write the final generator prompt in English.",
    "Spanish": "Write the final generator prompt in Spanish.",
    "Auto": "Use the language of the user's request.",
}

DEFAULT_STATE = {
    "version": 4,
    "request": "",
    "mode": "Photography",
    "target": "Krea 2",
    "creativity": "Balanced",
    "runtime": "Transformers local",
    "model_id": local_vlm.DEFAULT_MODEL,
    "endpoint": "http://127.0.0.1:11434",
    "director": "Photography Director",
    "preserve": {
        "subject": True,
        "face_identity": True,
        "body_proportions": True,
        "outfit": False,
        "pose": False,
        "materials": False,
        "composition": False,
        "scene_environment": False,
        "lighting": False,
        "camera": False,
        "colors": False,
        "mood_style": False,
    },
    "reference_map": {key: "Auto" for key, _ in REFERENCE_KEYS},
    "map_preset": "Manual",
    "intention_preset": "Identity + scene + outfit",
    "content_mode": "SFW",
    "prompt_language": "English",
    "trigger_phrase": "vixmodel",
    "run_mode": "Prompt only",
    "denoise_hint": "0.72–0.80 (outfit); lower to 0.65 if the pose is lost",
    "prompt_length": "Detailed",
    "workflow_rules": "",
    "custom_director_behavior": "",
    "lock_generated_prompt": False,
    "generated_prompt": "",
    "seed": 0,
    "four_bit": False,
    "attention": "auto",
    "unload_after": True,
    "max_image_edge": 1536,
    "timeout": 600,
}

DEFAULT_STATE_JSON = json.dumps(DEFAULT_STATE, ensure_ascii=False, separators=(",", ":"))





LENGTH_INSTRUCTIONS = {
    "Short": "Aim for 60 to 100 words.",
    "Medium": "Aim for 120 to 200 words.",
    "Detailed": "Aim for 220 to 350 words.",
    "Maximum Detail": "Aim for 350 to 550 words, but never pad with repetition.",
}

TEMPERATURES = {"Strict": 0.1, "Balanced": 0.4, "Creative": 0.7, "Dice": 1.0}
MAX_TOKENS = {"Short": 256, "Medium": 448, "Detailed": 768, "Maximum Detail": 1200}


def _choice(value, options, default):
    return value if value in options else default


def _bounded_text(value, limit, default=""):
    if value is None:
        return default
    value = str(value).replace("\x00", "").strip()
    return value[:limit]


def _bounded_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(minimum, min(value, maximum))


def _boolean(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return default


# Old Spanish option values from workflows saved before the English
# translation. Mapped on load so those workflows keep their presets instead of
# silently falling back to defaults.
LEGACY_VALUES = {
    "Identidad + escena + ropa": "Identity + scene + outfit",
    "Identidad + escena": "Identity + scene",
    "Identidad 1 / Escena 2": "Identity 1 / Scene 2",
    "Solo cambiar ropa": "Outfit change only",
    "Solo estilo de 2": "Style only from 2",
    "Ropa en 3": "Outfit from 3",
    "Close-up cara": "Face close-up",
    "Solo prompt": "Prompt only",
    "Cola completa": "Full queue",
    "Español": "Spanish",
    "0.72–0.80 (ropa); baja a 0.65 si se pierde la pose":
        "0.72–0.80 (outfit); lower to 0.65 if the pose is lost",
}


def _read_state(raw):
    if len(raw or "") > 256_000:
        raise ValueError("[Onyx Director] The internal state exceeds the allowed size.")
    try:
        incoming = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("[Onyx Director] The node's internal state is not valid JSON.") from exc
    if not isinstance(incoming, dict):
        raise ValueError("[Onyx Director] The internal state must be a JSON object.")

    for key in ("intention_preset", "map_preset", "run_mode", "prompt_language", "denoise_hint"):
        value = incoming.get(key)
        if isinstance(value, str) and value in LEGACY_VALUES:
            incoming[key] = LEGACY_VALUES[value]

    state = dict(DEFAULT_STATE)
    state.update(incoming)
    state["mode"] = _choice(state.get("mode"), MODES, DEFAULT_STATE["mode"])
    state["target"] = _choice(state.get("target"), TARGETS, DEFAULT_STATE["target"])
    state["creativity"] = _choice(state.get("creativity"), CREATIVITY, DEFAULT_STATE["creativity"])
    state["director"] = _choice(state.get("director"), DIRECTORS, DEFAULT_STATE["director"])
    state["runtime"] = _choice(
        state.get("runtime"), ["Transformers local", "Ollama", "llama.cpp"], DEFAULT_STATE["runtime"]
    )
    state["prompt_length"] = _choice(
        state.get("prompt_length"), list(LENGTH_INSTRUCTIONS), DEFAULT_STATE["prompt_length"]
    )
    state["prompt_language"] = _choice(
        state.get("prompt_language"), list(PROMPT_LANGUAGES), DEFAULT_STATE["prompt_language"]
    )
    state["attention"] = _choice(state.get("attention"), ["auto", "sdpa", "eager", "flash_attention_2"], "auto")

    preserve = dict(DEFAULT_STATE["preserve"])
    if isinstance(incoming.get("preserve"), dict):
        for key in preserve:
            preserve[key] = _boolean(incoming["preserve"].get(key), preserve[key])
    state["preserve"] = preserve

    reference_map = dict(DEFAULT_STATE["reference_map"])
    if isinstance(incoming.get("reference_map"), dict):
        for key in reference_map:
            reference_map[key] = _choice(incoming["reference_map"].get(key), REFERENCE_OPTIONS, "Auto")
    state["reference_map"] = reference_map

    state["max_image_edge"] = _bounded_int(state.get("max_image_edge"), 1536, 512, 3072)
    state["timeout"] = _bounded_int(state.get("timeout"), 600, 30, 3600)
    state["four_bit"] = _boolean(state.get("four_bit"), False)
    state["unload_after"] = _boolean(state.get("unload_after"), True)
    state["lock_generated_prompt"] = _boolean(state.get("lock_generated_prompt"), False)
    state["map_preset"] = _choice(state.get("map_preset"), list(MAP_PRESETS), "Manual")
    state["content_mode"] = _choice(state.get("content_mode"), ["SFW", "NSFW"], "SFW")
    state["run_mode"] = _choice(state.get("run_mode"), ["Prompt only", "Full queue"], "Prompt only")
    state["request"] = _bounded_text(state.get("request"), 12_000)
    state["model_id"] = _bounded_text(state.get("model_id"), 300, local_vlm.DEFAULT_MODEL)
    state["endpoint"] = _bounded_text(state.get("endpoint"), 500, DEFAULT_STATE["endpoint"])
    state["workflow_rules"] = _bounded_text(state.get("workflow_rules"), 12_000)
    state["custom_director_behavior"] = _bounded_text(
        state.get("custom_director_behavior"), 8_000
    )
    state["generated_prompt"] = _bounded_text(state.get("generated_prompt"), 24_000)
    state["trigger_phrase"] = _bounded_text(state.get("trigger_phrase"), 120)
    state["intention_preset"] = _bounded_text(state.get("intention_preset"), 120, "Manual")
    state["denoise_hint"] = _bounded_text(state.get("denoise_hint"), 300)
    state["seed"] = _bounded_int(state.get("seed"), 0, 0, (2**32) - 1)
    state["version"] = 4
    return state


def _tensor_to_pil(image, max_edge):
    if image is None:
        return None
    if getattr(image, "ndim", None) != 4 or image.shape[0] < 1 or image.shape[-1] < 3:
        raise ValueError("[Onyx Director] Each reference must be a valid ComfyUI IMAGE.")
    if image.shape[0] != 1:
        raise ValueError(
            "[Onyx Director] Each input accepts a single image, not a batch."
        )
    array = image[0, :, :, :3].detach().cpu().float().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(array, mode="RGB")
    if max(pil.size) > max_edge:
        scale = max_edge / max(pil.size)
        size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(size, Image.Resampling.LANCZOS)
    return pil


def _encode_image(pil):
    buffer = bytes_io.BytesIO()
    pil.save(buffer, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _local_endpoint(value, default):
    endpoint = (value or "").strip() or default
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("[Onyx Director] Invalid local endpoint.")
    if parsed.hostname.lower() not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(
            "[Onyx Director] For security, Ollama/llama.cpp only accept localhost."
        )
    if parsed.username or parsed.password:
        raise ValueError("[Onyx Director] The local endpoint must not include credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("[Onyx Director] The local endpoint does not accept a query or fragment.")
    return endpoint.rstrip("/")


def _apply_map_preset(*args, **kwargs):
    return _onyx_core()['_apply_map_preset'](*args, **kwargs)



def _resolve_reference_map(*args, **kwargs):
    return _onyx_core()['_resolve_reference_map'](*args, **kwargs)



def _preflight(*args, **kwargs):
    return _onyx_core()['_preflight'](*args, **kwargs)



def _build_prompts(*args, **kwargs):
    return _onyx_core()['_build_prompts'](*args, **kwargs)



def _clean_prompt(*args, **kwargs):
    return _onyx_core()['_clean_prompt'](*args, **kwargs)



def _ollama_generate(*args, **kwargs):
    return _onyx_core()['_ollama_generate'](*args, **kwargs)



def _llamacpp_generate(*args, **kwargs):
    return _onyx_core()['_llamacpp_generate'](*args, **kwargs)


def _negative_for_target(*args, **kwargs):
    return _onyx_core()['_negative_for_target'](*args, **kwargs)



def generate_directed_prompt(*args, **kwargs):
    return _onyx_core()['generate_directed_prompt'](*args, **kwargs)


class OnyxVisualPromptDirector:
    """Local-VLM visual prompt director (V1 port of NodoForgeLab2 Beta)."""

    SEARCH_ALIASES = ["onyx director", "nodoforge", "nodeforge", "visual prompt", "prompt director", "krea prompt", "qwen vl"]
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Generates a prompt with a local VLM. Routes identity, pose, camera, light, "
        "outfit (IMAGE 3) and scene across up to three references without mixing attributes."
    )
    CATEGORY = "Onyx/Krea2 Carousel"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "resolved_map", "negative_prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "director_state": ("STRING", {
                    "default": DEFAULT_STATE_JSON, "socketless": True, "advanced": True,
                    "tooltip": "Serialized state managed by the node UI.",
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {
                    "display_name": "IMAGE 1",
                    "tooltip": "Main reference (identity / subject). The map decides the attributes.",
                }),
                "image_2": ("IMAGE", {
                    "display_name": "IMAGE 2",
                    "tooltip": "Second reference (scene / pose / style depending on the map).",
                }),
                "image_3": ("IMAGE", {
                    "display_name": "IMAGE 3 (outfit)",
                    "tooltip": "Outfit / clothing reference. In Auto, outfit and materials come from here when connected.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, director_state, image_1=None, image_2=None, image_3=None):
        # Cache when locked on the same saved prompt; otherwise hash state so unchanged
        # settings + unchanged images (fingerprinted by Comfy) can skip work.
        try:
            state = _read_state(director_state)
        except Exception:
            return float("nan")
        if state.get("lock_generated_prompt"):
            return json.dumps(["lock", (state.get("generated_prompt") or "").strip()], ensure_ascii=False)
        # Exclude volatile UI-only fields from the fingerprint payload.
        payload = {
            k: state[k]
            for k in (
                "request", "mode", "target", "creativity", "runtime", "model_id", "endpoint",
                "director", "preserve", "reference_map", "map_preset", "intention_preset", "content_mode", "prompt_length",
                "prompt_language", "trigger_phrase",
                "workflow_rules", "custom_director_behavior", "seed", "four_bit",
                "attention", "max_image_edge",
            )
            if k in state
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def execute(self, director_state, image_1=None, image_2=None, image_3=None):
        state = _read_state(director_state)
        max_edge = state["max_image_edge"]
        pil_1 = _tensor_to_pil(image_1, max_edge)
        pil_2 = _tensor_to_pil(image_2, max_edge)
        pil_3 = _tensor_to_pil(image_3, max_edge)
        result = generate_directed_prompt(state, pil_1, pil_2, pil_3)
        resolved_json = json.dumps(result["resolved_map"], ensure_ascii=False)
        return {
            "ui": {
                "generated_prompt": [result["prompt"]],
                "status": [result["status"]],
                "resolved_map": [resolved_json],
            },
            "result": (result["prompt"], resolved_json, result["negative_prompt"]),
        }
