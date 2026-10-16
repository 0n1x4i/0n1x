# -*- coding: utf-8 -*-
"""Deterministic pose-only carousel director for FLUX.2 Klein."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

from ..onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('krea2_bundle', 'carousel', __package__)
    return ns




MAX_SEED = (1 << 64) - 1
MAX_SLIDES = 4

LOCK_KEYS = (
    "identity",
    "facial_structure",
    "body_proportions",
    "skin_details",
    "hairstyle",
    "outfit",
    "accessories",
    "location",
    "background_geometry",
    "camera_position",
    "framing",
    "lens",
    "expression",
    "gaze",
    "lighting_direction",
    "color_palette",
    "photographic_style",
)

DEFAULT_GLOBAL_LOCK = {
    "identity": "Use the exact same adult person from Master Image 1; Master Image 1 is the sole identity source.",
    "facial_structure": "Preserve facial bone structure, eye spacing, nose, lips, jawline and cheek volume from Master Image 1.",
    "body_proportions": "Preserve shoulder width, torso-to-leg ratio, limb length and body volume from Master Image 1.",
    "skin_details": "Preserve complexion, skin texture and all visible stable marks from Master Image 1.",
    "hairstyle": "Preserve hair length, parting, volume, color and finish from Master Image 1.",
    "outfit": "Preserve every garment, design, color, texture, material and coverage from Master Image 1; allow only physically necessary fold and tension changes caused by the new pose.",
    "accessories": "Preserve the same jewelry, footwear, eyewear and handheld items from Master Image 1.",
    "location": "Preserve the exact location and environmental identity from Master Image 1.",
    "background_geometry": "Preserve background objects, their geometry, depth relationships and placement from Master Image 1.",
    "camera_position": "Preserve the camera position, height, angle and orbit from Master Image 1.",
    "framing": "Preserve crop, subject scale, headroom and composition from Master Image 1.",
    "lens": "Preserve perspective, focal-length character and depth of field from Master Image 1.",
    "expression": "Preserve the facial expression from Master Image 1.",
    "gaze": "Preserve eye direction and gaze target from Master Image 1.",
    "lighting_direction": "Preserve key-light direction, intensity, color, fill ratio and shadow pattern from Master Image 1.",
    "color_palette": "Preserve white balance, exposure family, contrast and dominant hues from Master Image 1.",
    "photographic_style": "Preserve photographic realism, finish, detail density and processing from Master Image 1.",
}


def _default_slide(index: int) -> dict:
    return {
        "index": index,
        "purpose": f"Independent re-pose edit using Pose Reference {index}.",
        "action": "Transfer the selected guide's joint configuration and body posture only.",
        "pose": f"Use Pose Reference {index} as the sole pose guide.",
        "expression": "Locked to Master Image 1.",
        "framing": "Locked to Master Image 1.",
        "camera": "Locked to Master Image 1.",
        "lens": "Locked to Master Image 1.",
        "background": "Locked to Master Image 1.",
        "allowed_changes": ["pose"],
        "locked_attributes": list(LOCK_KEYS),
        "transition": "This frame is generated independently; never use another generated slide as a reference.",
        "prompt_delta": (
            f"Use DWPose Map {index}, routed to Painter Image 2, only for body pose, limb placement, "
            "hand placement and weight distribution. Treat the colored skeleton on its blank background as "
            "geometry only; it contributes no appearance or scene attributes."
        ),
    }


DEFAULT_SLIDES = [_default_slide(index) for index in range(1, MAX_SLIDES + 1)]

DEFAULT_STATE = {
    "version": 2,
    "project": "FLUX.2 Klein pose-only carousel (1-4 slides)",
    "target": "FLUX.2 Klein 9B",
    "slide_count": 1,
    "slide_index": 1,
    "seed_base": 12345,
    "seed_mode": "Sequential",
    "seed_stride": 1,
    "grain_seed_offset": 1_000_000,
    "require_all_pose_references": True,
    "base_prompt": "",
    "base_negative_prompt": "",
    "global_lock": DEFAULT_GLOBAL_LOCK,
    "slides": DEFAULT_SLIDES,
    "lock_generated_prompt": False,
    "locked_prompts": {},
}

DEFAULT_STATE_JSON = json.dumps(DEFAULT_STATE, ensure_ascii=False, separators=(",", ":"))


def _text(value, limit=4000):
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value or "")).strip()
    return value[:limit]


def _integer(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _boolean(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _string_list(value, limit=32):
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:limit]:
        item = _text(item, 240)
        if item and item not in result:
            result.append(item)
    return result


def _normalise_slide(raw, position):
    raw = raw if isinstance(raw, dict) else {}
    fallback = _default_slide(position)
    slide = {"index": position}
    for key in (
        "purpose", "action", "pose", "expression", "framing", "camera", "lens",
        "background", "transition", "prompt_delta",
    ):
        slide[key] = _text(raw.get(key, fallback[key]), 1800)
    slide["allowed_changes"] = _string_list(raw.get("allowed_changes")) or ["pose"]
    slide["locked_attributes"] = _string_list(raw.get("locked_attributes")) or list(LOCK_KEYS)
    return slide


def read_carousel_state(value):
    if isinstance(value, str):
        try:
            incoming = json.loads(value or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"[Onyx Carousel] carousel_state contains invalid JSON: {exc}") from exc
    elif isinstance(value, dict):
        incoming = value
    else:
        incoming = {}
    if not isinstance(incoming, dict):
        raise ValueError("[Onyx Carousel] carousel_state must be a JSON object.")

    source_version = _integer(incoming.get("version"), 1, 1, 99)
    state = deepcopy(DEFAULT_STATE)
    state["project"] = _text(incoming.get("project", state["project"]), 300)
    state["target"] = _text(incoming.get("target", state["target"]), 100) or "FLUX.2 Klein 9B"
    state["slide_count"] = _integer(incoming.get("slide_count"), 1, 1, MAX_SLIDES)
    state["slide_index"] = _integer(incoming.get("slide_index"), 1, 1, MAX_SLIDES)
    state["seed_base"] = _integer(incoming.get("seed_base"), 12345, 0, MAX_SEED)
    seed_mode = _text(incoming.get("seed_mode", "Sequential"), 30).title()
    state["seed_mode"] = seed_mode if seed_mode in {"Fixed", "Sequential", "Stride"} else "Sequential"
    state["seed_stride"] = _integer(incoming.get("seed_stride"), 1, 1, 10_000_000)
    state["grain_seed_offset"] = _integer(incoming.get("grain_seed_offset"), 1_000_000, 0, MAX_SEED)
    state["require_all_pose_references"] = _boolean(
        incoming.get("require_all_pose_references", incoming.get("require_all_references", True)), True
    )
    state["base_prompt"] = _text(incoming.get("base_prompt"), 12_000)
    state["base_negative_prompt"] = _text(incoming.get("base_negative_prompt"), 8_000)

    # Version 1 used identity/scene/outfit references. Never carry that incompatible
    # contract into the pose-only design when an old workflow is opened.
    if source_version >= 2:
        lock = incoming.get("global_lock")
        if isinstance(lock, dict):
            cleaned = {
                _text(key, 80): _text(val, 1400)
                for key, val in list(lock.items())[:48]
                if _text(key, 80) and _text(val, 1400)
            }
            if cleaned:
                state["global_lock"] = cleaned

        raw_slides = incoming.get("slides")
        if isinstance(raw_slides, list) and raw_slides:
            state["slides"] = [_normalise_slide(item, i + 1) for i, item in enumerate(raw_slides[:MAX_SLIDES])]
            state["slide_count"] = min(state["slide_count"], len(state["slides"]))
    else:
        state["project"] = state["project"].replace("carousel", "pose-only carousel")

    state["slide_index"] = min(state["slide_index"], state["slide_count"])
    state["lock_generated_prompt"] = _boolean(incoming.get("lock_generated_prompt"), False)
    state["locked_prompts"] = {}
    locked = incoming.get("locked_prompts")
    if isinstance(locked, dict):
        for key, item in list(locked.items())[:MAX_SLIDES]:
            if isinstance(item, str):
                prompt, negative = _text(item, 24_000), ""
            elif isinstance(item, dict):
                prompt = _text(item.get("prompt"), 24_000)
                negative = _text(item.get("negative_prompt"), 8_000)
            else:
                continue
            if prompt:
                state["locked_prompts"][str(key)] = {"prompt": prompt, "negative_prompt": negative}
    return state


def _attribute_key(*args, **kwargs):
    return _onyx_core()['_attribute_key'](*args, **kwargs)



def _directive_conflicts(*args, **kwargs):
    return _onyx_core()['_directive_conflicts'](*args, **kwargs)



def validate_carousel(*args, **kwargs):
    return _onyx_core()['validate_carousel'](*args, **kwargs)



def _seed_for_slide(*args, **kwargs):
    return _onyx_core()['_seed_for_slide'](*args, **kwargs)



def build_master_prompt(*args, **kwargs):
    return _onyx_core()['build_master_prompt'](*args, **kwargs)



def build_slide_prompt(*args, **kwargs):
    return _onyx_core()['build_slide_prompt'](*args, **kwargs)



def build_negative_prompt(*args, **kwargs):
    return _onyx_core()['build_negative_prompt'](*args, **kwargs)



def generate_carousel_outputs(*args, **kwargs):
    return _onyx_core()['generate_carousel_outputs'](*args, **kwargs)


class OnyxPoseCarouselDirector:
    """Master + pose-guide carousel director (V1 port of NodoForgeLab2 Beta)."""

    SEARCH_ALIASES = ["carousel director", "pose carousel", "flux2 carousel", "klein carousel"]
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Generates one to four slides and selects only the active pose guides. "
        "The master image keeps everything except the pose; this is not a face swap."
    )
    CATEGORY = "Onyx/Krea2 Carousel"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = (
        "prompt", "negative_prompt", "master_prompt", "manifest", "validation_report",
        "seed", "grain_seed", "slide_index", "slide_count", "selected_pose",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "carousel_state": ("STRING", {
                    "default": DEFAULT_STATE_JSON, "socketless": True, "advanced": True,
                    "tooltip": "Serialized state managed by the Carousel Director UI.",
                }),
            },
            "optional": {
                "master_image": ("IMAGE", {"display_name": "MASTER IMAGE · LOCK EVERYTHING"}),
                "pose_1": ("IMAGE", {"display_name": "POSE 1 · POSE ONLY"}),
                "pose_2": ("IMAGE", {"display_name": "POSE 2 · POSE ONLY"}),
                "pose_3": ("IMAGE", {"display_name": "POSE 3 · POSE ONLY"}),
                "pose_4": ("IMAGE", {"display_name": "POSE 4 · POSE ONLY"}),
                "base_prompt": ("STRING", {"display_name": "BASE PROMPT (optional)", "forceInput": True}),
                "negative_prompt": ("STRING", {"display_name": "NEGATIVE (optional)", "forceInput": True}),
                "image_1": ("IMAGE", {"display_name": "LEGACY MASTER (v1)", "advanced": True}),
                "image_2": ("IMAGE", {"display_name": "LEGACY POSE 1 (v1)", "advanced": True}),
                "image_3": ("IMAGE", {"display_name": "LEGACY POSE 2 (v1)", "advanced": True}),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls, carousel_state, master_image=None, pose_1=None, pose_2=None, pose_3=None, pose_4=None,
        base_prompt=None, negative_prompt=None, image_1=None, image_2=None, image_3=None,
    ):
        try:
            state = read_carousel_state(carousel_state)
        except Exception:
            return float("nan")
        resolved_master = master_image if master_image is not None else image_1
        resolved_poses = (
            pose_1 if pose_1 is not None else image_2,
            pose_2 if pose_2 is not None else image_3,
            pose_3,
            pose_4,
        )
        payload = {
            "state": state,
            "base_prompt": _text(base_prompt, 12_000),
            "negative_prompt": _text(negative_prompt, 8_000),
            "master": resolved_master is not None,
            "poses": [item is not None for item in resolved_poses],
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def execute(
        self, carousel_state, master_image=None, pose_1=None, pose_2=None, pose_3=None, pose_4=None,
        base_prompt=None, negative_prompt=None, image_1=None, image_2=None, image_3=None,
    ):
        state = read_carousel_state(carousel_state)
        resolved_master = master_image if master_image is not None else image_1
        resolved_poses = (
            pose_1 if pose_1 is not None else image_2,
            pose_2 if pose_2 is not None else image_3,
            pose_3,
            pose_4,
        )
        result = generate_carousel_outputs(
            state,
            base_prompt=base_prompt or "",
            negative_prompt=negative_prompt or "",
            has_master=resolved_master is not None,
            pose_presence=tuple(item is not None for item in resolved_poses),
        )
        selected_pose = resolved_poses[result["slide_index"] - 1]
        return {
            "ui": {
                "generated_prompt": [result["prompt"]],
                "negative_prompt": [result["negative_prompt"]],
                "master_prompt": [result["master_prompt"]],
                "manifest": [result["manifest"]],
                "validation_report": [result["validation_report"]],
                "status": [result["status"]],
            },
            "result": (
                result["prompt"],
                result["negative_prompt"],
                result["master_prompt"],
                result["manifest"],
                result["validation_report"],
                result["seed"],
                result["grain_seed"],
                result["slide_index"],
                result["slide_count"],
                selected_pose,
            ),
        }
