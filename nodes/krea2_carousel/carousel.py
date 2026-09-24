# -*- coding: utf-8 -*-
"""Deterministic pose-only carousel director for FLUX.2 Klein."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy



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
            raise ValueError(f"[Onyx Carousel] carousel_state contiene JSON inválido: {exc}") from exc
    elif isinstance(value, dict):
        incoming = value
    else:
        incoming = {}
    if not isinstance(incoming, dict):
        raise ValueError("[Onyx Carousel] carousel_state debe ser un objeto JSON.")

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


def _attribute_key(value):
    value = re.sub(r"[^a-z0-9]+", "_", _text(value, 200).lower()).strip("_")
    aliases = {
        "body_pose": "pose", "hand_pose": "pose", "hands": "pose", "limb_placement": "pose",
        "face": "identity", "facial_identity": "identity", "body": "body_proportions",
        "proportions": "body_proportions", "hair": "hairstyle", "wardrobe": "outfit",
        "clothing": "outfit", "scene": "location", "environment": "location",
        "background": "background_geometry", "camera": "camera_position", "composition": "framing",
        "lighting": "lighting_direction", "palette": "color_palette", "style": "photographic_style",
    }
    return aliases.get(value, value)


def _directive_conflicts(text):
    lowered = _text(text, 5000).lower()
    change_words = r"(?:change|replace|different|new|alter|modify|switch|cambiar|reemplazar|nuevo|nueva|distinto|diferente)"
    topics = {
        "identity": r"(?:identity|person|face|identidad|persona|rostro)",
        "outfit": r"(?:outfit|clothing|garment|wardrobe|ropa|vestuario)",
        "location": r"(?:location|scene|environment|background|ubicación|escena|fondo)",
        "camera_position": r"(?:camera|angle|orbit|cámara|ángulo)",
        "framing": r"(?:framing|crop|composition|encuadre|recorte|composición)",
        "lighting_direction": r"(?:light|lighting|shadow|luz|iluminación|sombra)",
        "expression": r"(?:expression|smile|expresión|sonrisa)",
        "gaze": r"(?:gaze|eye direction|mirada)",
    }
    conflicts = []
    for key, topic in topics.items():
        if re.search(rf"{change_words}.{{0,36}}{topic}|{topic}.{{0,36}}{change_words}", lowered):
            conflicts.append(key)
    return conflicts


def validate_carousel(state, has_master, pose_presence):
    errors, warnings = [], []
    pose_presence = tuple(bool(item) for item in pose_presence[:MAX_SLIDES])
    if state["slide_index"] > state["slide_count"]:
        errors.append(f"El slide seleccionado ({state['slide_index']}) supera slide_count ({state['slide_count']}).")
    if len(state["slides"]) < state["slide_count"]:
        errors.append("El manifiesto contiene menos slides que slide_count.")
    if not has_master:
        errors.append("Falta Master Image 1.")
    selected = state["slide_index"] - 1
    required_indices = range(state["slide_count"]) if state["require_all_pose_references"] else (selected,)
    for index in required_indices:
        if index >= len(pose_presence) or not pose_presence[index]:
            errors.append(
                f"Falta Pose Reference {index + 1}; está activa dentro de un carousel de "
                f"{state['slide_count']} slide(s)."
            )
    if state["target"] != "FLUX.2 Klein 9B":
        warnings.append(
            f"El target {state['target']!r} está reservado para compatibilidad futura; "
            "esta versión fue validada para FLUX.2 Klein 9B."
        )
    if not state["global_lock"]:
        errors.append("global_lock está vacío.")

    for slide in state["slides"][:state["slide_count"]]:
        allowed = {_attribute_key(value) for value in slide["allowed_changes"]}
        forbidden = sorted(value for value in allowed if value != "pose")
        if forbidden:
            warnings.append(f"Slide {slide['index']}: sólo pose puede cambiar; se ignorarán: {', '.join(forbidden)}.")
        locked = {_attribute_key(value) for value in slide["locked_attributes"]}
        missing_locks = [key for key in LOCK_KEYS if key not in locked]
        if missing_locks:
            warnings.append(
                f"Slide {slide['index']}: faltan bloqueos locales ({', '.join(missing_locks[:8])}); "
                "el manifiesto global seguirá teniendo prioridad."
            )
        conflicts = _directive_conflicts(" ".join((slide.get("prompt_delta", ""), slide.get("purpose", ""))))
        if conflicts:
            warnings.append(
                f"Slide {slide['index']}: la dirección contradice atributos bloqueados: {', '.join(conflicts)}. "
                "El bloqueo de la imagen maestra tiene prioridad."
            )
        if not slide["prompt_delta"]:
            warnings.append(f"Slide {slide['index']}: prompt_delta está vacío; se usará el contrato pose-only.")
    if state["lock_generated_prompt"] and str(state["slide_index"]) not in state["locked_prompts"]:
        errors.append(f"LOCK está activo, pero no existe un prompt bloqueado para el slide {state['slide_index']}.")
    return errors, warnings


def _seed_for_slide(state, slide_index):
    offset = slide_index - 1
    if state["seed_mode"] == "Fixed":
        seed = state["seed_base"]
    elif state["seed_mode"] == "Stride":
        seed = state["seed_base"] + offset * state["seed_stride"]
    else:
        seed = state["seed_base"] + offset
    seed &= MAX_SEED
    return seed, (seed + state["grain_seed_offset"]) & MAX_SEED


def build_master_prompt(state, base_prompt=""):
    lines = [
        "Edit Painter Image 1 into exactly one seamless photographic frame.",
        "OUTPUT FORMAT: one image only. Never create a collage, grid, contact sheet, diptych, split screen, before/after comparison, inset, panel or multiple view.",
        "POSE-ONLY REFERENCE CONTRACT:",
        "- Painter Image 1 is the MASTER and the sole source for identity, face, body proportions, hair, outfit, accessories, location, background, camera, crop, lens, expression, gaze, lighting, color and photographic style.",
        "- Painter Image 2 is the SELECTED DWPOSE SKELETON MAP. Read only its body and hand joints, posture, limb placement and weight distribution.",
        "- Painter Image 2 contains geometry only. Do not invent appearance, clothing, hair, environment, camera, lighting, color or style from it.",
        "- Replace the original body pose completely with the skeleton pose. Do not retain, duplicate or ghost the original arm, hand or leg positions.",
        "- Generate exactly one person with one left arm and one right arm, and no additional limbs. A hand or arm may remain naturally occluded when it is not visible in the skeleton map; never expose, retain or invent a hidden hand merely to match the master.",
        "- Never use a previously generated slide as a reference; every slide starts from the same master image.",
        "GLOBAL CONTINUITY LOCK — these clauses override any conflicting instruction:",
    ]
    for key, value in state["global_lock"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    base_prompt = _text(base_prompt or state.get("base_prompt"), 12_000)
    if base_prompt:
        lines.extend(("ADDITIONAL CREATIVE DIRECTION (subject to all locks above):", base_prompt))
    return "\n".join(lines).strip()


def build_slide_prompt(state, slide_index, master_prompt=None):
    if slide_index < 1 or slide_index > state["slide_count"]:
        raise ValueError(f"[Onyx Carousel] slide_index {slide_index} fuera de rango.")
    slide = state["slides"][slide_index - 1]
    master_prompt = master_prompt or build_master_prompt(state)
    details = [
        f"- selected pose slot: Pose Reference {slide_index} (routed as Painter Image 2)",
        "- allowed change: pose only",
        "- locked: " + ", ".join(LOCK_KEYS),
    ]
    if slide.get("purpose"):
        details.append(f"- purpose: {slide['purpose']}")
    if slide.get("prompt_delta"):
        details.append(f"- pose direction: {slide['prompt_delta']}")
    return (
        master_prompt
        + "\n\nCURRENT FRAME DIRECTIVE:\n"
        + "\n".join(details)
        + "\nGenerate only this frame. If the pose guide conflicts with the locked camera or framing, adapt the pose to fit without moving the camera or changing the crop."
    ).strip()


def build_negative_prompt(state, supplied=""):
    parts = [
        _text(supplied or state.get("base_negative_prompt"), 8_000),
        "collage, grid, contact sheet, diptych, split screen, before and after, comparison layout, multiple views, multiple panels, inset image, retained original pose, ghost arms, duplicated arms, extra arms, extra hands, duplicate hands, extra limbs, duplicate body parts, malformed hands, identity drift, different person, face from pose guide, altered facial structure, changed body proportions, changed hairstyle, wardrobe substitution, changed accessories, outfit color drift, location replacement, background drift, camera movement, reframing, crop change, lens change, expression change, gaze change, lighting drift, color shift, pose-guide clothing, pose-guide background, watermark, unrequested text",
    ]
    seen, result = set(), []
    for part in parts:
        for phrase in (item.strip() for item in part.split(",")):
            key = phrase.lower()
            if phrase and key not in seen:
                seen.add(key)
                result.append(phrase)
    return ", ".join(result)


def generate_carousel_outputs(state, base_prompt="", negative_prompt="", has_master=True, pose_presence=(True,) * 4):
    errors, warnings = validate_carousel(state, has_master, pose_presence)
    if errors:
        raise ValueError("[Onyx Carousel] " + " ".join(errors))

    slide_index = state["slide_index"]
    master = build_master_prompt(state, base_prompt)
    negative = build_negative_prompt(state, negative_prompt)
    if state["lock_generated_prompt"]:
        locked = state["locked_prompts"][str(slide_index)]
        prompt = locked["prompt"]
        if locked.get("negative_prompt"):
            negative = locked["negative_prompt"]
        lock_status = "locked"
    else:
        prompt = build_slide_prompt(state, slide_index, master)
        lock_status = "generated"

    seed, grain_seed = _seed_for_slide(state, slide_index)
    prompt_records = []
    for index in range(1, state["slide_count"] + 1):
        candidate = state["locked_prompts"].get(str(index), {}).get("prompt") if state["lock_generated_prompt"] else None
        candidate = candidate or build_slide_prompt(state, index, master)
        prompt_records.append({
            "index": index,
            "pose_reference": index,
            "seed": _seed_for_slide(state, index)[0],
            "prompt_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        })

    manifest = {
        "schema": 2,
        "project": state["project"],
        "target": state["target"],
        "topology": "one_master_plus_selected_pose",
        "slide_count": state["slide_count"],
        "active_pose_references": list(range(1, state["slide_count"] + 1)),
        "selected_slide": slide_index,
        "selected_pose_reference": slide_index,
        "allowed_changes": ["pose"],
        "seed_base": state["seed_base"],
        "seed_mode": state["seed_mode"],
        "seed_stride": state["seed_stride"],
        "grain_seed_offset": state["grain_seed_offset"],
        "reference_presence": {
            "master_image": bool(has_master),
            "pose_1": bool(pose_presence[0]),
            "pose_2": bool(pose_presence[1]),
            "pose_3": bool(pose_presence[2]),
            "pose_4": bool(pose_presence[3]),
        },
        "global_lock": state["global_lock"],
        "slides": prompt_records,
    }
    report = {
        "ok": True,
        "target": state["target"],
        "selected_slide": slide_index,
        "selected_pose_reference": slide_index,
        "locked_prompt": state["lock_generated_prompt"],
        "locked_attributes": list(LOCK_KEYS),
        "allowed_changes": ["pose"],
        "warnings": warnings,
    }
    status = (
        f"Slide {slide_index}/{state['slide_count']} · Pose {slide_index} · seed {seed} · "
        f"{lock_status} · {len(warnings)} warning(s)"
    )
    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "master_prompt": master,
        "manifest": json.dumps(manifest, ensure_ascii=False, indent=2),
        "validation_report": json.dumps(report, ensure_ascii=False, indent=2),
        "seed": seed,
        "grain_seed": grain_seed,
        "slide_index": slide_index,
        "slide_count": state["slide_count"],
        "status": status,
    }

class OnyxPoseCarouselDirector:
    """Master + pose-guide carousel director (V1 port of NodoForgeLab2 Beta)."""

    SEARCH_ALIASES = ["carousel director", "pose carousel", "flux2 carousel", "klein carousel"]
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Genera de uno a cuatro slides y selecciona únicamente las guías de pose activas. "
        "La imagen maestra conserva todo salvo la pose; no es face swap."
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
                    "tooltip": "Estado serializado administrado por la interfaz del Carousel Director.",
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
