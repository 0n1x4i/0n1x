# -*- coding: utf-8 -*-
"""Krea 2 Identity Edit carousel director.

This node emits one plain-English edit instruction per requested slide.  It is
deliberately separate from the FLUX.2 Klein directors: Krea2Edit consumes an
existing master image through its VAE-token and Qwen3-VL grounding paths, so it
needs edit instructions rather than text-to-image descriptions.
"""

from __future__ import annotations

import json
import re



MAX_SLIDES = 8
OUTPUT_ALL = "All active slides"
OUTPUT_SELECTED = "Selected slide only"
CAMERA_POLICIES = ("AUTO", "LOCK", "FREE")
CONTINUITY_LEVELS = ("Balanced", "Strong", "Maximum")

DEFAULT_POSES = """standing naturally with both hands resting on her head
crossing both arms naturally over her torso
turning around and looking back toward the camera
taking a close-up selfie from a slightly high camera angle"""


def _clean(value, limit=24_000):
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value or "")).strip()
    return value[:limit]


def _pose_lines(value):
    result = []
    for raw in _clean(value).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip()
        if line and line not in result:
            result.append(line)
    return result[:MAX_SLIDES]


def _camera_contract(policy):
    if policy == "LOCK":
        return (
            "Keep the original camera position, lens perspective, crop, framing and subject scale. "
            "Adapt the requested pose to that unchanged camera setup."
        )
    if policy == "FREE":
        return (
            "The camera angle, distance, crop and framing may change as requested by the pose direction, "
            "while every continuity lock remains mandatory."
        )
    return (
        "Keep the established camera and framing unless the requested pose physically requires a different "
        "viewpoint or crop; permit only that necessary camera change."
    )


def _continuity_contract(level):
    if level == "Maximum":
        return (
            "Preserve the exact same adult identity, facial geometry and details, body proportions, skin marks, "
            "hairstyle, every garment, fabric detail, accessory, location, background geometry, lighting direction, "
            "color grade, texture and photographic finish from the reference image."
        )
    if level == "Balanced":
        return (
            "Keep the same recognizable adult person, hairstyle, complete outfit, location, lighting and overall "
            "photographic session from the reference image."
        )
    return (
        "Preserve the same adult identity, facial features, body proportions, hairstyle, complete outfit and "
        "accessories, location, background, lighting, color palette and photographic appearance from the reference."
    )


def _contradiction_warnings(text, camera_policy, label):
    lowered = text.lower()
    warnings = []
    locked_topics = {
        "identity": r"\b(?:different|another|new|change|replace|swap)\b.{0,36}\b(?:person|face|identity|woman|man)\b",
        "outfit": r"\b(?:change|replace|different|new|remove)\b.{0,36}\b(?:outfit|clothes|clothing|dress|shirt|wardrobe|bikini)\b",
        "scene": r"\b(?:change|replace|different|new|move)\b.{0,36}\b(?:scene|background|location|room|beach|street)\b",
        "lighting": r"\b(?:change|replace|different|new|relight)\b.{0,36}\b(?:light|lighting|shadow|sunset|night)\b",
    }
    for topic, pattern in locked_topics.items():
        if re.search(pattern, lowered):
            warnings.append(f"{label}: instruction may contradict locked {topic}.")
    if camera_policy == "LOCK" and re.search(
        r"\b(?:selfie|close[- ]?up|wide shot|from behind|rear view|profile|overhead|low angle|high angle|"
        r"camera|reframe|zoom|first-person|pov)\b",
        lowered,
    ):
        warnings.append(
            f"{label}: camera policy LOCK conflicts with a camera/viewpoint request; the original framing wins."
        )
    return warnings


def build_krea2_carousel(
    pose_list,
    slide_count=4,
    output_mode=OUTPUT_ALL,
    slide_selector=1,
    camera_policy="AUTO",
    continuity_strength="Strong",
    session_direction="",
    character_trigger="",
    has_master=True,
    extra_prompt="",
):
    """Compile a pose list into Krea2Edit instructions and a continuity manifest."""
    poses = _pose_lines(pose_list)
    requested_count = max(1, min(MAX_SLIDES, int(slide_count)))
    camera_policy = camera_policy if camera_policy in CAMERA_POLICIES else "AUTO"
    continuity_strength = (
        continuity_strength if continuity_strength in CONTINUITY_LEVELS else "Strong"
    )
    output_mode = output_mode if output_mode in (OUTPUT_ALL, OUTPUT_SELECTED) else OUTPUT_ALL

    warnings = []
    errors = []
    if not has_master:
        warnings.append(
            "Master image is not connected to the Director; it must still feed both Krea2EditGroundedEncode "
            "and Krea2EditModelPatch in the workflow."
        )
    if not poses:
        errors.append("Pose list is empty.")
    if len(poses) < requested_count:
        warnings.append(
            f"slide_count requests {requested_count} slides but only {len(poses)} non-empty pose lines exist."
        )

    active_count = min(requested_count, len(poses)) if poses else 0
    selector = max(1, min(max(1, active_count), int(slide_selector)))
    session_direction = _clean(session_direction)
    character_trigger = _clean(character_trigger, 1_000)
    extra_prompt = _clean(extra_prompt, 4_000)
    if session_direction:
        warnings.extend(_contradiction_warnings(session_direction, camera_policy, "Session direction"))

    master_parts = [
        "Edit the reference image and re-stage the same person; do not generate an unrelated person.",
        _continuity_contract(continuity_strength),
        _camera_contract(camera_policy),
        (
            "Only the requested body pose, expression, hand gesture and the camera adjustment permitted above may "
            "change. Reconstruct the arms, hands and body naturally instead of copying limb positions that conflict "
            "with the new pose. Produce one coherent photorealistic frame with one person, never a collage, grid or "
            "character sheet."
        ),
    ]
    if session_direction:
        master_parts.append("Additional session direction, subordinate to the locks: " + session_direction)
    master_instruction = " ".join(master_parts)

    all_prompts = []
    slide_records = []
    for index, pose in enumerate(poses[:active_count], start=1):
        warnings.extend(_contradiction_warnings(pose, camera_policy, f"Slide {index}"))
        instruction = (
            master_instruction
            + f" Requested change for slide {index}: {pose}. Apply this change decisively while preserving all "
            "locked attributes."
        )
        if character_trigger:
            instruction = character_trigger + ", " + instruction
        # Ajoute tel quel, en DERNIER, a chaque instruction : c'est le texte
        # que l'utilisateur veut voir arriver au node suivant sans retouche.
        if extra_prompt:
            instruction = instruction.rstrip() + " " + extra_prompt
        all_prompts.append(instruction)
        slide_records.append(
            {
                "index": index,
                "pose_instruction": pose,
                "allowed_changes": ["pose", "expression", "hand_gesture", "camera_if_policy_allows"],
            }
        )

    selected_prompt = all_prompts[selector - 1] if all_prompts else ""
    if output_mode == OUTPUT_SELECTED and selected_prompt:
        prompts = [selected_prompt]
        emitted_indices = [selector]
    else:
        prompts = all_prompts
        emitted_indices = list(range(1, active_count + 1))

    manifest = {
        "schema": 1,
        "target": "Krea 2 Turbo + Krea 2 Identity Edit v1.2",
        "conditioning": "dual: in-context VAE tokens + image-grounded Qwen3-VL",
        "topology": "one_master_reference_plus_text_pose_list",
        "recommended_settings": {
            "steps": 10,
            "cfg": 1.0,
            "identity_edit_lora_strength": 1.0,
            "ref_boost": 4.0,
            "grounding_px": 768,
            "fit_mode": "fit",
            "max_output_megapixels": 2.0,
        },
        "output_mode": output_mode,
        "camera_policy": camera_policy,
        "continuity_strength": continuity_strength,
        "requested_slide_count": requested_count,
        "active_slide_count": active_count,
        "selected_slide": selector,
        "emitted_slides": emitted_indices,
        "locked_attributes": [
            "identity",
            "facial_geometry",
            "body_proportions",
            "hairstyle",
            "outfit",
            "accessories",
            "location",
            "background",
            "lighting",
            "color_palette",
            "photographic_finish",
        ],
        "slides": slide_records,
    }
    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "locked_attributes": manifest["locked_attributes"],
        "allowed_to_change": ["pose", "expression", "hand gesture", "camera/framing per camera_policy"],
        "active_slide_count": active_count,
    }
    if errors:
        raise ValueError("[Onyx Krea2 Carousel] " + " ".join(errors))

    # CFG 1 does not use negative guidance, but a same-length empty list keeps
    # list execution aligned if the user later raises CFG and grounds negatives.
    negatives = ["" for _ in prompts]
    return {
        "prompts": prompts,
        "negative_prompts": negatives,
        "selected_prompt": selected_prompt,
        "master_instruction": master_instruction,
        "manifest": json.dumps(manifest, ensure_ascii=False, indent=2),
        "validation_report": json.dumps(report, ensure_ascii=False, indent=2),
        "slide_count": active_count,
        "selected_index": selector,
        "status": f"{len(prompts)} Krea2 edit instruction(s) · {camera_policy} camera · {len(warnings)} warning(s)",
    }

class OnyxKrea2CarouselDirector:
    """Krea2 Identity Edit carousel director (V1 port of NodoForgeLab2 Beta)."""

    SEARCH_ALIASES = ["krea2 edit carousel", "identity edit director", "krea pose carousel"]
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Builds one image-grounded Krea2 Identity Edit instruction per pose while locking the master "
        "image's identity, wardrobe and photographic session."
    )
    CATEGORY = "Onyx/Krea2 Carousel"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "edit_prompts", "empty_negative_prompts", "selected_prompt", "master_instruction",
        "manifest", "validation_report", "slide_count", "selected_index",
    )
    OUTPUT_IS_LIST = (True, True, False, False, False, False, False, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Les widgets sont pilotes par l'UI du node (JS) et caches : sans
                # "socketless", le frontend leur cree quand meme une entree, et ces
                # entrees de widgets caches s'empilent toutes au meme endroit en haut
                # du node — impossible de savoir sur laquelle on branche.
                "pose_list": ("STRING", {"default": DEFAULT_POSES, "multiline": True, "dynamicPrompts": False, "socketless": True}),
                "slide_count": ("INT", {"default": 4, "min": 1, "max": MAX_SLIDES, "step": 1, "socketless": True}),
                "output_mode": ([OUTPUT_ALL, OUTPUT_SELECTED], {"socketless": True}),
                "slide_selector": ("INT", {"default": 1, "min": 1, "max": MAX_SLIDES, "step": 1, "socketless": True}),
                "camera_policy": (list(CAMERA_POLICIES), {"socketless": True}),
                "continuity_strength": (list(CONTINUITY_LEVELS), {"socketless": True}),
            },
            "optional": {
                "master_image": ("IMAGE", {"display_name": "MASTER IMAGE · CANONICAL"}),
                # Vraies entrees, listees sous MASTER IMAGE, pour brancher le Pose
                # List Prep. Branchees, elles remplacent les widgets correspondants.
                "pose_list_in": ("STRING", {"forceInput": True, "display_name": "POSE LIST · FROM PREP"}),
                "slide_count_in": ("INT", {"forceInput": True, "display_name": "SLIDE COUNT · FROM PREP"}),
                "session_direction": ("STRING", {"default": "", "multiline": True, "socketless": True}),
                "character_trigger": ("STRING", {"default": "", "multiline": False, "socketless": True}),
                # En dernier : les valeurs des widgets sont enregistrees par position.
                "extra_prompt": ("STRING", {"default": "", "multiline": True, "socketless": True,
                                            "tooltip": "Appended verbatim to the end of EVERY edit "
                                                       "prompt sent to the next node."}),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        pose_list,
        slide_count,
        output_mode,
        slide_selector,
        camera_policy,
        continuity_strength,
        session_direction="",
        character_trigger="",
        master_image=None,
        pose_list_in=None,
        slide_count_in=None,
        extra_prompt="",
    ):
        if pose_list_in:
            pose_list = pose_list_in
        if slide_count_in is not None:
            slide_count = slide_count_in
        return json.dumps(
            {
                "pose_list": _clean(pose_list),
                "slide_count": slide_count,
                "output_mode": output_mode,
                "slide_selector": slide_selector,
                "camera_policy": camera_policy,
                "continuity_strength": continuity_strength,
                "session_direction": _clean(session_direction),
                "character_trigger": _clean(character_trigger),
                "extra_prompt": _clean(extra_prompt),
                "master_image": master_image is not None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def execute(
        self,
        pose_list,
        slide_count,
        output_mode,
        slide_selector,
        camera_policy,
        continuity_strength,
        session_direction="",
        character_trigger="",
        master_image=None,
        pose_list_in=None,
        slide_count_in=None,
        extra_prompt="",
    ):
        if pose_list_in:
            pose_list = pose_list_in
        if slide_count_in is not None:
            slide_count = slide_count_in
        result = build_krea2_carousel(
            pose_list=pose_list,
            slide_count=slide_count,
            output_mode=output_mode,
            slide_selector=slide_selector,
            camera_policy=camera_policy,
            continuity_strength=continuity_strength,
            session_direction=session_direction,
            character_trigger=character_trigger,
            has_master=master_image is not None,
            extra_prompt=extra_prompt,
        )
        return {
            "ui": {
                "status": [result["status"]],
                "selected_prompt": [result["selected_prompt"]],
                "validation_report": [result["validation_report"]],
                "pose_list_used": [pose_list],
            },
            "result": (
                result["prompts"],
                result["negative_prompts"],
                result["selected_prompt"],
                result["master_instruction"],
                result["manifest"],
                result["validation_report"],
                result["slide_count"],
                result["selected_index"],
            ),
        }
