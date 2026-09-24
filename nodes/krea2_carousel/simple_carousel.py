# -*- coding: utf-8 -*-
"""Text-driven carousel director for one canonical image and N pose prompts."""

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
covering her face with both hands"""

NEGATIVE_BASE = (
    "collage, grid, contact sheet, diptych, split screen, multiple panels, multiple people, "
    "duplicate person, identity drift, different person, altered outfit, wardrobe substitution, "
    "unrequested jewelry, added accessories, removed accessories, accessory drift, "
    "background replacement, extra arms, extra hands, extra fingers, duplicate limbs, malformed hands, "
    "ghost limbs, watermark, text"
)


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
            "Lock the original camera position, lens character, crop, subject scale and framing. "
            "Adapt the requested pose so it fits naturally inside that unchanged composition."
        )
    if policy == "FREE":
        return (
            "Camera position, angle, distance and framing may change when requested by the pose direction. "
            "All identity, wardrobe, location, lighting and photographic-session locks remain mandatory."
        )
    return (
        "Preserve the established camera and framing by default, but allow only the camera movement or "
        "reframing physically required to satisfy the requested pose, viewpoint or selfie direction."
    )


def _continuity_contract(level):
    if level == "Maximum":
        return (
            "Use the reference image as the exclusive canonical source for the adult subject's identity, "
            "facial anatomy, body proportions, hairstyle, every garment and accessory, location, background "
            "geometry, lighting direction, color palette, texture and photographic finish. Reproduce those "
            "attributes exactly; only the requested pose and explicitly permitted camera adjustment may change."
        )
    if level == "Strong":
        return (
            "Preserve the same adult subject, facial identity, body proportions, hairstyle, complete outfit, "
            "accessories, location, background, lighting, colors and photographic appearance from the reference. "
            "Only the requested pose and any camera adjustment allowed below may change."
        )
    return (
        "Keep the same adult subject, recognizable identity, hairstyle, outfit, location, lighting and overall "
        "photographic session from the reference while applying the requested pose."
    )


def _warnings_for_pose(pose, camera_policy, index):
    lowered = pose.lower()
    warnings = []
    locked_topics = {
        "identity": r"\b(?:different|another|new|change|replace)\b.{0,30}\b(?:person|face|identity|woman|man)\b",
        "outfit": r"\b(?:change|replace|different|new)\b.{0,30}\b(?:outfit|clothes|clothing|dress|shirt|wardrobe)\b",
        "scene": r"\b(?:change|replace|different|new)\b.{0,30}\b(?:scene|background|location|room|outdoor|indoor)\b",
        "lighting": r"\b(?:change|replace|different|new)\b.{0,30}\b(?:light|lighting|shadow|sunset|night)\b",
    }
    for topic, pattern in locked_topics.items():
        if re.search(pattern, lowered):
            warnings.append(f"Slide {index}: pose direction may contradict locked {topic}.")

    camera_change = re.search(
        r"\b(?:selfie|close[- ]?up|wide shot|from behind|rear view|profile|overhead|low angle|high angle|"
        r"camera|reframe|zoom|first-person|pov)\b",
        lowered,
    )
    if camera_policy == "LOCK" and camera_change:
        warnings.append(
            f"Slide {index}: camera policy LOCK conflicts with a viewpoint/reframing request; the pose will be "
            "adapted to the original frame."
        )
    return warnings


def build_directed_carousel(
    pose_list,
    slide_count=4,
    output_mode=OUTPUT_ALL,
    slide_selector=1,
    camera_policy="AUTO",
    continuity_strength="Strong",
    base_prompt="",
    negative_prompt="",
    has_master=True,
):
    poses = _pose_lines(pose_list)
    requested_count = max(1, min(MAX_SLIDES, int(slide_count)))
    camera_policy = camera_policy if camera_policy in CAMERA_POLICIES else "AUTO"
    continuity_strength = (
        continuity_strength if continuity_strength in CONTINUITY_LEVELS else "Strong"
    )
    warnings = []
    errors = []
    if not has_master:
        warnings.append("Master image is not connected; ReferenceLatent must still receive the canonical image.")
    if not poses:
        errors.append("Pose list is empty.")
    if len(poses) < requested_count:
        warnings.append(
            f"slide_count requests {requested_count} slides but only {len(poses)} non-empty pose lines exist."
        )
    active_count = min(requested_count, len(poses)) if poses else 0
    active_poses = poses[:active_count]
    selector = max(1, min(max(1, active_count), int(slide_selector)))

    master_parts = [
        "Create exactly one standalone photorealistic image from the canonical reference image.",
        "This is an image-to-image reconstruction, not a face swap and not a collage.",
        _continuity_contract(continuity_strength),
        _camera_contract(camera_policy),
        (
            "Maintain natural anatomy and clothing physics. Do not preserve the original arm, hand or leg "
            "positions when they conflict with the requested pose."
        ),
    ]
    base_prompt = _clean(base_prompt)
    if base_prompt:
        master_parts.append("Additional direction, subordinate to the continuity contract: " + base_prompt)
    master_prompt = "\n".join(master_parts)

    all_prompts = []
    slide_records = []
    for index, pose in enumerate(active_poses, start=1):
        warnings.extend(_warnings_for_pose(pose, camera_policy, index))
        prompt = (
            master_prompt
            + f"\n\nSLIDE {index} CHANGE REQUEST:\n"
            + pose
            + "\nGenerate only this frame and only one person."
        )
        all_prompts.append(prompt)
        slide_records.append({"index": index, "pose": pose, "allowed_changes": ["pose", "camera_if_policy_allows"]})

    selected_prompt = all_prompts[selector - 1] if all_prompts else ""
    if output_mode == OUTPUT_SELECTED and selected_prompt:
        prompts = [selected_prompt]
        emitted_indices = [selector]
    else:
        prompts = all_prompts
        emitted_indices = list(range(1, active_count + 1))

    supplied_negative = _clean(negative_prompt, 8_000)
    combined_negative = ", ".join(part for part in (supplied_negative, NEGATIVE_BASE) if part)
    negatives = [combined_negative for _ in prompts]
    manifest = {
        "schema": 1,
        "target": "FLUX.2 Klein 9B",
        "topology": "one_master_reference_plus_text_pose_list",
        "output_mode": output_mode,
        "camera_policy": camera_policy,
        "continuity_strength": continuity_strength,
        "requested_slide_count": requested_count,
        "active_slide_count": active_count,
        "selected_slide": selector,
        "emitted_slides": emitted_indices,
        "locked_attributes": [
            "identity", "facial_anatomy", "body_proportions", "hairstyle", "outfit",
            "accessories", "location", "background", "lighting", "color_palette", "photographic_style",
        ],
        "slides": slide_records,
    }
    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "allowed_to_change": ["pose", "camera/framing according to camera_policy"],
        "active_slide_count": active_count,
    }
    if errors:
        raise ValueError("[Onyx Text Carousel] " + " ".join(errors))
    return {
        "prompts": prompts,
        "negative_prompts": negatives,
        "selected_prompt": selected_prompt,
        "master_prompt": master_prompt,
        "manifest": json.dumps(manifest, ensure_ascii=False, indent=2),
        "validation_report": json.dumps(report, ensure_ascii=False, indent=2),
        "slide_count": active_count,
        "selected_index": selector,
        "status": f"{len(prompts)} prompt(s) emitted · {camera_policy} camera · {len(warnings)} warning(s)",
    }

class OnyxTextCarouselDirector:
    """Text pose-list carousel director for FLUX.2 Klein (V1 port of NodoForgeLab2 Beta)."""

    SEARCH_ALIASES = ["text carousel", "pose list director", "flux2 pose carousel"]
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Compiles one pose instruction per line into a FLUX.2 Klein prompt list while preserving the "
        "canonical image's identity, clothing and photographic session."
    )
    CATEGORY = "Onyx/Krea2 Carousel"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "prompts", "negative_prompts", "selected_prompt", "master_prompt",
        "manifest", "validation_report", "slide_count", "selected_index",
    )
    OUTPUT_IS_LIST = (True, True, False, False, False, False, False, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_list": ("STRING", {"default": DEFAULT_POSES, "multiline": True, "dynamicPrompts": False}),
                "slide_count": ("INT", {"default": 4, "min": 1, "max": MAX_SLIDES, "step": 1}),
                "output_mode": ([OUTPUT_ALL, OUTPUT_SELECTED],),
                "slide_selector": ("INT", {"default": 1, "min": 1, "max": MAX_SLIDES, "step": 1}),
                "camera_policy": (list(CAMERA_POLICIES),),
                "continuity_strength": (list(CONTINUITY_LEVELS),),
            },
            "optional": {
                "master_image": ("IMAGE", {"display_name": "MASTER IMAGE · CANONICAL"}),
                "base_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
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
        base_prompt="",
        negative_prompt="",
        master_image=None,
    ):
        return json.dumps(
            {
                "pose_list": _clean(pose_list),
                "slide_count": slide_count,
                "output_mode": output_mode,
                "slide_selector": slide_selector,
                "camera_policy": camera_policy,
                "continuity_strength": continuity_strength,
                "base_prompt": _clean(base_prompt),
                "negative_prompt": _clean(negative_prompt),
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
        base_prompt="",
        negative_prompt="",
        master_image=None,
    ):
        result = build_directed_carousel(
            pose_list=pose_list,
            slide_count=slide_count,
            output_mode=output_mode,
            slide_selector=slide_selector,
            camera_policy=camera_policy,
            continuity_strength=continuity_strength,
            base_prompt=base_prompt,
            negative_prompt=negative_prompt,
            has_master=master_image is not None,
        )
        return {
            "ui": {
                "status": [result["status"]],
                "selected_prompt": [result["selected_prompt"]],
                "validation_report": [result["validation_report"]],
            },
            "result": (
                result["prompts"],
                result["negative_prompts"],
                result["selected_prompt"],
                result["master_prompt"],
                result["manifest"],
                result["validation_report"],
                result["slide_count"],
                result["selected_index"],
            ),
        }
