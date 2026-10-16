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

from ..onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('krea2_bundle', 'krea2_carousel', __package__)
    return ns




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


def _camera_contract(*args, **kwargs):
    return _onyx_core()['_camera_contract'](*args, **kwargs)



def _continuity_contract(*args, **kwargs):
    return _onyx_core()['_continuity_contract'](*args, **kwargs)



def _contradiction_warnings(*args, **kwargs):
    return _onyx_core()['_contradiction_warnings'](*args, **kwargs)



def build_krea2_carousel(*args, **kwargs):
    return _onyx_core()['build_krea2_carousel'](*args, **kwargs)


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
