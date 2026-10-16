# -*- coding: utf-8 -*-
"""Text-driven carousel director for one canonical image and N pose prompts."""

from __future__ import annotations

import json
import re

from ..onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('krea2_bundle', 'simple_carousel', __package__)
    return ns




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


def _camera_contract(*args, **kwargs):
    return _onyx_core()['_camera_contract'](*args, **kwargs)



def _continuity_contract(*args, **kwargs):
    return _onyx_core()['_continuity_contract'](*args, **kwargs)



def _warnings_for_pose(*args, **kwargs):
    return _onyx_core()['_warnings_for_pose'](*args, **kwargs)



def build_directed_carousel(*args, **kwargs):
    return _onyx_core()['build_directed_carousel'](*args, **kwargs)


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
