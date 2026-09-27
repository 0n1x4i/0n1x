"""Pose-list bridge between the master image and the Krea2 Director.

Two ways to run:

- One-step (default): the pose list is generated INSIDE the execution, from the
  master image, every time the workflow is queued. A single Queue press does
  analysis + rendering, and every run gets a fresh set of poses.
- Two-step: the list is prepared and reviewed in the frontend first; this node
  then only releases that reviewed list at execution time. Reference Copy
  always uses this path, because it analyses several uploaded references from
  the Batch Loader and must not advance it during rendering.
"""

from __future__ import annotations

import json
import random

import numpy as np
from PIL import Image

from .krea2_carousel import MAX_SLIDES, _pose_lines

ONE_STEP = "One-step"


def _read_config(auto_config):
    try:
        config = json.loads(auto_config or "{}")
    except (TypeError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _is_one_step(config):
    return config.get("run_mode") == ONE_STEP


def _tensor_to_pil(image):
    if image is None:
        return None
    frame = image[0] if getattr(image, "ndim", 0) == 4 else image
    array = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array[..., :3]).convert("RGB")


class OnyxKrea2PoseListPrep:
    """Pose-list bridge for the Krea2 carousel (V1 port of NodoForgeLab2 Beta)."""

    EXPERIMENTAL = True
    DESCRIPTION = ("Generate (one-step, every run) or release a reviewed (two-step) pose list "
                   "for a fixed Krea2 master.")
    CATEGORY = "Onyx/Krea2 Carousel"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("pose_list", "slide_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepared_pose_list": ("STRING", {"default": "", "multiline": True}),
                "prepared_count": ("INT", {"default": 4, "min": 1, "max": MAX_SLIDES}),
            },
            "optional": {
                "pose_reference": ("IMAGE", {"display_name": "POSE REFERENCES ONLY · NOT MASTER", "lazy": True}),
                "master_image": ("IMAGE", {"display_name": "MASTER · AUTO POSE ANALYSIS", "lazy": True}),
                # Written by the node UI; holds the generator settings (never an API key).
                "auto_config": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def check_lazy_status(self, auto_config="", master_image=None, **kwargs):
        # Two-step renders consume only the prepared text and must not advance
        # the reference Batch Loader. One-step Master Auto needs the master.
        config = _read_config(auto_config)
        if _is_one_step(config) and config.get("mode") == "Master Auto" and master_image is None:
            return ["master_image"]
        return []

    @classmethod
    def IS_CHANGED(cls, prepared_pose_list, prepared_count, pose_reference=None,
                   master_image=None, auto_config=""):
        if _is_one_step(_read_config(auto_config)):
            return float("nan")  # always re-run: fresh poses on every execution
        return json.dumps({"poses": prepared_pose_list, "count": prepared_count}, ensure_ascii=False)

    def _generate(self, config, master_image):
        from .pose_generator import generate_pose_list

        mode = config.get("mode", "Master Auto")
        if mode == "Reference Copy":
            raise ValueError(
                "[Onyx Pose List Prep] Reference Copy is two-step only: switch Run mode to "
                "Two-step, press Run pose analysis, then Queue."
            )
        image = _tensor_to_pil(master_image) if mode == "Master Auto" else None
        if mode == "Master Auto" and image is None:
            raise ValueError("[Onyx Pose List Prep] Master Auto needs master_image connected.")
        request = dict(config)
        request["seed"] = random.randint(0, 2**31 - 1)
        request.pop("api_key", None)  # keys come from GEMINI_API_KEY / XAI_API_KEY in one-step
        print(f"[Onyx Pose List Prep] One-step: generating {request.get('count', 4)} poses "
              f"({mode}, {request.get('backend')})…")
        result = generate_pose_list(request, image)
        return result["poses"]

    def execute(self, prepared_pose_list, prepared_count, pose_reference=None,
                master_image=None, auto_config=""):
        config = _read_config(auto_config)
        if _is_one_step(config):
            poses = _pose_lines("\n".join(self._generate(config, master_image)))
            status = f"{len(poses)} poses generated this run · {config.get('backend')}"
        else:
            poses = _pose_lines(prepared_pose_list)
            if not poses:
                raise ValueError("[Onyx Pose List Prep] Prepare the pose list before running the carousel.")
            if len(poses) != int(prepared_count):
                raise ValueError(
                    f"[Onyx Pose List Prep] The pose list has {len(poses)} entries, "
                    f"but the counter says {prepared_count}."
                )
            status = f"{len(poses)} poses prepared · fixed master"
        if not poses:
            raise ValueError("[Onyx Pose List Prep] No pose was generated.")
        return {
            "ui": {"status": [status], "pose_list": ["\n".join(poses)]},
            "result": ("\n".join(poses), len(poses)),
        }


__all__ = ["OnyxKrea2PoseListPrep"]
