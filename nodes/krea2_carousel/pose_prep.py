"""Prepared pose-list bridge between reference images and Krea2 Director.

Vision analysis happens explicitly in the frontend before queueing. This node
only releases the reviewed list at execution time; it never calls an API or
changes the master image used by Krea2.
"""

from __future__ import annotations

import json


from .krea2_carousel import MAX_SLIDES, _pose_lines

class OnyxKrea2PoseListPrep:
    """Reviewed pose-list bridge for the Krea2 carousel (V1 port of NodoForgeLab2 Beta)."""

    EXPERIMENTAL = True
    DESCRIPTION = "Prepare and review pose-only instructions before rendering a fixed Krea2 master."
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
            },
        }

    def check_lazy_status(self, **kwargs):
        # These links identify browser preflight sources. Rendering consumes only
        # the prepared text; it must not advance the reference Batch Loader.
        return []

    @classmethod
    def IS_CHANGED(cls, prepared_pose_list, prepared_count, pose_reference=None, master_image=None):
        return json.dumps({"poses": prepared_pose_list, "count": prepared_count}, ensure_ascii=False)

    def execute(self, prepared_pose_list, prepared_count, pose_reference=None, master_image=None):
        poses = _pose_lines(prepared_pose_list)
        if not poses:
            raise ValueError("[Onyx Pose List Prep] Prepara la pose list antes de ejecutar el carrusel.")
        if len(poses) != int(prepared_count):
            raise ValueError(
                f"[Onyx Pose List Prep] La pose list tiene {len(poses)} entradas, "
                f"pero el contador indica {prepared_count}."
            )
        return {
            "ui": {"status": [f"{len(poses)} poses preparadas · master fijo"], "pose_list": ["\n".join(poses)]},
            "result": ("\n".join(poses), len(poses)),
        }


__all__ = ["OnyxKrea2PoseListPrep"]
