"""Onyx Krea2 Carousel — V1 port of the NodoForgeLab2 Beta pack (v2.3).

The original shipped as a V3 (comfy_api.latest / comfy_entrypoint) extension.
Onyx registers everything through NODE_CLASS_MAPPINGS, and ComfyUI only reads
ONE of the two mechanisms per pack, so the five nodes were rewritten as classic
INPUT_TYPES / RETURN_TYPES classes. Business logic is unchanged.
"""

from .carousel import OnyxPoseCarouselDirector
from .director import OnyxVisualPromptDirector
from .krea2_carousel import OnyxKrea2CarouselDirector
from .pose_prep import OnyxKrea2PoseListPrep
from .simple_carousel import OnyxTextCarouselDirector
from . import routes as _routes  # noqa: F401 - registers the /onyx/nf/* endpoints

NODE_CLASS_MAPPINGS = {
    "OnyxVisualPromptDirector": OnyxVisualPromptDirector,
    "OnyxPoseCarouselDirector": OnyxPoseCarouselDirector,
    "OnyxTextCarouselDirector": OnyxTextCarouselDirector,
    "OnyxKrea2CarouselDirector": OnyxKrea2CarouselDirector,
    "OnyxKrea2PoseListPrep": OnyxKrea2PoseListPrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxVisualPromptDirector": "Onyx Visual Prompt Director",
    "OnyxPoseCarouselDirector": "Onyx Pose Carousel Director",
    "OnyxTextCarouselDirector": "Onyx Text Carousel Director",
    "OnyxKrea2CarouselDirector": "Onyx Krea2 Edit Carousel Director",
    "OnyxKrea2PoseListPrep": "Onyx Krea2 Pose List Prep",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
