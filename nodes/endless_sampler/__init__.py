"""Onyx Endless Sampler — vendored from ComfyUI-HR-Endless-Sampler v0.9.0
(https://github.com/hradec/ComfyUI-HR-Endless-Sampler), Apache License 2.0,
see LICENSE in this folder.

Changes made for Onyx (Apache 2.0 §4b notice):
  - node ids, display names, routes, events and custom types renamed
    (HREndless* -> OnyxEndless*, /hr_endless_sampler_* -> /onyx_endless_sampler_*)
    so this copy can live next to the original pack without clashing;
  - category moved to Onyx/Video, Onyx session check added to the sampler and
    the save node;
  - OnyxEndlessSampler gained ref_image_1..5: one socket per Ref2VA reference
    image, so images keep their own size/aspect and empty slots are skipped.
Requires llama-cpp-python (CUDA build) for the Gemma 4 prompt director; the
Gemma 4 12B QAT files are downloaded on first use.
"""

from .nodes import OnyxEndlessSampler
from .seamless import OnyxH3SeamlessSampler
from .preview import OnyxEndlessSamplerPreview
from .video_io import OnyxEndlessSamplerLoadVideo, OnyxEndlessSamplerSaveVideo

NODE_CLASS_MAPPINGS = {
    "OnyxEndlessSampler": OnyxEndlessSampler,
    "OnyxH3SeamlessSampler": OnyxH3SeamlessSampler,
    "OnyxEndlessSamplerPreview": OnyxEndlessSamplerPreview,
    "OnyxEndlessSamplerSaveVideo": OnyxEndlessSamplerSaveVideo,
    "OnyxEndlessSamplerLoadVideo": OnyxEndlessSamplerLoadVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxEndlessSampler": "Onyx Endless Sampler",
    "OnyxH3SeamlessSampler": "Onyx H3 Seamless Sampler",
    "OnyxEndlessSamplerPreview": "Onyx Endless Sampler Preview",
    "OnyxEndlessSamplerSaveVideo": "Onyx Endless Sampler Save Video",
    "OnyxEndlessSamplerLoadVideo": "Onyx Endless Sampler Load Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
