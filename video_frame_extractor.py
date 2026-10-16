"""
ComfyUI node - Onyx Video Frame Extractor.

Outputs one frame from a video: either a file picked in ComfyUI's input folder,
or a frame sequence arriving on `video_frames` — which is what lets a batch of
videos drive it without choosing a file by hand for each one.

Deliberately plain. An earlier version carried a JS extension with an on-node
gallery, an Extract button and an animated theme; it wrote preview PNGs to temp
on every run, forced its own size, and grew a widget row at each restart. None of
that helped pick a frame index, so it is gone. What remains is a node that reads
a video and returns a frame.
"""

import os
import logging
import random

import av
import numpy as np
import torch
from PIL import Image
import folder_paths

from .nodes.onyx_render_profile import ensure_profile_ready

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('utils_bundle', 'video_frame_extractor', __package__)
    return ns

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def _list_video_files():
    """List video files in ComfyUI's input folder."""
    input_dir = folder_paths.get_input_directory()
    files = []
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in _VIDEO_EXTENSIONS and os.path.isfile(os.path.join(input_dir, f)):
            files.append(f)
    files.sort(key=str.lower)
    if not files:
        files = ["(no videos found)"]
    return files


def _decode_selection(*args, **kwargs):
    return _onyx_core()['_decode_selection'](*args, **kwargs)



class OnyxVideoFrameExtractorNode:
    """Extract one frame from a video file or from a connected frame sequence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_list_video_files(), {"video_upload": True}),
                "start_second": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1}),
                "frame_count": ("INT", {"default": 10, "min": 1, "max": 9999, "step": 1,
                                        "tooltip": "How many frames are sampled, and therefore how "
                                                   "far selected_frame can reach."}),
                "frame_interval": ("INT", {"default": 1, "min": 1, "max": 300, "step": 1,
                                           "tooltip": "Take one frame every N. 1 samples every frame."}),
                "selected_frame": ("INT", {"default": 0, "min": 0, "max": 9998, "step": 1,
                                           "tooltip": "Which sampled frame to output. 0 with "
                                                      "start_second=0 gives the very first frame."}),
            },
            "optional": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Connect the `video` output of Onyx Image and Video Batch Loader.\n"
                               "When connected it REPLACES the file dropdown entirely — nothing is "
                               "read from disk."}),
                "video_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate of the connected frames, so start_second can be turned "
                               "into a frame index. Left at 0, start_second is ignored."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract"
    CATEGORY = "image"

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        # Le placeholder n'est pas une erreur : le node peut etre pilote par
        # video_frames, et VALIDATE_INPUTS ne sait pas si une entree est branchee.
        # Le controle reel est dans extract(), ou l'information existe.
        if video == "(no videos found)":
            return True
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True

    def extract(self, *args, **kwargs):
        return _onyx_core()['OnyxVideoFrameExtractorNode'].extract(self, *args, **kwargs)
