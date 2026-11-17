# -*- coding: utf-8 -*-
# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
# Copying, redistribution, resale, extraction of the protected code or prompts, and
# circumvention of the licence verification are prohibited.
"""Onyx Fast Video Grade."""

import os
import queue
import shutil
import subprocess
import threading

import numpy as np
import torch
import torch.nn.functional as F

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'onyx_fast_grade', __package__)
    return ns


_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LUT_DIR = os.path.join(_HERE, "luts")
_CHUNK = 12
_ENCODERS = ["auto (GPU when available)", "GPU (NVENC)", "CPU (x264)"]
_LUMA = (0.2126, 0.7152, 0.0722)


# ── tools ────────────────────────────────────────────────────────────────────

def _ffmpeg(*args, **kwargs):
    return _onyx_core()['_ffmpeg'](*args, **kwargs)



_NVENC = {}


def _has_nvenc(*args, **kwargs):
    return _onyx_core()['_has_nvenc'](*args, **kwargs)



def _lut_files():
    try:
        return sorted(f for f in os.listdir(_LUT_DIR) if f.lower().endswith(".cube"))
    except OSError:
        return []


def _load_cube(*args, **kwargs):
    return _onyx_core()['_load_cube'](*args, **kwargs)



def _probe(*args, **kwargs):
    return _onyx_core()['_probe'](*args, **kwargs)



def _resolve_source(*args, **kwargs):
    return _onyx_core()['_resolve_source'](*args, **kwargs)



# ── the grade: one pass, any number of frames ───────────────────────────────

def _smooth(*args, **kwargs):
    return _onyx_core()['_smooth'](*args, **kwargs)



def grade(*args, **kwargs):
    return _onyx_core()['grade'](*args, **kwargs)



# ── the node ─────────────────────────────────────────────────────────────────

def _slider(lo=-100.0, hi=100.0, step=1.0, default=0.0):
    return ("FLOAT", {"default": default, "min": lo, "max": hi, "step": step, "display": "slider"})


class OnyxFastVideoGrade:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exposure": _slider(-3.0, 3.0, 0.05),
                "contrast": _slider(),
                "highlights": _slider(),
                "shadows": _slider(),
                "whites": _slider(),
                "blacks": _slider(),
                "temperature": _slider(),
                "tint": _slider(),
                "saturation": _slider(),
                "vibrance": _slider(),
                "lut": (["none"] + _lut_files(), {"default": "none"}),
                "lut_strength": _slider(0.0, 1.0, 0.05, 1.0),
                "grain_amount": _slider(0.0, 100.0, 1.0, 0.0),
                "grain_size": _slider(0.5, 4.0, 0.1, 1.2),
                "grain_color": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "encoder": (_ENCODERS, {"default": _ENCODERS[0]}),
                "quality": ("INT", {"default": 18, "min": 10, "max": 35,
                                    "tooltip": "Lower = better quality, bigger file (CRF / CQ)."}),
                "filename_prefix": ("STRING", {"default": "onyx_grade/video"}),
            },
            "optional": {
                "video": ("VIDEO",),
                "video_info": ("VHS_VIDEOINFO",),
                "video_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "file_path")
    FUNCTION = "run"
    CATEGORY = "Onyx/Post-Processing"
    OUTPUT_NODE = True
    DESCRIPTION = ("Colour grade (exposure, contrast, highlights, shadows, whites, blacks, white balance, "
                   "saturation, vibrance, .cube LUT) and film grain on a video file, streamed through the GPU "
                   "and a hardware encoder: fast on long videos, flat memory.")

    def run(self, *args, **kwargs):
        return _onyx_core()['OnyxFastVideoGrade'].run(self, *args, **kwargs)



def process_file(*args, **kwargs):
    return _onyx_core()['process_file'](*args, **kwargs)



NODE_CLASS_MAPPINGS = {"OnyxFastVideoGrade": OnyxFastVideoGrade}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxFastVideoGrade": "Onyx Fast Video Grade (GPU, streamed)"}
