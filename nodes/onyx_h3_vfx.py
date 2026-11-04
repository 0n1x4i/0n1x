# -*- coding: utf-8 -*-
"""
MiniMax H3 VFX modes — shared rules + the "Onyx H3 VFX LoRA" node.

H3 can EDIT an existing video instead of generating from references, with one
of the community VFX LoRAs. Each mode has three rules:

  mode                          LoRA file                                          trigger          reference image
  vfx_edit                      minimax_h3_vfx_edit_v1.0_r128.safetensors           "vfx_edit:"      none (ignored)
  vfx_edit (edited first frame) minimax_h3_vfx_edit_v1.0_r128_ffp.safetensors       "vfx_edit:"      REQUIRED: your retouched first frame
  style_transfer                minimax_h3_style_transfer_v1.0_r64.safetensors      "style_transfer:" optional: style reference

In every mode the SOURCE VIDEO goes into the graph as a guide
(MiniMaxH3AddGuide, frame_idx 0 — core ComfyUI v0.34.0+), not as a reference
video. Consequences for the prompt, applied by postprocess_result():
  * the prompt starts with the trigger word;
  * <Video 1> (the source) is named "the source video", and any later
    <Video n> is renumbered n-1, because the source no longer occupies a
    reference slot; same for <Audio 1> -> "the source soundtrack" when the
    source audio is guided too.

"Edited first frame": retouch frame 0 of the source (any image editor or a
Nano Banana edit), connect it as the reference image, and H3 propagates the
change through the whole clip.

LoRAs: https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI (folder loras/),
into ComfyUI/models/loras/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .onyx_render_profile import ensure_profile_ready

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('h3_tools_bundle', 'onyx_h3_vfx', __package__)
    return ns


VFX_NONE = "none"
VFX_EDIT = "vfx_edit"
VFX_EDIT_FRAME = "vfx_edit (edited first frame)"
VFX_STYLE = "style_transfer"
VFX_MODES = [VFX_NONE, VFX_EDIT, VFX_EDIT_FRAME, VFX_STYLE]

_HF = "https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/"

VFX_TOOLTIP = (
    "MiniMax H3 only. Edit the connected source VIDEO instead of generating from references. "
    "Needs the matching LoRA loaded (Onyx H3 VFX LoRA node) and the source video wired into "
    "MiniMaxH3AddGuide (ComfyUI v0.34+).\n"
    "• vfx_edit — change something in the clip (\"make the jacket red\"). Reference images are "
    "not used and are not analysed.\n"
    "• vfx_edit (edited first frame) — retouch frame 0 yourself, connect it as image_1: H3 "
    "propagates the retouch through the whole clip.\n"
    "• style_transfer — restyle the clip; image_1 = optional style reference.\n"
    "The prompt gets the trigger word, and <Video 1> becomes \"the source video\"."
)


@dataclass(frozen=True)
class VfxMode:
    mode: str
    lora_file: str
    trigger: str
    reference_rule: str          # none | optional | required

    @property
    def url(self) -> str:
        return _HF + self.lora_file


_MODES = {
    VFX_EDIT: VfxMode(VFX_EDIT, "minimax_h3_vfx_edit_v1.0_r128.safetensors", "vfx_edit:", "none"),
    VFX_EDIT_FRAME: VfxMode(VFX_EDIT_FRAME, "minimax_h3_vfx_edit_v1.0_r128_ffp.safetensors",
                            "vfx_edit:", "required"),
    VFX_STYLE: VfxMode(VFX_STYLE, "minimax_h3_style_transfer_v1.0_r64.safetensors",
                       "style_transfer:", "optional"),
}

STRENGTH_MIN, STRENGTH_MAX, STRENGTH_DEFAULT = -2.0, 2.0, 1.0


def resolve(*args, **kwargs):
    return _onyx_core()['resolve'](*args, **kwargs)



def check_inputs(*args, **kwargs):
    return _onyx_core()['check_inputs'](*args, **kwargs)



# ─────────────────────────────────────────────────────────────────────────────
# Prompt rules
# ─────────────────────────────────────────────────────────────────────────────

def _name_first_tag(*args, **kwargs):
    return _onyx_core()['_name_first_tag'](*args, **kwargs)



def apply_prompt_rules(*args, **kwargs):
    return _onyx_core()['apply_prompt_rules'](*args, **kwargs)



def slot_warnings(*args, **kwargs):
    return _onyx_core()['slot_warnings'](*args, **kwargs)



def postprocess_result(*args, **kwargs):
    return _onyx_core()['postprocess_result'](*args, **kwargs)



# ─────────────────────────────────────────────────────────────────────────────
# Node: load the LoRA that matches the mode
# ─────────────────────────────────────────────────────────────────────────────

class OnyxH3VfxLora:
    """Load the MiniMax H3 VFX LoRA that matches a vfx_mode, and output its trigger word."""

    DESCRIPTION = (
        "Loads the MiniMax H3 VFX LoRA for the chosen mode (vfx_edit, edited first frame, "
        "style_transfer) onto the H3 model and outputs its trigger word. Use the SAME mode as "
        "on H3 Context-IR. Wiring: source video frames -> MiniMaxH3AddGuide (frame_idx 0) "
        "between the conditioning and the sampler; the edited first frame or the style image "
        "-> the reference image slot. Needs ComfyUI v0.34+."
    )

    _loaded: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vfx_mode": (VFX_MODES, {"default": VFX_EDIT, "tooltip": VFX_TOOLTIP}),
                "strength": ("FLOAT", {
                    "default": STRENGTH_DEFAULT, "min": STRENGTH_MIN, "max": STRENGTH_MAX, "step": 0.05,
                    "tooltip": "LoRA strength. 1.0 = as trained. Lower it if the edit spills "
                               "onto parts of the clip that should not change."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "trigger")
    FUNCTION = "apply"
    CATEGORY = "Onyx/H3"

    def apply(self, *args, **kwargs):
        return _onyx_core()['OnyxH3VfxLora'].apply(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {"OnyxH3VfxLora": OnyxH3VfxLora}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxH3VfxLora": "Onyx H3 VFX LoRA"}
