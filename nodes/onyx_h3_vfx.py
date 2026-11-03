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


def resolve(mode: str) -> VfxMode | None:
    return _MODES.get(mode)


def check_inputs(vfx: VfxMode, *, has_source_video: bool, image_count: int, node: str) -> None:
    """Refuse a combination H3 cannot render, before any API call is paid for."""
    if not has_source_video:
        raise RuntimeError(f"{node}: vfx_mode '{vfx.mode}' edits a video — connect the source video.")
    if vfx.reference_rule == "required" and image_count == 0:
        raise RuntimeError(f"{node}: '{vfx.mode}' needs your retouched first frame as image_1.")
    if vfx.reference_rule == "none" and image_count:
        print(f"ℹ️  [{node}] vfx_edit uses no reference image: the {image_count} connected image(s) "
              f"are ignored (not analysed, not referenced in the prompt).")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt rules
# ─────────────────────────────────────────────────────────────────────────────

def _name_first_tag(prompt: str, kind: str, phrase: str) -> str:
    """<Kind 1> -> phrase, <Kind n> -> <Kind n-1> (the source left the reference slots)."""
    def repl(m):
        n = int(m.group(1))
        return phrase if n == 1 else f"<{kind} {n - 1}>"
    return re.sub(rf"<{kind} (\d+)>", repl, prompt)


def apply_prompt_rules(prompt: str, vfx: VfxMode, *, source_audio_guided: bool) -> str:
    out = _name_first_tag(prompt, "Video", "the source video")
    if source_audio_guided:
        out = _name_first_tag(out, "Audio", "the source soundtrack")
    out = out.strip()
    if not out.lower().startswith(vfx.trigger.lower()):
        out = f"{vfx.trigger} {out}"
    return out


def slot_warnings(prompt: str, *, image_count: int | None, has_audio_input: bool) -> list:
    """Tags the prompt uses but nothing feeds. Warnings only: the H3 node may be
    wired differently from this Context-IR node, so nothing is rewritten."""
    warns = []
    if image_count is not None:
        pics = sorted({int(n) for n in re.findall(r"<Picture (\d+)>", prompt)})
        extra = [n for n in pics if n > image_count]
        if extra:
            warns.append(f"prompt references <Picture {extra[0]}> but only {image_count} image(s) "
                         f"were given — check that slot on the H3 node.")
    if not has_audio_input and re.search(r"<Audio \d+>", prompt):
        warns.append("prompt references <Audio n> but no audio was connected here — if the H3 node "
                     "has no audio reference either, replace the tag with \"the soundtrack\".")
    return warns


def postprocess_result(result, vfx: VfxMode | None, *, source_audio_guided: bool,
                       has_audio_input: bool, node: str):
    """Apply the VFX prompt rules to output 0 (the H3 prompt) of a node result,
    whatever its shape (tuple, list, or {"ui": ..., "result": (...)})."""
    wrapper = None
    values = result
    if isinstance(result, dict) and "result" in result:
        wrapper, values = result, result["result"]
    if not isinstance(values, (tuple, list)) or not values or not isinstance(values[0], str):
        return result
    prompt = values[0]
    if vfx is not None:
        prompt = apply_prompt_rules(prompt, vfx, source_audio_guided=source_audio_guided)
    for w in slot_warnings(prompt, image_count=None, has_audio_input=has_audio_input):
        print(f"⚠️  [{node}] {w}")
    values = (prompt, *values[1:])
    if wrapper is not None:
        wrapper = dict(wrapper)
        wrapper["result"] = values
        return wrapper
    return values


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

    def apply(self, model, vfx_mode, strength):
        ensure_profile_ready()
        vfx = resolve(vfx_mode)
        if vfx is None or strength == 0:
            return (model, "")
        import folder_paths
        path = folder_paths.get_full_path("loras", vfx.lora_file)
        if not path:
            raise RuntimeError(
                f"Onyx H3 VFX LoRA: {vfx.lora_file} not found in models/loras/.\n"
                f"Download: {vfx.url}")
        import comfy.sd
        import comfy.utils
        lora = self._loaded.get(path)
        if lora is None:
            lora = comfy.utils.load_torch_file(path, safe_load=True)
            self._loaded.clear()            # keep one LoRA in RAM, like the core loader
            self._loaded[path] = lora
        patched, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return (patched, vfx.trigger)


NODE_CLASS_MAPPINGS = {"OnyxH3VfxLora": OnyxH3VfxLora}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxH3VfxLora": "Onyx H3 VFX LoRA"}
