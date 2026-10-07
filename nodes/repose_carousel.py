"""
Onyx Re-pose for Carousel
Generates N pose variations of a single input image, with presets controlling
the degree of freedom -- from a barely-changed re-pose to a whole new location.

The preset directives and the generation/retry loop live server-side — see
nodes/onyx_remote_exec.py. This file keeps the ComfyUI scaffolding (INPUT_TYPES,
preset/model lists) local.
"""

import torch

from .onyx_render_profile import ensure_profile_ready
from .onyx_remote_exec import load_remote

# ─────────────────────────────────────────────────────────────────────────────
REPOSE_PRESETS = [
    "Subtle",
    "Normal",
    "Heavy",
]

# Image models available for this node (image-only, no video)
_IMAGE_MODELS = ["Nano Banana Pro", "Nano Banana 2.1", "Nano Banana 2", "Seedream 5 Pro", "GPT Image 2.0"]


# ─────────────────────────────────────────────────────────────────────────────
# Node class
# ─────────────────────────────────────────────────────────────────────────────
class OnyxReposeCarouselNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1":  ("IMAGE", {"tooltip": "Identity reference (single image -- face/person to preserve)."}),
                "provider": (["GOOGLE", "WAVESPEED", "KIE", "FAL", "VERTEX"], {"default": "GOOGLE"}),
                "model":    (_IMAGE_MODELS, {"default": "Nano Banana Pro"}),
                "preset":   (REPOSE_PRESETS, {
                    "default": "Subtle",
                    "tooltip": "Controls how much the model is allowed to deviate from the original.",
                }),
                "prompt":   ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "Describe the subject, style, or any specific detail.",
                }),
                "count":    ("INT", {
                    "default": 4, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Number of re-posed images to generate per scene image (image_2 frame).",
                }),
                "retry_count": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of retries per image on failure.",
                }),
                "image_size": (["1K", "2K", "4K", "8K"], {
                    "default": "2K",
                    "tooltip": "Output resolution. 8K available with Nano Banana Pro via WaveSpeed only.\n"
                               "GPT Image 2.0 tops out at 4K — 8K is snapped down automatically.",
                }),
            },
            "optional": {
                "image_2":  ("IMAGE", {"tooltip": "Scene/environment reference. Can be a batch -- count variations will be generated per frame."}),
                "is_selfie": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Adds selfie framing instructions to the prompt. Ignored when image_2 is a batch.",
                }),
                "is_mirror_selfie": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Adds mirror selfie instructions to the prompt. Ignored when image_2 is a batch.",
                }),
                "gemini_api_key":    ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "[GOOGLE] Google AI Studio API key."}),
                "wavespeed_api_key": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "[WAVESPEED] WaveSpeed API key."}),
                "kie_api_key":       ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "[KIE] Kie.ai API key."}),
                "fal_api_key":       ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "[FAL] Fal.ai API key."}),
                "vertex_json_folder": ("STRING", {"default": "", "multiline": False,
                                       "tooltip": "[VERTEX] Path to service account JSON folder."}),
                "disable_safety_threshold": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable the safety filter (on by default).",
                }),
                "fal_safety_tolerance": (["1", "2", "3", "4", "5", "6"], {
                    "default": "6",
                    "tooltip": "[FAL] 1 = strict | 6 = permissive (default)",
                }),
                "gpt2_image_quality": (["high", "medium", "low"], {
                    "default": "high",
                    "tooltip": "[GPT Image 2.0] Generation quality. Ignored by every other model.\n"
                               "Lower tiers are markedly cheaper — worth using while you dial in a "
                               "preset, since this node fires count x scenes requests per run.",
                }),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                     "9:16", "16:9", "21:9"],
                    {"default": "auto",
                     "tooltip": "auto = detected from image_2 (or image_1 if no image_2)."},
                ),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "[GOOGLE / VERTEX] Model creativity.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("images",)
    OUTPUT_NODE   = False
    FUNCTION      = "run"
    CATEGORY      = "Onyx/Automation"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(
        self,
        image_1,
        image_2                  = None,
        provider                 = "GOOGLE",
        model                    = "Nano Banana Pro",
        preset                   = "Subtle",
        prompt                   = "",
        count                    = 4,
        retry_count              = 3,
        image_size               = "2K",
        is_selfie                = False,
        is_mirror_selfie         = False,
        gemini_api_key           = "",
        wavespeed_api_key        = "",
        kie_api_key              = "",
        fal_api_key              = "",
        vertex_json_folder       = "",
        disable_safety_threshold = True,
        fal_safety_tolerance     = "6",
        aspect_ratio             = "auto",
        temperature              = 1.0,
        gpt2_image_quality       = "high",
    ):
        ensure_profile_ready()
        from .nano_banana_aio import OnyxNanoBananaAIO

        ns = load_remote("repose_carousel_core")
        return ns["run_impl"](
            image_1, image_2, provider, model, preset, prompt, count, retry_count,
            image_size, is_selfie, is_mirror_selfie, gemini_api_key, wavespeed_api_key,
            kie_api_key, fal_api_key, vertex_json_folder, disable_safety_threshold,
            fal_safety_tolerance, aspect_ratio, temperature, gpt2_image_quality,
            OnyxNanoBananaAIO=OnyxNanoBananaAIO,
        )
