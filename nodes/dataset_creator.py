"""
Onyx Dataset Creator
Generates a face-swapped dataset from pre-made body preset samples.
Presets (Flat / Skinny / Normal / Curvy / Large) live under Resources/.
Model chain: Nano Banana Pro (5 retries) → Nano Banana 2 (5 retries).
Resume support via processed.json in the output folder.

The orchestration logic (prompt tuning per preset/image, retry loop, background
and pose checks, resume bookkeeping) lives server-side — see
nodes/onyx_remote_exec.py. This file keeps only what ComfyUI needs to register
the node with no network access: INPUT_TYPES, and the local state (resources
path, captions cache) plus the local imports (nano_banana_aio,
instagram_faceswap) that the protected code doesn't own yet and receives as
plain function/class arguments.
"""

import os
import torch

from ..utils.image_utils import pil_to_tensor
from .onyx_render_profile import ensure_profile_ready
from .onyx_remote_exec import load_remote


def _find_resources_dir():
    # Primary: standard location (nodes/dataset_creator.py -> .. -> Resources)
    primary = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Resources",
    )
    if os.path.isdir(primary):
        return primary

    # Fallback: scan ComfyUI custom_nodes for any onyx/ofm folder with Resources/
    try:
        import folder_paths
        custom_nodes_dir = os.path.join(folder_paths.base_path, "custom_nodes")
        for folder_name in os.listdir(custom_nodes_dir):
            name_lower = folder_name.lower()
            if "onyx" in name_lower or "ofm" in name_lower:
                candidate = os.path.join(custom_nodes_dir, folder_name, "Resources")
                if os.path.isdir(candidate):
                    print(f"[Dataset] Resources found via fallback: {candidate}")
                    return candidate
    except Exception:
        pass

    # Last resort: return primary anyway (error will surface at runtime)
    return primary


_RESOURCES_DIR = _find_resources_dir()

PRESET_NAMES = ["Flat", "Skinny", "Normal", "Curvy", "Large", "Athletic", "Test"]


class OnyxDatasetCreatorNode:
    """
    Onyx Dataset Creator
    Generates a face-swapped dataset using pre-made body preset samples.
    """

    # Cle = PRESET en majuscules. Passe par reference au module distant pour
    # que le cache de captions survive entre plusieurs runs, meme si le module
    # distant lui-meme est reexecute a chaque appel (aucun cache cote serveur,
    # par design — voir onyx_remote_exec.py).
    _CAPTIONS_CACHE = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_reference": ("IMAGE",),

                "preset": (PRESET_NAMES, {
                    "default": "Normal",
                    "tooltip": "Body preset. Matching folder must exist under Resources/.",
                }),
                "resolution": (["1K", "2K", "4K"], {
                    "default": "2K",
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Extra instructions appended to the face-swap prompt.",
                }),
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder where the dataset images will be saved.",
                }),
                "trigger_word": ("STRING", {
                    "default": "MyChar",
                    "tooltip": "Used for file naming: TriggerWord_001.jpg, TriggerWord_002.jpg…",
                }),
                "test_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When ON: process only the single test image from Resources/{preset}/_test/. "
                               "Use this to preview the preset before launching the full dataset.",
                }),

                # ── Provider ──────────────────────────────────────────────────
                "provider": (["WAVESPEED", "GOOGLE", "FAL", "KIE", "VERTEX"], {
                    "default": "WAVESPEED",
                }),
                "disable_safety": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable safety filters where supported.",
                }),

                # ── API keys ──────────────────────────────────────────────────
                "gemini_api_key":     ("STRING", {"default": ""}),
                "wavespeed_api_key":  ("STRING", {"default": ""}),
                "fal_api_key":        ("STRING", {"default": ""}),
                "kie_api_key":        ("STRING", {"default": ""}),
                "vertex_json_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder containing Vertex AI service-account JSON files. "
                               "1 JSON = 1 project = 1 parallel worker slot.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("last_image", "summary")
    OUTPUT_NODE   = False
    FUNCTION      = "run"
    CATEGORY      = "Onyx/NanoBanana"
    DESCRIPTION   = (
        "Onyx Dataset Creator\n"
        "Generates a face-swapped dataset from pre-made body preset samples.\n"
        "Chain: Nano Banana Pro (×5) → Nano Banana 2 (×5).\n"
        "Resume support: already-processed images are automatically skipped."
    )

    def run(
        self,
        face_reference,
        preset,
        resolution,
        custom_prompt,
        output_folder,
        trigger_word,
        test_mode,
        provider,
        disable_safety,
        gemini_api_key,
        wavespeed_api_key,
        fal_api_key,
        kie_api_key,
        vertex_json_folder,
    ):
        ensure_profile_ready()
        try:
            from .nano_banana_aio import (
                OnyxNanoBananaAIO, _load_vertex_json_folder, _ensure_yunet_model,
            )
            from .instagram_faceswap import _load_skip_log, _save_skip_log, _do_face_swap
        except ImportError as e:
            return (torch.zeros(1, 64, 64, 3), f"❌ Could not import OnyxNanoBananaAIO: {e}")

        ns = load_remote("dataset_creator_core")
        return ns["run_impl"](
            face_reference, preset, resolution, custom_prompt, output_folder,
            trigger_word, test_mode, provider, disable_safety,
            gemini_api_key, wavespeed_api_key, fal_api_key, kie_api_key, vertex_json_folder,
            resources_dir=_RESOURCES_DIR,
            captions_cache=OnyxDatasetCreatorNode._CAPTIONS_CACHE,
            OnyxNanoBananaAIO=OnyxNanoBananaAIO,
            load_vertex_json_folder=_load_vertex_json_folder,
            load_skip_log=_load_skip_log,
            save_skip_log=_save_skip_log,
            do_face_swap=_do_face_swap,
            ensure_yunet_model=_ensure_yunet_model,
            pil_to_tensor=pil_to_tensor,
        )


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "OnyxDatasetCreatorNode": OnyxDatasetCreatorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxDatasetCreatorNode": "Onyx Dataset Creator",
}
