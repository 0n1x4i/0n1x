import os
import numpy as np
import torch

# LUTS_DIR : dossier "luts/" à la racine du pack (un niveau au-dessus de "post processing/")
import os as _onyx_os, importlib.util as _onyx_ilu

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'apply_lut', __package__)
    return ns

_onyx_prof_path = _onyx_os.path.join(_onyx_os.path.dirname(_onyx_os.path.dirname(_onyx_os.path.abspath(__file__))), "nodes", "onyx_render_profile.py")
_onyx_prof_spec = _onyx_ilu.spec_from_file_location("onyx_render_profile_pp", _onyx_prof_path)
_onyx_prof_mod = _onyx_ilu.module_from_spec(_onyx_prof_spec)
_onyx_prof_spec.loader.exec_module(_onyx_prof_mod)
ensure_profile_ready = _onyx_prof_mod.ensure_profile_ready
LUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "luts")


def _list_luts():
    """Liste les fichiers .cube disponibles dans LUTS_DIR."""
    if not os.path.isdir(LUTS_DIR):
        return ["<aucun fichier .cube trouvé>"]
    files = sorted([f for f in os.listdir(LUTS_DIR) if f.lower().endswith(".cube")])
    return files if files else ["<aucun fichier .cube trouvé>"]


class Onyx_Apply_LUT:
    """
    Onyx Apply LUT
    Applique un fichier .cube (LUT 3D ou 1D) sur une image.
    Placez vos fichiers .cube dans le dossier luts/ à la racine du pack.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":            ("IMAGE",),
                "lut_file":         (_list_luts(),),
                "gamma_correction": ("BOOLEAN", {"default": True}),
                "clip_values":      ("BOOLEAN", {"default": True}),
                "strength":         ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "execute"
    CATEGORY      = "Onyx/Post-Processing"

    DESCRIPTION = (
        "Onyx Apply LUT\n"
        "Applies a .cube LUT file to your image.\n"
        "Place your .cube files in the luts/ folder at the pack root.\n"
        "gamma_correction: applies a 2.2 gamma correction before/after.\n"
        "clip_values: clips LUT values within the declared domain.\n"
        "strength: 1.0 = full effect, 0.0 = original image."
    )

    def execute(self, *args, **kwargs):
        return _onyx_core()['Onyx_Apply_LUT'].execute(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {
    "Onyx_Apply_LUT": Onyx_Apply_LUT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Onyx_Apply_LUT": "🎨 Onyx Apply LUT",
}
