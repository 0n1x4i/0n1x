import torch
import torch.nn.functional as F
import math

import os as _onyx_os, importlib.util as _onyx_ilu

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'renoise', __package__)
    return ns

_onyx_prof_path = _onyx_os.path.join(_onyx_os.path.dirname(_onyx_os.path.dirname(_onyx_os.path.abspath(__file__))), "nodes", "onyx_render_profile.py")
_onyx_prof_spec = _onyx_ilu.spec_from_file_location("onyx_render_profile_pp", _onyx_prof_path)
_onyx_prof_mod = _onyx_ilu.module_from_spec(_onyx_prof_spec)
_onyx_prof_spec.loader.exec_module(_onyx_prof_mod)
ensure_profile_ready = _onyx_prof_mod.ensure_profile_ready
try:
    import kornia.filters as kfilters
    import kornia.color as kcolor
    KORNIA_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    KORNIA_AVAILABLE = False
    print("⚠️ Onyx Renoïse: Kornia not found. Install with 'pip install kornia'.")

# ---------------------------------------------------------------------------
# ISO preset configurations: (sensor_grain, color_scatter, grain_softness, color_softness, shadow_emphasis)
# ---------------------------------------------------------------------------
ISO_PRESETS = {
    "ISO 0 (Off)":     (0.000, 0.000, 0.00, 0.0, 0.00),
    "ISO 50":          (0.004, 0.002, 0.35, 1.2, 0.98),
    "ISO 100 (Clean)": (0.008, 0.005, 0.40, 1.5, 0.95),
    "ISO 200":         (0.012, 0.008, 0.45, 1.8, 0.90),
    "ISO 400":         (0.018, 0.012, 0.50, 2.0, 0.85),
    "ISO 800":         (0.025, 0.016, 0.55, 2.5, 0.78),
    "ISO 1600":        (0.035, 0.025, 0.65, 3.0, 0.70),
    "Night Mode":      (0.025, 0.035, 1.20, 4.0, 0.50),
}


class Onyx_Renoise:
    """
    Onyx Renoïse — Injects realistic digital sensor noise.

    Separates luminance grain (fine structure) from chroma scatter (color blotches),
    applies independent Gaussian softening to each, and biases the effect toward
    darker regions where real sensor noise is most visible.
    """

    @classmethod
    def INPUT_TYPES(cls):
        if not KORNIA_AVAILABLE:
            return {
                "required": {
                    "error": ("STRING", {
                        "default": "Kornia not installed. Run: pip install kornia",
                        "multiline": True,
                    })
                }
            }

        return {
            "required": {
                "image":      ("IMAGE",),
                "seed":       ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": "randomize",
                }),
                "iso_preset": (list(ISO_PRESETS.keys()), {
                    "default": "ISO 400",
                    "tooltip": "ISO noise level. ISO 0 = off | ISO 50 = barely visible | ISO 100 = clean | ISO 1600 = heavy grain | Night Mode = chroma-heavy.",
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Taille des grains de bruit (PAS l'intensite, qui vient de l'ISO preset). 1.0 = pixel par pixel (comportement d'origine). Au-dessus de 1 = grains plus gros/visibles (bruit genere a resolution reduite puis agrandi en nearest-neighbor). En dessous de 1 (jusqu'a 0.1) = grain plus fin/serre (bruit genere a resolution superieure puis moyenne vers le bas).",
                }),
            },
        }

    CATEGORY    = "Onyx/Post-Processing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION    = "renoise"

    DESCRIPTION = (
        "Onyx Renoïse\n"
        "Injects authentic digital camera sensor noise.\n"
        "Select an ISO preset — from ISO 0 (off) to ISO 1600 (strong grain).\n"
        "Night Mode adds chroma-heavy noise for a low-light look."
    )

    def _blur(self, *args, **kwargs):
        return _onyx_core()['Onyx_Renoise']._blur(self, *args, **kwargs)


    def _gen_grain_noise(self, *args, **kwargs):
        return _onyx_core()['Onyx_Renoise']._gen_grain_noise(self, *args, **kwargs)


    def renoise(self, *args, **kwargs):
        return _onyx_core()['Onyx_Renoise'].renoise(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {
    "Onyx_Renoise": Onyx_Renoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Onyx_Renoise": "🎞️ Onyx Renoïse",
}
