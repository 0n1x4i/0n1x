"""
Onyx — Clarity / Dehaze / Match Levels

Contre le voile laiteux (haze) qu'ajoute une passe de refine (ex. Flux 2 Klein 9B
+ LoRAs en img2img à faible denoise) : noirs remontés, contraste local des
tons moyens écrasé.

  • Onyx Dehaze        — retire le voile additif (Dark Channel Prior + guided filter)
  • Onyx Clarity       — contraste local des tons moyens (≈ Clarté Lightroom/Photoshop)
  • Onyx Match Levels  — recale noirs/blancs du résultat refine sur l'image
                             AVANT refine (le plus fidèle quand on a la source)

Ordre conseillé :  refine → Match Levels (ref = avant refine) → Dehaze (léger) → Clarity
IMAGE ComfyUI = [B, H, W, C] float 0..1.
"""
import torch
import torch.nn.functional as F
import os as _onyx_os, importlib.util as _onyx_ilu

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'clarity_dehaze', __package__)
    return ns

_onyx_prof_path = _onyx_os.path.join(_onyx_os.path.dirname(_onyx_os.path.dirname(_onyx_os.path.abspath(__file__))), "nodes", "onyx_render_profile.py")
_onyx_prof_spec = _onyx_ilu.spec_from_file_location("onyx_render_profile_pp", _onyx_prof_path)
_onyx_prof_mod = _onyx_ilu.module_from_spec(_onyx_prof_spec)
_onyx_prof_spec.loader.exec_module(_onyx_prof_mod)
ensure_profile_ready = _onyx_prof_mod.ensure_profile_ready


# ───────────────────────── helpers ─────────────────────────

def _to_bchw(*args, **kwargs):
    return _onyx_core()['_to_bchw'](*args, **kwargs)



def _to_bhwc(*args, **kwargs):
    return _onyx_core()['_to_bhwc'](*args, **kwargs)



def _box(*args, **kwargs):
    return _onyx_core()['_box'](*args, **kwargs)



def _guided_filter(*args, **kwargs):
    return _onyx_core()['_guided_filter'](*args, **kwargs)



def _luma(*args, **kwargs):
    return _onyx_core()['_luma'](*args, **kwargs)



def _min_filter(*args, **kwargs):
    return _onyx_core()['_min_filter'](*args, **kwargs)



def _radius_px(*args, **kwargs):
    return _onyx_core()['_radius_px'](*args, **kwargs)



def _device(*args, **kwargs):
    return _onyx_core()['_device'](*args, **kwargs)



def _run_per_image(*args, **kwargs):
    return _onyx_core()['_run_per_image'](*args, **kwargs)



def _keep_alpha(*args, **kwargs):
    return _onyx_core()['_keep_alpha'](*args, **kwargs)



def _quantiles(*args, **kwargs):
    return _onyx_core()['_quantiles'](*args, **kwargs)



# ───────────────────────── Clarity ─────────────────────────

class Onyx_Clarity:
    """Contraste local des tons moyens, sur la luminance uniquement (pas de dérive couleur)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 2.0, "step": 0.01,
                                     "tooltip": "Négatif = adoucit (clarté négative)"}),
                "radius_pct": ("FLOAT", {"default": 2.5, "min": 0.2, "max": 10.0, "step": 0.1,
                                         "tooltip": "Rayon en % du petit côté. 1.5-4 = clarté, 0.3-0.8 = texture"}),
                "edge_preserve": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.2, "step": 0.0001,
                                            "tooltip": "eps du guided filter. Plus bas = moins de halos mais effet plus faible"}),
                "midtone_focus": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                                            "tooltip": "0 = tous les tons, plus haut = protège ombres et hautes lumières"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Onyx/Post-Processing"

    def execute(self, *args, **kwargs):
        return _onyx_core()['Onyx_Clarity'].execute(self, *args, **kwargs)



# ───────────────────────── Dehaze ─────────────────────────

class Onyx_Dehaze:
    """Dark Channel Prior (He et al. 2009) + raffinage guided filter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.98, "step": 0.01,
                                       "tooltip": "Part du voile estimé à retirer"}),
                "patch_pct": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 5.0, "step": 0.1,
                                        "tooltip": "Taille du patch dark channel en % du petit côté"}),
                "min_transmission": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.9, "step": 0.01,
                                               "tooltip": "Garde-fou : monter si les ombres se bouchent ou deviennent bruitées"}),
                "refine_radius_pct": ("FLOAT", {"default": 3.0, "min": 0.2, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "veil_map")
    FUNCTION = "execute"
    CATEGORY = "Onyx/Post-Processing"

    def execute(self, *args, **kwargs):
        return _onyx_core()['Onyx_Dehaze'].execute(self, *args, **kwargs)



# ───────────────────────── Match Levels ─────────────────────────

class Onyx_Match_Levels:
    """
    Recale le point noir / point blanc (par canal) de l'image refinée sur ceux de
    l'image avant refine. Tailles différentes acceptées (upscale dans le refine OK).
    Corrige précisément le voile ajouté par la passe, sans rien inventer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Sortie du refine (voilée)"}),
                "reference": ("IMAGE", {"tooltip": "Image AVANT le refine"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "black_pct": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1,
                                        "tooltip": "Percentile utilisé comme point noir"}),
                "white_pct": ("FLOAT", {"default": 99.5, "min": 90.0, "max": 100.0, "step": 0.1,
                                        "tooltip": "Percentile utilisé comme point blanc"}),
                "per_channel": ("BOOLEAN", {"default": True,
                                            "tooltip": "True = corrige aussi la teinte du voile. False = luminance seule"}),
                "match_midtones": ("BOOLEAN", {"default": True,
                                               "tooltip": "Recale aussi la médiane (gamma) pour retrouver la densité des tons moyens"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Onyx/Post-Processing"

    def execute(self, *args, **kwargs):
        return _onyx_core()['Onyx_Match_Levels'].execute(self, *args, **kwargs)
