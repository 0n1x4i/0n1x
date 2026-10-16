"""
Onyx — Skin Tone Line

Recale automatiquement la teinte de peau sur la « skin tone line » du vectorscope
(l'axe I de l'espace YIQ, ~123° sur un vectorscope broadcast).
Corrige les peaux qui virent au marron-vert / jaune-vert (Krea, refine...) sans
toucher au reste : seuls les pixels dont la teinte/saturation/luminance
ressemblent à de la peau sont modifiés, avec un masque doux.

Léger : uniquement des opérations pixel à pixel + un petit flou du masque.
Luminance jamais modifiée -> pas de changement de volume/modelé, uniquement la couleur.
"""
import math
import torch
import torch.nn.functional as F
import os as _onyx_os, importlib.util as _onyx_ilu

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'skin_tone', __package__)
    return ns

_onyx_prof_path = _onyx_os.path.join(_onyx_os.path.dirname(_onyx_os.path.dirname(_onyx_os.path.abspath(__file__))), "nodes", "onyx_render_profile.py")
_onyx_prof_spec = _onyx_ilu.spec_from_file_location("onyx_render_profile_pp", _onyx_prof_path)
_onyx_prof_mod = _onyx_ilu.module_from_spec(_onyx_prof_spec)
_onyx_prof_spec.loader.exec_module(_onyx_prof_mod)
ensure_profile_ready = _onyx_prof_mod.ensure_profile_ready

_M = torch.tensor([[0.299, 0.587, 0.114],
                   [0.596, -0.274, -0.322],
                   [0.211, -0.523, 0.312]], dtype=torch.float32)
_MI = torch.linalg.inv(_M)


def _device(*args, **kwargs):
    return _onyx_core()['_device'](*args, **kwargs)



def _smooth(*args, **kwargs):
    return _onyx_core()['_smooth'](*args, **kwargs)



def _wrap(*args, **kwargs):
    return _onyx_core()['_wrap'](*args, **kwargs)



class Onyx_Skin_Tone:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.01}),
                "target_offset_deg": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.5,
                                                "tooltip": "Décale la cible. + = plus rosé, - = plus doré/jaune. 0 = skin tone line exacte"}),
                "uniformity": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "0 = décale toute la peau en bloc (garde les variations naturelles), "
                                                    "1 = ramène chaque pixel sur la ligne (peau très uniforme, risque d'effet plastique)"}),
                "saturation_fix": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Redonne de la saturation à une peau terne/grise (typique du marron-vert). 0 = teinte seule"}),
                "detection_width_deg": ("FLOAT", {"default": 35.0, "min": 15.0, "max": 60.0, "step": 1.0,
                                                  "tooltip": "Largeur de la fenêtre de teinte considérée comme peau. Plus large = attrape des peaux très déviées mais aussi bois/cuir"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optionnel : limite la correction (ex. masque personne/SAM) pour protéger bois, cuir, murs beiges"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "skin_mask")
    FUNCTION = "execute"
    CATEGORY = "Onyx/Post-Processing"

    def execute(self, *args, **kwargs):
        return _onyx_core()['Onyx_Skin_Tone'].execute(self, *args, **kwargs)
