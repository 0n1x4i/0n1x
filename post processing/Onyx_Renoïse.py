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
    "ISO 3200":        (0.045, 0.034, 0.75, 3.5, 0.62),
    "ISO 6400":        (0.060, 0.048, 0.85, 4.0, 0.55),
}

# ---------------------------------------------------------------------------
# "Look" presets: a full phone-camera degradation recipe (high ISO + phone noise
# reduction + JPEG). "Manual" uses the individual sliders below instead.
# Amounts are scaled by look_strength.
# ---------------------------------------------------------------------------
LOOK_KEYS = ("fine_grain", "grain_color", "mottle", "chroma_blotch", "blotch_size", "purple_green",
             "shadow_weight", "nr_smear", "chroma_bleed", "color_degrade", "jpeg_quality")
LOOKS = {
    "Manual": None,
    # Calibrated band-by-band on a real indoor low-light phone selfie
    # (luma and chroma noise energy per scale on flat skin).
    "Phone low light (iPhone indoor)": dict(
        fine_grain=0.010, grain_color=0.30, mottle=0.010, chroma_blotch=0.009, blotch_size=5.0,
        purple_green=0.55, shadow_weight=0.70, nr_smear=0.45, chroma_bleed=1.2, color_degrade=0.30,
        jpeg_quality=80),
    "Phone night (heavy)": dict(
        fine_grain=0.014, grain_color=0.35, mottle=0.014, chroma_blotch=0.016, blotch_size=6.5,
        purple_green=0.65, shadow_weight=0.80, nr_smear=0.55, chroma_bleed=1.6, color_degrade=0.45,
        jpeg_quality=74),
    "Phone indoor (subtle)": dict(
        fine_grain=0.007, grain_color=0.25, mottle=0.006, chroma_blotch=0.005, blotch_size=4.0,
        purple_green=0.50, shadow_weight=0.60, nr_smear=0.30, chroma_bleed=0.8, color_degrade=0.18,
        jpeg_quality=86),
}
# Which look values are amounts (scaled by look_strength). The others are shapes.
_LOOK_SCALED = ("fine_grain", "mottle", "chroma_blotch", "nr_smear", "chroma_bleed", "color_degrade")

_REF_SIDE = 2000.0  # sizes in px are expressed for a 2000 px long side, then scaled to the image


_TT = {
    "look": "Recette complete 'photo de telephone' (grain + taches chroma + lissage NR + couleurs degradees + JPEG). "
            "Manual = utilise les reglages ci-dessous. Avec un look, les reglages manuels sont ignores (look_strength reste actif). "
            "Conseil : iso_preset sur 'ISO 0 (Off)' ou 'ISO 200' avec un look, sinon les deux bruits s'additionnent.",
    "look_strength": "Multiplicateur global du bruit (grain fin, marbrures, taches chroma), en Manual comme avec un look. "
                     "0 = pas de bruit, 1 = normal, 2 = double, 5 = maximum. Avec un look, il dose aussi le lissage, la bavure, "
                     "la degradation des couleurs et le JPEG.",
    "fine_grain": "Grain fin pixel par pixel (luminance). 0.01 = realiste, 0.03 = fort.",
    "grain_color": "Part du grain fin qui est coloree (0 = grain monochrome, 1 = grain RVB independant).",
    "mottle": "Marbrures de luminance basse frequence (le 'bruit' par dessus le grain, typique des capteurs de telephone apres reduction de bruit).",
    "chroma_blotch": "Taches de couleur basse frequence (violet / vert / magenta). C'est l'effet 'zones de couleurs differentes' d'un ISO eleve. 0.009 = realiste (calibre sur une vraie photo), 0.02 = bien visible, 0.05 = extreme.",
    "blotch_size": "Taille des taches (marbrures + chroma) en px pour une image de 2000 px de cote (adapte automatiquement a la resolution).",
    "purple_green": "Teinte des taches : 0 = toutes les teintes, 1 = uniquement axe violet <-> vert (le plus typique des capteurs de telephone).",
    "shadow_weight": "Concentration du nouveau bruit dans les ombres (0 = uniforme, 1 = presque que dans les ombres).",
    "nr_smear": "Lissage 'reduction de bruit' de telephone : efface la micro-texture dans les zones plates (peau cireuse) en gardant les contours.",
    "chroma_bleed": "Bavure des couleurs de l'image (flou de la chroma seule, comme le 4:2:0 et le debruitage chroma). En px pour 2000 px. N'attenue plus les taches de bruit.",
    "color_degrade": "Degradation des couleurs : ombres desaturees, noirs releves et legerement olive, hautes lumieres ternes. Au-dessus de 1 = exagere.",
    "jpeg_quality": "Recompression JPEG reelle (0 = desactive). 75-85 = rendu photo de telephone partagee.",
}


class Onyx_Renoise:
    """
    Onyx Renoïse — Injects realistic digital sensor noise.

    Separates luminance grain (fine structure) from chroma scatter (color blotches),
    applies independent Gaussian softening to each, and biases the effect toward
    darker regions where real sensor noise is most visible.

    Optional phone-camera layer: fine grain, low-frequency luma mottle,
    purple/green chroma blotches, noise-reduction smear, chroma bleed,
    colour degradation and a real JPEG round-trip.
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
                    "tooltip": "ISO noise level. ISO 0 = off | ISO 50 = barely visible | ISO 100 = clean | ISO 1600 = heavy grain | ISO 3200/6400 = very heavy | Night Mode = chroma-heavy.",
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Taille des grains de bruit (PAS l'intensite, qui vient de l'ISO preset). 1.0 = pixel par pixel (comportement d'origine). Au-dessus de 1 = grains plus gros/visibles (bruit genere a resolution reduite puis agrandi en nearest-neighbor). En dessous de 1 (jusqu'a 0.1) = grain plus fin/serre (bruit genere a resolution superieure puis moyenne vers le bas).",
                }),
            },
            "optional": {
                "look":          (list(LOOKS.keys()), {"default": "Manual", "tooltip": _TT["look"]}),
                "look_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": _TT["look_strength"]}),
                "fine_grain":    ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.25, "step": 0.001, "tooltip": _TT["fine_grain"]}),
                "grain_color":   ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": _TT["grain_color"]}),
                "mottle":        ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.25, "step": 0.001, "tooltip": _TT["mottle"]}),
                "chroma_blotch": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.35, "step": 0.001, "tooltip": _TT["chroma_blotch"]}),
                "blotch_size":   ("FLOAT", {"default": 5.0, "min": 2.0, "max": 96.0, "step": 0.5, "tooltip": _TT["blotch_size"]}),
                "purple_green":  ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": _TT["purple_green"]}),
                "shadow_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": _TT["shadow_weight"]}),
                "nr_smear":      ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": _TT["nr_smear"]}),
                "chroma_bleed":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip": _TT["chroma_bleed"]}),
                "color_degrade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": _TT["color_degrade"]}),
                "jpeg_quality":  ("INT",   {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": _TT["jpeg_quality"]}),
            },
        }

    CATEGORY    = "Onyx/Post-Processing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION    = "renoise"

    DESCRIPTION = (
        "Onyx Renoïse\n"
        "Injects authentic digital camera sensor noise.\n"
        "Select an ISO preset — from ISO 0 (off) to ISO 6400 (very strong grain).\n"
        "Night Mode adds chroma-heavy noise for a low-light look.\n"
        "Optional 'look' / sliders: phone high-ISO rendering — fine grain, luma mottle, "
        "purple/green chroma blotches, noise-reduction smear, colour bleed and degradation, JPEG."
    )

    def _blur(self, *args, **kwargs):
        return _onyx_core()['Onyx_Renoise']._blur(self, *args, **kwargs)


    def _gen_grain_noise(self, *args, **kwargs):
        return _onyx_core()['Onyx_Renoise']._gen_grain_noise(self, *args, **kwargs)


    def renoise(self, *args, **kwargs):
        fn = _onyx_core()['Onyx_Renoise'].renoise
        # An older server-side core may not know the newer options yet: drop
        # the ones it does not accept instead of failing the whole run.
        try:
            import inspect
            params = inspect.signature(fn).parameters
            if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
                kwargs = {k: v for k, v in kwargs.items() if k in params}
        except (TypeError, ValueError):
            pass
        return fn(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {
    "Onyx_Renoise": Onyx_Renoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Onyx_Renoise": "🎞️ Onyx Renoïse",
}
