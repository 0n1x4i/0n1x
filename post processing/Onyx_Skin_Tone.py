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
_onyx_prof_path = _onyx_os.path.join(_onyx_os.path.dirname(_onyx_os.path.dirname(_onyx_os.path.abspath(__file__))), "nodes", "onyx_render_profile.py")
_onyx_prof_spec = _onyx_ilu.spec_from_file_location("onyx_render_profile_pp", _onyx_prof_path)
_onyx_prof_mod = _onyx_ilu.module_from_spec(_onyx_prof_spec)
_onyx_prof_spec.loader.exec_module(_onyx_prof_mod)
ensure_profile_ready = _onyx_prof_mod.ensure_profile_ready

_M = torch.tensor([[0.299, 0.587, 0.114],
                   [0.596, -0.274, -0.322],
                   [0.211, -0.523, 0.312]], dtype=torch.float32)
_MI = torch.linalg.inv(_M)


def _device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _smooth(a, lo, hi):
    t = ((a - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


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

    def execute(self, image, strength, target_offset_deg, uniformity, saturation_fix,
                detection_width_deg, mask=None):
        ensure_profile_ready()
        dev = _device()
        M, MI = _M.to(dev), _MI.to(dev)
        tol = detection_width_deg
        target = math.radians(target_offset_deg)
        outs, masks = [], []

        for i in range(image.shape[0]):
            x = image[i].to(dev, torch.float32)          # [H,W,C]
            rgb = x[..., :3]
            yiq = rgb @ M.T
            Y, I, Q = yiq[..., 0], yiq[..., 1], yiq[..., 2]
            C = torch.sqrt(I * I + Q * Q)
            th = torch.atan2(Q, I)
            d = _wrap(th - target)
            dd = torch.rad2deg(d)
            s = C / Y.clamp(min=1e-3)

            # ── détection peau (fenêtre asymétrique : plus large côté jaune/vert) ──
            wh = torch.where(dd < 0, _smooth(dd, -1.4 * tol, -0.6 * tol), 1 - _smooth(dd, 0.6 * tol, tol))
            ws = _smooth(s, 0.08, 0.15) * (1 - _smooth(s, 0.6, 0.8))   # exclut blancs/gris et couleurs criardes
            wy = _smooth(Y, 0.08, 0.18) * (1 - _smooth(Y, 0.9, 0.98))  # exclut noirs et hautes lumières brûlées
            m = wh * ws * wy

            H, W = m.shape
            if mask is not None:
                mk = mask[min(i, mask.shape[0] - 1)].to(dev, torch.float32)
                if mk.shape != m.shape:
                    mk = F.interpolate(mk[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                m = m * mk

            k = max(3, int(min(H, W) * 0.008) | 1)
            m = F.avg_pool2d(m[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0]

            msum = m.sum()
            if msum < 1e-3 or strength == 0:
                outs.append(x.cpu()); masks.append(m.cpu()); continue

            # teinte moyenne de la peau (moyenne circulaire pondérée par chroma)
            w = m * C
            mean = torch.atan2((w * torch.sin(d)).sum(), (w * torch.cos(d)).sum())
            mean_sat = (m * s).sum() / msum

            rot = strength * m * ((1 - uniformity) * (-mean) + uniformity * (-d))
            gain_target = (0.28 / mean_sat.clamp(min=1e-3)).clamp(0.7, 1.5)
            gain = 1 + strength * m * saturation_fix * (gain_target - 1)

            th2 = th + rot
            C2 = C * gain
            out = torch.stack([Y, C2 * torch.cos(th2), C2 * torch.sin(th2)], -1) @ MI.T
            out = out.clamp(0, 1)
            if x.shape[-1] > 3:
                out = torch.cat([out, x[..., 3:]], -1)

            print(f"[Onyx Skin Tone] teinte peau {math.degrees(mean.item()):+.1f}° vs cible, "
                  f"saturation {mean_sat.item():.3f}, couverture {m.mean().item() * 100:.1f}%")
            outs.append(out.cpu()); masks.append(m.cpu())

        return (torch.stack(outs).to(image.dtype), torch.stack(masks))
