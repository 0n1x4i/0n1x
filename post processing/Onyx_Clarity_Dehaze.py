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
_onyx_prof_path = _onyx_os.path.join(_onyx_os.path.dirname(_onyx_os.path.dirname(_onyx_os.path.abspath(__file__))), "nodes", "onyx_render_profile.py")
_onyx_prof_spec = _onyx_ilu.spec_from_file_location("onyx_render_profile_pp", _onyx_prof_path)
_onyx_prof_mod = _onyx_ilu.module_from_spec(_onyx_prof_spec)
_onyx_prof_spec.loader.exec_module(_onyx_prof_mod)
ensure_profile_ready = _onyx_prof_mod.ensure_profile_ready


# ───────────────────────── helpers ─────────────────────────

def _to_bchw(img):
    return img.permute(0, 3, 1, 2).contiguous()


def _to_bhwc(img):
    return img.permute(0, 2, 3, 1).contiguous()


def _box(x, r):
    k = 2 * r + 1
    return F.avg_pool2d(x, k, stride=1, padding=r, count_include_pad=False)


def _guided_filter(I, p, r, eps):
    """Guided filter (He et al.) : flou qui respecte les bords -> pas de halo cheveux/ciel."""
    mean_I = _box(I, r)
    mean_p = _box(p, r)
    cov_Ip = _box(I * p, r) - mean_I * mean_p
    var_I = _box(I * I, r) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return _box(a, r) * I + _box(b, r)


def _luma(x):
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def _min_filter(x, k):
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)


def _radius_px(pct, h, w):
    # rayon en % du petit côté -> même rendu quelle que soit la résolution
    return max(1, int(round(min(h, w) * pct / 100.0)))


def _device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_per_image(image, fn):
    """Traite le batch image par image sur GPU (évite les pics VRAM en 4K)."""
    dev = _device()
    outs, extras = [], []
    for i in range(image.shape[0]):
        x = _to_bchw(image[i:i + 1]).to(dev, torch.float32)
        res = fn(x)
        if isinstance(res, tuple):
            outs.append(res[0].cpu())
            extras.append(res[1].cpu())
        else:
            outs.append(res.cpu())
    out = _to_bhwc(torch.cat(outs, 0)).to(image.dtype)
    if extras:
        return out, torch.cat(extras, 0)
    return out


def _keep_alpha(x, rgb):
    return torch.cat([rgb, x[:, 3:]], dim=1) if x.shape[1] > 3 else rgb


def _quantiles(v, qs):
    # torch.quantile est limité à 16M éléments -> sous-échantillonnage régulier
    v = v.flatten()
    if v.numel() > 4_000_000:
        v = v[:: (v.numel() // 4_000_000) + 1]
    return torch.quantile(v, torch.tensor(qs, device=v.device, dtype=v.dtype))


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

    def execute(self, image, amount, radius_pct, edge_preserve, midtone_focus):
        ensure_profile_ready()
        def fn(x):
            rgb = x[:, :3]
            _, _, h, w = rgb.shape
            L = _luma(rgb)
            base = _guided_filter(L, L, _radius_px(radius_pct, h, w), edge_preserve)
            mask = (1.0 - (2.0 * L - 1.0) ** 2).clamp(0, 1)
            mask = mask ** midtone_focus if midtone_focus > 0 else torch.ones_like(mask)
            out = (rgb + amount * (L - base) * mask).clamp(0, 1)
            return _keep_alpha(x, out)

        return (_run_per_image(image, fn),)


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

    def execute(self, image, strength, patch_pct, min_transmission, refine_radius_pct):
        ensure_profile_ready()
        def fn(x):
            I = x[:, :3]
            _, _, h, w = I.shape
            k = _radius_px(patch_pct, h, w) * 2 + 1

            dark = _min_filter(I.min(dim=1, keepdim=True).values, k)
            # airlight = couleur moyenne des 0.1% pixels les plus voilés
            n = max(1, int(h * w * 0.001))
            idx = dark.reshape(-1).topk(n).indices
            A = I.reshape(3, -1)[:, idx].mean(dim=1).clamp(min=0.05).view(1, 3, 1, 1)

            t = 1.0 - strength * _min_filter((I / A).min(dim=1, keepdim=True).values, k)
            t = _guided_filter(_luma(I), t, _radius_px(refine_radius_pct, h, w), 1e-3)
            t = t.clamp(min_transmission, 1.0)

            J = ((I - A) / t + A).clamp(0, 1)
            return _keep_alpha(x, J), (1.0 - t)[:, 0]

        out, veil = _run_per_image(image, fn)
        return (out, veil)


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

    def execute(self, image, reference, strength, black_pct, white_pct, per_channel, match_midtones):
        ensure_profile_ready()
        dev = _device()
        qs = [black_pct / 100.0, 0.5, white_pct / 100.0]
        outs = []
        for i in range(image.shape[0]):
            x = _to_bchw(image[i:i + 1]).to(dev, torch.float32)
            r = _to_bchw(reference[min(i, reference.shape[0] - 1):][:1]).to(dev, torch.float32)
            rgb, ref = x[:, :3], r[:, :3]

            if per_channel:
                chans_x = [rgb[:, c:c + 1] for c in range(3)]
                chans_r = [ref[:, c:c + 1] for c in range(3)]
            else:
                chans_x, chans_r = [_luma(rgb)], [_luma(ref)]

            mapped = []
            for cx, cr in zip(chans_x, chans_r):
                lo_x, md_x, hi_x = _quantiles(cx, qs)
                lo_r, md_r, hi_r = _quantiles(cr, qs)
                n = ((cx - lo_x) / (hi_x - lo_x).clamp(min=1e-4)).clamp(0, 1)
                if match_midtones:
                    # gamma pour que la médiane normalisée tombe au même endroit que la référence
                    mx = ((md_x - lo_x) / (hi_x - lo_x).clamp(min=1e-4)).clamp(1e-3, 1 - 1e-3)
                    mr = ((md_r - lo_r) / (hi_r - lo_r).clamp(min=1e-4)).clamp(1e-3, 1 - 1e-3)
                    g = (torch.log(mr) / torch.log(mx)).clamp(0.5, 2.0)
                    n = n ** g
                mapped.append(n * (hi_r - lo_r) + lo_r)

            if per_channel:
                new = torch.cat(mapped, dim=1)
            else:
                L = _luma(rgb)
                new = rgb * (mapped[0] / L.clamp(min=1e-4))

            out = (rgb + strength * (new - rgb)).clamp(0, 1)
            outs.append(_keep_alpha(x, out).cpu())

        return (_to_bhwc(torch.cat(outs, 0)).to(image.dtype),)
