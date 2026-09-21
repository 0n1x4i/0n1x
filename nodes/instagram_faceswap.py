"""
Onyx Social Face Swap
Downloads posts from Instagram, Threads or Pinterest, filters images,
and runs the face swap automation on each one with model fallback chain.

The download/resume/quality-routing orchestration (OnyxInstagramFaceSwapNode.run)
lives server-side — see nodes/onyx_remote_exec.py. `_do_face_swap` (the
per-model provider dispatcher), `_has_face`, `_load_skip_log` / `_save_skip_log`
and the gallery-dl install helpers stay HERE, local: nodes/dataset_creator.py
imports the three named functions directly, so they have to keep existing as
ordinary module-level functions regardless of this split.
"""

import os
import sys
import json
import base64
import time
import torch
import numpy as np

from ..utils.image_utils import tensor_to_pil, pil_to_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Instaloader auto-install
# ─────────────────────────────────────────────────────────────────────────────

from .onyx_render_profile import ensure_profile_ready
from .onyx_remote_exec import load_remote

_GALLERY_DL_UPGRADE_MARKER = os.path.join(
    os.path.dirname(__file__), ".gallery_dl_last_upgrade")
_GALLERY_DL_UPGRADE_COOLDOWN = 24 * 3600  # 1 fois par jour max


def _pip_install_gallery_dl(upgrade: bool) -> bool:
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", "gallery-dl",
           "--quiet", "--break-system-packages"]
    if upgrade:
        cmd.insert(4, "--upgrade")
    subprocess.check_call(cmd, timeout=120)
    try:
        with open(_GALLERY_DL_UPGRADE_MARKER, "w") as fh:
            fh.write(str(time.time()))
    except OSError:
        pass
    return True


def _ensure_gallery_dl() -> bool:
    """Installe gallery-dl, et le met a jour tout seul.

    Instagram change son site frequemment, et gallery-dl sort de nouvelles
    versions rien que pour reparer ses extracteurs Instagram/Threads/Pinterest.
    L'ancien code n'installait qu'une fois ("if not already present"), donc la
    version se figeait pour toujours des sa premiere installation : le
    telechargement finissait par se casser silencieusement chez tout le monde,
    des le jour ou Instagram changeait quelque chose - sans jamais se reparer.

    On verifie donc, au plus une fois par jour (fichier marqueur horodate),
    s'il existe une mise a jour, et on l'installe.
    """
    try:
        import gallery_dl  # noqa: F401
        installed = True
    except ImportError:
        installed = False

    if not installed:
        print("[gallery-dl] gallery-dl not found — installing...")
        try:
            _pip_install_gallery_dl(upgrade=False)
            print("[gallery-dl] gallery-dl installed successfully.")
            return True
        except Exception as e:
            print(f"[gallery-dl] gallery-dl install failed: {e}")
            return False

    # Deja installe : on ne retente une mise a jour qu'une fois par jour, pour
    # ne pas ralentir chaque run avec un appel reseau a pip/PyPI.
    last = 0.0
    try:
        with open(_GALLERY_DL_UPGRADE_MARKER) as fh:
            last = float(fh.read().strip())
    except (OSError, ValueError):
        pass
    if time.time() - last > _GALLERY_DL_UPGRADE_COOLDOWN:
        try:
            print("[gallery-dl] Checking for a newer version (Instagram/Threads/"
                  "Pinterest extractors break often, this keeps them current)...")
            _pip_install_gallery_dl(upgrade=True)
        except Exception as e:
            print(f"[gallery-dl] Upgrade check failed (using current version): {e}")
    return True


def _force_upgrade_gallery_dl() -> bool:
    """Appelee quand un telechargement a echoue : force une mise a jour hors
    cooldown, au cas ou l'echec vienne justement d'une version perimee."""
    try:
        print("[gallery-dl] Download failed/empty — forcing an upgrade and retrying once...")
        return _pip_install_gallery_dl(upgrade=True)
    except Exception as e:
        print(f"[gallery-dl] Forced upgrade failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Skip log (processed.json per account folder)
# ─────────────────────────────────────────────────────────────────────────────

def _load_skip_log(folder: str) -> dict:
    path = os.path.join(folder, "processed.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_skip_log(folder: str, log: dict):
    path = os.path.join(folder, "processed.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Social] Warning: could not save skip log: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Face-swap dispatcher  (extends _run_face_swap_automation to GPT2 / Seedream)
# ─────────────────────────────────────────────────────────────────────────────

def _has_face(pil_img) -> bool:
    """Quick face presence check using YuNet (same model as _mask_face_in_tensor).
    Returns True if at least one face is detected, False otherwise."""
    try:
        import cv2
        import numpy as np
        img_rgb = np.array(pil_img.convert("RGB"))
        img_bgr = img_rgb[:, :, ::-1].copy()
        h, w    = img_bgr.shape[:2]
        model_dir  = os.path.join(os.path.dirname(__file__), ".yunet_cache")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
        # Meme garde que dans nano_banana_aio : os.path.exists() seul laisse un
        # telechargement rate (page HTML, pointeur git-LFS, fichier tronque) en
        # cache pour toujours, et le detecteur ne trouve alors plus aucun visage.
        from .nano_banana_aio import _ensure_yunet_model
        if not _ensure_yunet_model(model_path):
            print("⚠️  [FaceCheck] No usable detection model — 0 face will be reported.")
            return False
        detector = cv2.FaceDetectorYN.create(
            model_path, "", (w, h),
            score_threshold=0.6, nms_threshold=0.3, top_k=5000,
        )
        min_face = min(w, h) * 0.04
        _, detections = detector.detect(img_bgr)
        if detections is not None:
            for det in detections:
                fw, fh = int(det[2]), int(det[3])
                if fw >= min_face and fh >= min_face:
                    return True
        return False
    except Exception as e:
        print(f"⚠️  [FaceCheck] Detection failed ({e}) — assuming face present")
        return True  # fail-safe: don't skip if detection errors


_EXPRESSION_MAP = {
    "Auto":         "",
    "Neutral":      "neutral facial expression",
    "Sensual":      "sensual facial expression, intense eyes look",
    "Playful":      "playful facial expression",
    "Subtle smile": "subtle smile",
    "Smile":        "warm smile",
    "Laugh":        "she's laughing",
}


def _do_face_swap(
    aio,
    model_name:             str,
    face_ref_tensor,        # IMAGE tensor  (the reference face)
    target_tensor,          # IMAGE tensor  (the Instagram image)
    provider:               str,
    image_size:             str,
    disable_safety:         bool,
    face_expression:        str,
    gemini_api_key:         str,
    wavespeed_api_key:      str,
    kie_api_key:            str,
    fal_api_key:            str,
    vertex_json_folder:     str,
    vertex_location:        str,
    vertex_json_file_override: str = "",   # single JSON file — bypasses folder scan
    custom_prompt:          str = "",      # appended to face swap prompt
    yunet_score_threshold:  float = 0.7,
    temperature:            float = 1.0,
):
    """
    Route a single face-swap to the right provider method.
    NB Pro / NB2  → _run_face_swap_automation (existing, well-tested).
    GPT2 / Seedream → manual routing using the same masked_image + fs_prompt.
    """
    # ── NB Pro / NB2 use the existing automation (unless Auto expression) ────────
    if model_name in ("Nano Banana Pro", "Nano Banana 2") and face_expression != "Auto":
        return aio._run_face_swap_automation(
            yunet_score_threshold  = yunet_score_threshold,
            provider               = provider,
            model                  = model_name,
            image_size             = image_size,
            aspect_ratio           = "1:1",   # auto-detected inside _run_face_swap_automation
            batch_size             = 1,
            disable_safety         = disable_safety,
            face_expression        = face_expression,
            image_1                = face_ref_tensor,
            image_2                = target_tensor,
            gemini_api_key         = gemini_api_key,
            wavespeed_api_key      = wavespeed_api_key,
            kie_api_key            = kie_api_key,
            fal_api_key            = fal_api_key,
            vertex_json_folder     = vertex_json_folder,
            breast_refiner_enabled = False,
            low_neck_enabled       = False,
            face_swap_custom_prompt = custom_prompt,
        )

    # ── Auto expression: build prompt + masked image manually (no expression) ────
    from .nano_banana_aio import (
        _FACESWAP_PROMPT_B64,
        _SEEDREAM5PRO_FACESWAP_PROMPT_B64,
        _MODEL_MAP,
    )

    # Mask face in target image (same logic as the existing automation)
    masked_target = aio._mask_face_in_tensor(target_tensor)

    # Decode face-swap prompt — Seedream 5 Pro always uses its dedicated prompt
    if model_name == "Seedream 5 Pro":
        fs_prompt = base64.b64decode(_SEEDREAM5PRO_FACESWAP_PROMPT_B64).decode("utf-8")
    else:
        fs_prompt = base64.b64decode(_FACESWAP_PROMPT_B64).decode("utf-8")
    _expr = _EXPRESSION_MAP.get(face_expression, "neutral facial expression")
    if _expr:
        fs_prompt += f"\n\n{_expr}"
    if custom_prompt:
        fs_prompt += f"\n\n{custom_prompt}"

    image_tensors = [face_ref_tensor, masked_target]

    # Auto-detect aspect ratio from target image
    auto_ratio = aio._detect_aspect_ratio(target_tensor)

    # ── NB Pro / NB2 (Auto expression only) ──────────────────────────────────
    if model_name in ("Nano Banana Pro", "Nano Banana 2"):
        is_nb2 = (model_name == "Nano Banana 2")
        if provider == "WAVESPEED":
            return aio._generate_wavespeed(
                fs_prompt, image_tensors, image_size, "jpeg",
                auto_ratio, batch_size=1,
                ws_api_key=wavespeed_api_key, is_nb2=is_nb2,
            )
        elif provider == "KIE":
            return aio._generate_kie(
                fs_prompt, image_tensors, image_size, auto_ratio,
                kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                batch_size=1, is_nb2=is_nb2,
            )
        elif provider == "FAL":
            return aio._generate_fal(
                prompt            = fs_prompt,
                image_tensors     = image_tensors,
                image_size        = image_size,
                aspect_ratio      = auto_ratio,
                batch_size        = 1,
                fal_api_key       = fal_api_key,
                is_nb2            = is_nb2,
                safety_tolerance  = "4",
                enable_web_search = False,
            )
        elif provider == "GOOGLE":
            if not gemini_api_key:
                return aio._handle_error("❌ gemini_api_key missing for GOOGLE provider.")
            if is_nb2:
                return aio._generate_nb2_google(
                    prompt=fs_prompt, image_tensors=image_tensors,
                    image_size=image_size, aspect_ratio=auto_ratio,
                    temperature=temperature, g_key=gemini_api_key,
                    batch_size=1, disable_safety=disable_safety,
                )
            return aio._generate_google(
                prompt=fs_prompt, image_tensors=image_tensors,
                image_size=image_size, aspect_ratio=auto_ratio,
                temperature=temperature, top_p=0.95, use_search=False,
                system_instructions=None, batch_size=1,
                g_key=gemini_api_key,
                safety_threshold="OFF" if disable_safety else None,
            )
        elif provider == "VERTEX":
            if not vertex_json_folder and not vertex_json_file_override:
                return aio._handle_error("❌ vertex_json_folder missing for VERTEX provider.")
            from .nano_banana_aio import _load_vertex_json_folder
            vj_files = ([vertex_json_file_override] if vertex_json_file_override
                        else _load_vertex_json_folder(vertex_json_folder))
            if is_nb2:
                return aio._generate_nb2_vertex(
                    prompt=fs_prompt, image_tensors=image_tensors,
                    image_size=image_size, aspect_ratio=auto_ratio,
                    temperature=temperature, vertex_json_files=vj_files,
                    vertex_location=vertex_location, batch_size=1,
                    disable_safety=disable_safety,
                )
            return aio._generate_vertex(
                prompt=fs_prompt, image_tensors=image_tensors,
                image_size=image_size, aspect_ratio=auto_ratio,
                temperature=temperature, top_p=0.95, use_search=False,
                system_instructions=None, batch_size=1,
                vertex_json_files=vj_files, vertex_location=vertex_location,
                model_name=_MODEL_MAP.get(model_name, "gemini-3-pro-image"),
                safety_threshold="BLOCK_NONE" if disable_safety else None,
            )
        else:
            raise ValueError(f"NB Pro/NB2 is not available on provider '{provider}'.")

    # ── GPT Image 2.0 ─────────────────────────────────────────────────────────
    if model_name == "GPT Image 2.0":
        if provider == "KIE":
            return aio._generate_gpt2_kie(
                fs_prompt, image_tensors, auto_ratio, image_size,
                kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                batch_size=1, disable_safety=disable_safety,
            )
        elif provider == "FAL":
            return aio._generate_gpt2_fal(
                fs_prompt, image_tensors, auto_ratio, image_size,
                gpt2_quality="high", batch_size=1, fal_api_key=fal_api_key,
            )
        elif provider == "WAVESPEED":
            return aio._generate_gpt2_wavespeed(
                fs_prompt, image_tensors, auto_ratio, image_size,
                batch_size=1, ws_api_key=wavespeed_api_key,
            )
        else:
            raise ValueError(f"GPT Image 2.0 is not available on provider '{provider}'.")

    # ── Seedream 4.5 ──────────────────────────────────────────────────────────
    if model_name == "Seedream 4.5":
        if provider == "KIE":
            return aio._generate_seedream_kie(
                fs_prompt, image_tensors, image_size, auto_ratio,
                kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                batch_size=1, disable_safety=disable_safety,
            )
        elif provider == "FAL":
            return aio._generate_seedream_fal(
                fs_prompt, image_tensors, image_size, auto_ratio,
                batch_size=1, fal_api_key=fal_api_key,
                disable_safety=disable_safety,
            )
        elif provider == "WAVESPEED":
            return aio._generate_seedream_wavespeed(
                fs_prompt, image_tensors, image_size, auto_ratio,
                batch_size=1, ws_api_key=wavespeed_api_key,
                disable_safety=disable_safety,
            )
        else:
            raise ValueError(f"Seedream 4.5 is not available on provider '{provider}'.")

    # ── Seedream 5 Pro ────────────────────────────────────────────────────────
    if model_name == "Seedream 5 Pro":
        if provider == "KIE":
            return aio._generate_seedream5pro_kie(
                fs_prompt, image_tensors, image_size, auto_ratio,
                kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                batch_size=1, disable_safety=disable_safety,
            )
        elif provider == "FAL":
            return aio._generate_seedream5pro_fal(
                fs_prompt, image_tensors, image_size, auto_ratio,
                batch_size=1, fal_api_key=fal_api_key,
                disable_safety=disable_safety,
            )
        elif provider == "WAVESPEED":
            return aio._generate_seedream5pro_wavespeed(
                fs_prompt, image_tensors, image_size, auto_ratio,
                batch_size=1, ws_api_key=wavespeed_api_key,
                disable_safety=disable_safety,
            )
        else:
            raise ValueError(f"Seedream 5 Pro is not available on provider '{provider}'.")

    raise ValueError(f"Unknown model: '{model_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class OnyxInstagramFaceSwapNode:
    """
    Onyx Content Remaker
    Downloads posts from Instagram, Threads or Pinterest, filters images (skips videos),
    and runs the face swap automation on each image.
    Supports a model fallback chain: NB Pro → NB2 → GPT Image 2.
    Low-light + blurry images can be automatically routed to GPT Image 2 first.
    Already processed images are skipped on re-run (resume support).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_reference": ("IMAGE",),

                "source_url": ("STRING", {
                    "default": "https://www.instagram.com/username/",
                    "tooltip": "Instagram, Threads or Pinterest URL.\n"
                               "• Instagram : https://www.instagram.com/username/\n"
                               "• Threads   : https://www.threads.net/@username/\n"
                               "• Pinterest : https://www.pinterest.com/username/ or /username/board/",
                }),
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder where swapped images will be saved. "
                               "A sub-folder named after the account will be created.",
                }),
                "max_posts": ("INT", {
                    "default": 50, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "Maximum number of image posts to process. 0 = all.",
                }),

                # ── Provider ──────────────────────────────────────────────────
                "provider": (["WAVESPEED", "GOOGLE", "FAL", "KIE", "VERTEX"], {
                    "default": "WAVESPEED",
                    "tooltip": "API provider. GPT Image 2 only works on KIE / FAL / WAVESPEED.",
                }),

                # ── Model toggles ─────────────────────────────────────────────
                "use_nano_banana_pro": ("BOOLEAN", {
                    "default": True,
                    "label_on":  "Nano Banana Pro ✅",
                    "label_off": "Nano Banana Pro ❌",
                }),
                "use_nano_banana_2": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "Nano Banana 2 ✅",
                    "label_off": "Nano Banana 2 ❌",
                }),
                "use_seedream_5_pro": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "Seedream 5 Pro ✅",
                    "label_off": "Seedream 5 Pro ❌",
                    "tooltip": "Available on KIE, FAL, WAVESPEED only.",
                }),
                "use_gpt_image_2": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "GPT Image 2 ✅",
                    "label_off": "GPT Image 2 ❌",
                    "tooltip": "Available on KIE, FAL, WAVESPEED only.",
                }),

                # ── Quality routing ───────────────────────────────────────────
                "auto_quality_routing": ("BOOLEAN", {
                    "default": True,
                    "label_on":  "Quality Routing ON",
                    "label_off": "Quality Routing OFF",
                    "tooltip": "When ON: low-light AND blurry images will use GPT Image 2 first "
                               "(if selected and provider supports it).",
                }),

                # ── Generation settings ───────────────────────────────────────
                "retry_count": ("INT", {
                    "default": 3, "min": 1, "max": 5, "step": 1,
                    "tooltip": "Number of attempts per model before moving to the next one in the fallback chain.",
                }),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "face_expression": (
                    ["Auto", "Neutral", "Sensual", "Playful", "Subtle smile", "Smile", "Laugh"],
                    {"default": "Auto"},
                ),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional extra instructions appended to the face swap prompt. "
                        "Examples: 'Preserve heterochromia', 'Keep the freckles', "
                        "'The character has a beauty mark under her left eye'."
                    ),
                }),
                "disable_safety": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "Safety OFF",
                    "label_off": "Safety ON",
                }),

                # ── Instagram auth (optional, fixes 403 blocks) ──────────────────
                "cookies_file": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to a cookies.txt file exported from your browser (Netscape format). "
                        "How to get it: install the 'Get cookies.txt LOCALLY' extension in Chrome, "
                        "go to instagram.com while logged in, click the extension and export. "
                        "This is the safest method and supports 2FA accounts. "
                        "Leave empty to browse anonymously (may cause 403 blocks)."
                    ),
                }),

                # ── API keys ──────────────────────────────────────────────────
                "gemini_api_key":      ("STRING", {"default": ""}),
                "wavespeed_api_key":   ("STRING", {"default": ""}),
                "fal_api_key":         ("STRING", {"default": ""}),
                "kie_api_key":         ("STRING", {"default": ""}),
                "vertex_json_folder":  ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("last_image", "summary")
    OUTPUT_NODE   = True
    FUNCTION      = "run"
    CATEGORY      = "Onyx/NanoBanana"
    DESCRIPTION   = (
        "Onyx Content Remaker\n"
        "Downloads posts from Instagram, Threads or Pinterest and runs the face swap automation "
        "on every image.\n"
        "Fallback chain: Nano Banana Pro → Nano Banana 2 → Seedream 5 Pro → GPT Image 2.\n"
        "Low-light + blurry images can be routed to GPT Image 2 first.\n"
        "Already processed images are skipped on re-run (resume support)."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # ─────────────────────────────────────────────────────────────────────────
    def run(
        self,
        face_reference,
        source_url,
        output_folder,
        max_posts,
        provider,
        use_nano_banana_pro,
        use_nano_banana_2,
        use_seedream_5_pro,
        use_gpt_image_2,
        auto_quality_routing,
        retry_count,
        image_size,
        face_expression,
        disable_safety,
        gemini_api_key,
        wavespeed_api_key,
        kie_api_key,
        fal_api_key,
        vertex_json_folder,
        cookies_file,
        custom_prompt,
    ):
        ensure_profile_ready()
        from .nano_banana_aio import OnyxNanoBananaAIO, _load_vertex_json_folder

        ns = load_remote("instagram_faceswap_core")
        return ns["run_impl"](
            face_reference, source_url, output_folder, max_posts, provider,
            use_nano_banana_pro, use_nano_banana_2, use_seedream_5_pro, use_gpt_image_2,
            auto_quality_routing, retry_count, image_size, face_expression,
            disable_safety, gemini_api_key, wavespeed_api_key, kie_api_key,
            fal_api_key, vertex_json_folder, cookies_file, custom_prompt,
            OnyxNanoBananaAIO=OnyxNanoBananaAIO,
            ensure_gallery_dl=_ensure_gallery_dl,
            force_upgrade_gallery_dl=_force_upgrade_gallery_dl,
            has_face=_has_face,
            do_face_swap=_do_face_swap,
            load_skip_log=_load_skip_log,
            save_skip_log=_save_skip_log,
            load_vertex_json_folder=_load_vertex_json_folder,
            pil_to_tensor=pil_to_tensor,
        )


# ─────────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "OnyxInstagramFaceSwapNode": OnyxInstagramFaceSwapNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxInstagramFaceSwapNode": "Onyx Content Remaker",
}
