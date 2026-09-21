"""
OnyxNanoBananaAIO — Unified multimodal node.
Google Gemini (API Studio) + WaveSpeed + Kie.ai + Fal.ai + Vertex AI
nano-banana-pro + nano-banana-2 + Seedream 4.5 + Seedream 5 Pro.
API Keys: connect an ApiKeysLoaderNode to the corresponding inputs.

The ~80 provider-specific generation methods (generate_unified and everything
it calls) live server-side — see nodes/onyx_remote_exec.py. This file keeps:
INPUT_TYPES / RETURN_TYPES (so ComfyUI can register the node with no network
call), the presets REST routes (must run at import time, not at call time),
and the handful of helpers/constants other node files in this pack import
directly and at zero cost: _detect_aspect_ratio, _ensure_yunet_model,
_load_vertex_json_folder, _mp4_strip_metadata, _MODEL_MAP,
_FACESWAP_PROMPT_B64 / _SEEDREAM5PRO_FACESWAP_PROMPT_B64 (the last two also
mirrored in the protected copy, since the class's own methods need them too —
duplicating a couple of base64 strings is cheap).

Unlike every other protected node in this pack, the remote module here is
fetched ONCE per OnyxNanoBananaAIO() instantiation rather than once per
method call — see the class below and nodes/onyx_remote_exec.py's docstring
for why: this class's methods are called in tight loops (once per image, per
retry, per segment) by this node and by Dataset Creator / Content Remaker /
Re-pose, and a full network fetch of a ~9000-line module per call would be a
real slowdown. A revoked key still stops this node, just on its next
instantiation (next node run) rather than on its very next method call.

Known, accepted trade-off from that instance-level fetch: _vertex_rotation_offset
and _failed_upload_services (both mutable class attributes on the remote
class, used to round-robin Vertex service-account JSON files and to briefly
skip an upload host that just failed) reset every time a fresh instance is
created, instead of persisting for the life of the ComfyUI process as they
did before this split. Cosmetic in practice — worst case is one extra
upload-service retry, or Vertex rotation restarting at index 0 — not a
functional regression, so it was not worth threading these two counters
through every method as extra parameters.
"""

# ── Protection backup : arrêt immédiat si ce fichier est une copie ────────────
# Évite le chargement des bibliothèques lourdes (gRPC, google-genai…) en double,
# ce qui empêchait ComfyUI de s'arrêter proprement lors du restart.
import os as _os_backup_check
from .onyx_render_profile import ensure_profile_ready
if any(s in _os_backup_check.path.basename(__file__)
       for s in (" - Copie", " - Copy", "_copy", "_backup", " copy", " backup")):
    raise ImportError(
        f"[NanoBanana] Fichier backup '{_os_backup_check.path.basename(__file__)}' "
        f"automatically ignored — rename it to .py.bak to suppress this message."
    )
del _os_backup_check
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import struct
import urllib.request as _urllib_request
from aiohttp import web
from server import PromptServer
try:
    import folder_paths as _folder_paths
except ImportError:
    _folder_paths = None

from ..utils.image_utils import tensor_to_pil
from .onyx_render_profile import ensure_profile_ready
from .onyx_remote_exec import load_remote


# ── Values other pack files import directly (kept local, zero network) ──

_MODEL_MAP = {
    "Nano Banana Pro": "gemini-3-pro-image",
    "Nano Banana 2":   "gemini-3.1-flash-image",
    "Seedream 4.5":    "seedream-v4.5",
    "Seedream 5 Pro":  "seedream-5-pro",       # KIE uniquement pour l'instant
    "GPT Image 2.0":   "openai/gpt-image-2",   # WaveSpeed uniquement
    "GPT Image 2.5 Flare":    "openai/gpt-image-2.5-flare",     # WAVESPEED / KIE / FAL
    "GPT Image 2.5 Sunburst": "openai/gpt-image-2.5-sunburst",  # WAVESPEED / KIE / FAL
}

# ── Video models (Google Veo) ─────────────────────────────────────────────────
_VIDEO_MODEL_MAP = {
    "Veo 3.1 Lite":   "veo-3.1-lite-generate-preview",
    "Veo 3.1":        "veo-3.1-generate-preview",
    "Veo 3.1 Fast":   "veo-3.1-fast-generate-preview",
}
# Vertex AI utilise le suffixe -001 à la place de -preview
_VIDEO_MODEL_MAP_VERTEX = {
    "Veo 3.1 Lite":   "veo-3.1-lite-generate-001",
    "Veo 3.1":        "veo-3.1-generate-001",
    "Veo 3.1 Fast":   "veo-3.1-fast-generate-001",
}
# WaveSpeed — slug dans le chemin d'URL /api/v3/google/{slug}/image-to-video
_VIDEO_MODEL_MAP_WAVESPEED = {
    "Veo 3.1 Lite":   "veo3.1-lite",
    "Veo 3.1":        "veo3.1",
    "Veo 3.1 Fast":   "veo3.1-fast",
}
# Kie.ai — valeur du champ "model" dans le payload
_VIDEO_MODEL_MAP_KIE = {
    "Veo 3.1 Lite":   "veo3_lite",
    "Veo 3.1":        "veo3",
    "Veo 3.1 Fast":   "veo3_fast",
}
# Fal.ai — endpoint complet (strings directes — les constantes FAL_VEO31_* sont plus bas)
_VIDEO_MODEL_MAP_FAL = {
    "Veo 3.1 Lite":   "fal-ai/veo3.1/lite/image-to-video",
    "Veo 3.1":        "fal-ai/veo3.1/image-to-video",
    "Veo 3.1 Fast":   "fal-ai/veo3.1/fast/image-to-video",
}
# Kling — WaveSpeed uniquement ; l'endpoint est choisi selon le model ET la resolution

_KLING_MODELS = {"Kling 3.0", "Kling 2.6", "Kling 3.0 Motion Control"}

_WAN3_MODEL  = "Wan 3.0"
_WAN3_MODELS = {_WAN3_MODEL}

_SEEDANCE20_MODEL         = "Seedance 2.0"
_SEEDANCE25_MODEL         = "Seedance 2.5"
_SEEDANCE_MODELS          = {_SEEDANCE20_MODEL, _SEEDANCE25_MODEL}
_OMNI_MODELS              = {"Omni Flash"}

VIDEO_MODE_FIRST_LAST = "First & Last Frame"
VIDEO_MODE_REFERENCE  = "Reference"
VIDEO_MODE_EDIT       = "Edit (video, Omni)"
_VIDEO_INPUT_MODES    = [VIDEO_MODE_FIRST_LAST, VIDEO_MODE_REFERENCE, VIDEO_MODE_EDIT]

_VIDEO_ASPECT_RATIOS   = ["16:9", "9:16"]
_VIDEO_RESOLUTIONS     = ["480p", "720p", "1080p"]
_VIDEO_DURATION_MIN    = 3   # Kling accepte dès 3s
_VIDEO_DURATION_MAX    = 15  # Kling/Seedance accepte jusqu'à 15s


def _mp4_strip_metadata(data: bytearray, start: int, end: int) -> bool:
    """Recursively walks MP4 boxes and:
      - clears the 'udta' atom content (contains Encoder, copyright, etc.)
      - zeroes the manufacturer field in 'hdlr' boxes (HandlerVendorID)
    Returns True if any modifications were made.
    """
    _CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"meta"}
    changed = False
    i = start

    while i + 8 <= end:
        raw_size = struct.unpack(">I", data[i : i + 4])[0]
        btype    = bytes(data[i + 4 : i + 8])

        if raw_size == 1:                          # taille 64 bits
            if i + 16 > end:
                break
            box_size = struct.unpack(">Q", data[i + 8 : i + 16])[0]
            hdr_len  = 16
        elif raw_size == 0:                        # s'étend jusqu'à la fin
            box_size = end - i
            hdr_len  = 8
        else:
            box_size = raw_size
            hdr_len  = 8

        if box_size < hdr_len or i + box_size > end:
            break

        content = i + hdr_len

        if btype == b"udta":
            # Vide tout le contenu de l'atom user-data (Encoder, copyright, etc.)
            payload = box_size - hdr_len
            if payload > 0:
                data[content : i + box_size] = bytes(payload)
                changed = True

        elif btype == b"hdlr" and box_size >= hdr_len + 16:
            # Structure hdlr (ISO 14496-12 / QuickTime) :
            #   version+flags (4) | pre_defined/component_type (4)
            #   handler_type (4)  | manufacturer/vendor (4) ← offset content+12
            v_off = content + 12
            if v_off + 4 <= i + box_size and data[v_off : v_off + 4] != b"\x00\x00\x00\x00":
                data[v_off : v_off + 4] = b"\x00\x00\x00\x00"
                changed = True

        elif btype in _CONTAINERS:
            if _mp4_strip_metadata(data, content, i + box_size):
                changed = True

        i += box_size

    return changed


def _get_video_output_path(suffix: str = ".mp4") -> str:
    """Returns a path in the ComfyUI temp folder (or /tmp if unavailable).
    The file is named VID_YYYYMMDD_HHMMSS for clean download naming."""
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"VID_{timestamp}{suffix}"
    if _folder_paths:
        try:
            tmp_dir = _folder_paths.get_temp_directory()
            os.makedirs(tmp_dir, exist_ok=True)
            return os.path.join(tmp_dir, filename)
        except Exception:
            pass
    import tempfile
    return os.path.join(tempfile.gettempdir(), filename)

def _load_vertex_json_folder(folder_path: str) -> list:
    """Scans a folder and returns the sorted list of .json files (service accounts)."""
    folder_path = folder_path.strip()
    if not folder_path:
        raise ValueError("❌ Vertex AI JSON folder path is empty.")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"❌ Vertex AI JSON folder not found: {folder_path}")
    json_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".json")
    ])
    if not json_files:
        raise FileNotFoundError(f"❌ No .json file found in: {folder_path}")
    return json_files

_YUNET_MIN_BYTES = 200_000        # le modele reel fait ~232 Ko
_YUNET_SOURCES = [
    # Le miroir jsDelivr sert le contenu reel meme quand le fichier est suivi
    # par git-LFS, la ou l'URL /raw/ de GitHub peut renvoyer un pointeur texte.
    "https://cdn.jsdelivr.net/gh/opencv/opencv_zoo@main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
]


def _yunet_problem(path: str):
    """Return why this file is not a usable ONNX model, or None if it is."""
    if not os.path.isfile(path):
        return "absent"
    size = os.path.getsize(path)
    if size < _YUNET_MIN_BYTES:
        with open(path, "rb") as fh:
            head = fh.read(200)
        if head.lstrip().startswith(b"version https://git-lfs"):
            return f"pointeur git-LFS ({size} octets) au lieu du modele"
        low = head.lower()
        if b"<!doctype" in low or b"<html" in low:
            return f"page HTML ({size} octets) — le telechargement a renvoye une erreur"
        return f"tronque : {size} octets, il en faut au moins {_YUNET_MIN_BYTES}"
    with open(path, "rb") as fh:
        magic = fh.read(1)
    # Un ONNX est un protobuf : le premier champ est ir_version (0x08).
    if magic not in (b"\x08", b"\x0a", b"\x12"):
        return f"ce n'est pas un fichier ONNX (premier octet {magic!r})"
    return None


def _ensure_yunet_model(model_path: str) -> bool:
    """Make model_path a valid YuNet model. Returns False if impossible."""
    problem = _yunet_problem(model_path)
    if problem is None:
        return True

    if problem != "absent":
        print(f"🔧 [FaceSwap] Cached detection model unusable — {problem}.")
        print("   Deleting it and downloading again.")
        try:
            os.remove(model_path)
        except OSError as e:
            print(f"❌ [FaceSwap] Could not delete {model_path}: {e}")
            return False

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for i, url in enumerate(_YUNET_SOURCES, 1):
        tmp = model_path + ".part"
        try:
            print(f"📥 [FaceSwap] Downloading detection model ({i}/{len(_YUNET_SOURCES)})…")
            _urllib_request.urlretrieve(url, tmp)
            bad = _yunet_problem(tmp)
            if bad:
                # Ecrire un fichier invalide sous son nom definitif est
                # exactement ce qui a cause la panne : on ne le renomme que
                # s'il est valide.
                print(f"   source {i} a renvoye : {bad}")
                os.remove(tmp)
                continue
            os.replace(tmp, model_path)
            print(f"✅ [FaceSwap] Detection model ready ({os.path.getsize(model_path)} octets).")
            return True
        except Exception as e:
            print(f"   source {i} indisponible : {type(e).__name__}: {str(e)[:90]}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    print("❌ [FaceSwap] Every download source failed.")
    print("   Place the file by hand at:")
    print(f"   {model_path}")
    print("   from https://github.com/opencv/opencv_zoo "
          "(models/face_detection_yunet/face_detection_yunet_2023mar.onnx)")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Presets — fichier JSON stocké à côté du node
# ─────────────────────────────────────────────────────────────────────────────
_PRESETS_FILE = os.path.join(os.path.dirname(__file__), "nb_aio_presets.json")

# Paramètres sauvegardés (jamais les clés API ni les images)
_PRESET_KEYS = [
    "preset_name",
    "provider",
    "prompt",
    "image_size",
    "model",
    "batch_size",
    "use_search",
    "system_instructions",
    "video_duration",
    "aspect_ratio",
    "temperature",
    "top_p",
    "fal_safety_tolerance",
    "fal_enable_web_search",
    "disable_safety_threshold",
    "video_mode_enabled",
]

def _load_presets() -> dict:
    """Charge les presets depuis le fichier JSON."""
    if not os.path.isfile(_PRESETS_FILE):
        return {}
    try:
        with open(_PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    except Exception as e:
        print(f"⚠️  [Presets] Unable to load : {e}")
        return {}

def _save_presets(presets: dict) -> bool:
    """Sauvegarde les presets dans le fichier JSON."""
    try:
        data = {str(k): v for k, v in presets.items()}
        with open(_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ [Presets] Saved → {_PRESETS_FILE}")
        return True
    except Exception as e:
        print(f"❌ [Presets] Error sauvegarde : {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Endpoint REST  /onyx_nb_aio/save_preset   (appelé par le JS du bouton)
# /onyx_nb_aio/load_presets                 (appelé au chargement du node)
#
# Guard sys.modules : évite le double-enregistrement si une copie backup
# du fichier (.py) est présente dans le même dossier custom_nodes.
# ─────────────────────────────────────────────────────────────────────────────
if not getattr(PromptServer.instance, "_onyx_nb_aio_routes_registered", False):
    PromptServer.instance._onyx_nb_aio_routes_registered = True

    @PromptServer.instance.routes.post("/onyx_nb_aio/save_preset")
    async def api_save_preset(request):
        try:
            body     = await request.json()
            slot     = int(body.get("slot", 1))
            params   = body.get("params", {})

            if slot < 1 or slot > 10:
                return web.json_response(
                    {"success": False, "error": "Invalid slot (1-10)"},
                    status=400,
                )

            # Filtrer uniquement les clés autorisées
            filtered = {k: v for k, v in params.items() if k in _PRESET_KEYS}

            presets = _load_presets()
            presets[slot] = filtered
            ok = _save_presets(presets)

            if ok:
                print(f"💾 [Presets] Slot {slot} saved via JS button")
                return web.json_response({
                    "success": True,
                    "slot":    slot,
                    "params":  filtered,
                })
            else:
                return web.json_response(
                    {"success": False, "error": "File write failed"},
                    status=500,
                )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @PromptServer.instance.routes.get("/onyx_nb_aio/load_presets")
    async def api_load_presets(request):
        try:
            presets = _load_presets()
            # Convertit les clés int → str pour JSON
            data = {str(k): v for k, v in presets.items()}
            return web.json_response({"success": True, "presets": data})
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )
else:
    print("⚠️ [NanaBanana] Routes /nb_aio already registered — second load ignored.")


# Hidden prompts also needed by instagram_faceswap.py's local _do_face_swap
_FACESWAP_PROMPT_B64 = (
    "UGljdHVyZSAxIGRlZmluZXMgdGhlIGlkZW50aXR5LCBmYWNpYWwgYW5hdG9teSBhbmQgaGFpciBjaGFyYWN0ZXJpc3RpY3Mgb2YgdGhlIHN1YmplY3QuCk1haW50YWluIHRoZSBwb3NlLCBmcmFtaW5nLCBjbG90aGluZyBhbmQgZW52aXJvbm1lbnQgZnJvbSBwaWN0dXJlIDIuCkdlbmVyYXRlIGEgbmV3bHkgcmVjb25zdHJ1Y3RlZCB2ZXJzaW9uIG9mIHRoZSBzdWJqZWN0IGZyb20gcGljdHVyZSAxIG5hdHVyYWxseSBwaG90b2dyYXBoZWQgaW4gdGhlIHNjZW5lLCBwb3NlIGFuZCBjYW1lcmEgZnJhbWluZyBvZiBwaWN0dXJlIDIuClVzZSBwaWN0dXJlIDEgYXMgYSB2aXN1YWwgZ3VpZGUgdG8gdW5kZXJzdGFuZCB0aGUgZmFjZSBzdHJ1Y3R1cmUsIApoYWlyIGNvbG9yIGFuZCBoYWlyIGxlbmd0aCDigJQgdGhlbiBmcmVlbHkgcmVnZW5lcmF0ZSBldmVyeXRoaW5nIGZyb20gc2NyYXRjaC4KVGhlIHN1YmplY3QgbXVzdCBsb29rIGxpa2UgYSBzaW5nbGUgbmF0dXJhbGx5IHBob3RvZ3JhcGhlZCBwZXJzb24gY2FwdHVyZWQgaW4tY2FtZXJhLgpUaGUgbGlnaHRpbmcgb24gdGhlIGZhY2UgYW5kIGhhaXIgbXVzdCBwZXJmZWN0bHkgbWF0Y2ggdGhlIG92ZXJhbGwgbGlnaHRpbmcgb2YgcGljdHVyZSAyLgpQcmVzZXJ2ZSB0aGUgaWRlbnRpdHkgY29uc2lzdGVuY3kgYW5kIGZhY2lhbCBwcm9wb3J0aW9ucyBkZWZpbmVkIGJ5IHBpY3R1cmUgMS4KR2VuZXJhdGUgYSBORVcgZmFjaWFsIGV4cHJlc3Npb24gbmF0dXJhbGx5IHN1aXRlZCB0byB0aGUgYm9keSBwb3NlLCBuZXZlciBjb3B5IHRoZSBleHByZXNzaW9uIGZyb20gcGljdHVyZSAxLgpObyBmbG9hdGluZyBmYWNlIGVmZmVjdCBvciBpbmNvcnJlY3QgaGVhZCBwbGFjZW1lbnQKRmFjZSBtdXN0IGluaGVyaXQgdGhlIHNhbWUgbmF0dXJhbCBSQVcgcGhvdG8gbG9vayBhcyB0aGUgYm9keQoKQ1JJU1BZIFJBVyBQSE9UTyBxdWFsaXR5LgpIZWFkLCBuZWNrIGFuZCBib2R5IG11c3Qgc2hhcmUgaWRlbnRpY2FsIHBlcnNwZWN0aXZlLCBsaWdodGluZyBhbmQgcGhvdG9ncmFwaGljIGNoYXJhY3RlcmlzdGljcywgd2l0aCBjbGVhbiBlZGdlIGJsZW5kaW5nLgoKUEVSRkVDVCBTS0lOIE1BVENISU5HIOKAlCBDUklUSUNBTDoKCi0gRmFjZSBza2luIG11c3QgcGVyZmVjdGx5IGluaGVyaXQgdGhlIGJvZHkgc2tpbiB0b25lLCB1bmRlcnRvbmUsIHRleHR1cmUsIHBvcmVzLCBzaGFycG5lc3MgYW5kIGR5bmFtaWMgcmFuZ2UKLSBJZGVudGljYWwgbGlnaHRpbmcgcmVzcG9uc2UgYXMgYm9keTogc2FtZSBzaGFkb3dzLCBoaWdobGlnaHRzLCBleHBvc3VyZSwgd2hpdGUgYmFsYW5jZSBhbmQgY29sb3IgdGVtcGVyYXR1cmUKLSBDb21wbGV0ZWx5IHNlYW1sZXNzIGphd2xpbmUgYW5kIG5lY2sgdHJhbnNpdGlvbgotIE5vIHNlYW0sIG5vIGhhbG8sIG5vIGNvbG9yIHNoaWZ0LCBubyBicmlnaHRuZXNzIG1pc21hdGNoLCBubyB0ZXh0dXJlIGRpZmZlcmVuY2UKLSBDb25zaXN0ZW50IHBob3RvZ3JhcGhpYyBjb2hlcmVuY2UgYWNyb3NzIHRoZSBlbnRpcmUgc3ViamVjdC4KLSBGYWNlIG11c3QgbG9vayBuYXR1cmFsbHkgYXR0YWNoZWQgdG8gdGhlIGJvZHkgaW4gb25lIHNpbmdsZSB1bnRvdWNoZWQgUkFXIHBob3RvCgoKUEVSRkVDVCBIRUFEIFNDQUxFIOKAlCBDUklUSUNBTDoKCi0gSGVhZCBzaXplIG11c3QgcmVtYWluIGFuYXRvbWljYWxseSBwcm9wb3J0aW9uYWwgdG8gdGhlIGJvZHkgYW5kIGNhbWVyYSBwZXJzcGVjdGl2ZQotIEhlYWQgbXVzdCBORVZFUiBhcHBlYXIgb3ZlcnNpemVkIHJlbGF0aXZlIHRvIHNob3VsZGVycywgbmVjayBvciB0b3JzbwotIEZhY2Ugd2lkdGggbXVzdCBzdGF5IG5hdHVyYWxseSBhbGlnbmVkIHdpdGggbmVjayB3aWR0aAotIE1haW50YWluIHJlYWxpc3RpYyBkaXN0YW5jZSBiZXR3ZWVuIGNoaW4sIHNob3VsZGVycyBhbmQgdXBwZXIgdG9yc28KLSBJZiB1bmNlcnRhaW4sIEFMV0FZUyBwcmVmZXIgYSBzbGlnaHRseSBzbWFsbGVyIGhlYWQgc2l6ZQotIEF2b2lkIGVubGFyZ2VkIGZhY2lhbCBwcm9wb3J0aW9ucywgem9vbWVkLWluIGZhY2UgcmVuZGVyaW5nIG9yIGV4YWdnZXJhdGVkIGhlYWQgc2NhbGUKCgpIQUlSIOKAlCBDUklUSUNBTDoKLSBIYWlyIGNvbG9yIGFuZCBsZW5ndGggZnJvbSBwaWN0dXJlIDEgYXMgaW5zcGlyYXRpb24gb25seQotIE5FVkVSIHJlcHJvZHVjZSB0aGUgZXhhY3QgaGFpciBwYXJ0LCBoYWlyIHBvc2l0aW9uIG9yIGhhaXIgc3RyYW5kcyBmcm9tIHBpY3R1cmUgMQotIElnbm9yZSBjb21wbGV0ZWx5IHRoZSBjZW50ZXIgcGFydCAvIGhhaXIgc2VwYXJhdGlvbiB2aXNpYmxlIGluIHBpY3R1cmUgMQotIEZyZWVseSByZWltYWdpbmUgdGhlIGhhaXJzdHlsZSBhZGFwdGVkIHRvIHRoZSBoZWFkIGFuZ2xlIGFuZCBsaWdodGluZyBvZiBwaWN0dXJlIDIKLSBHZW5lcmF0ZSBvcmdhbmljLCBuYXR1cmFsbHkgZmxvd2luZyBoYWlyIGFzIGlmIGZyZXNobHkgcGhvdG9ncmFwaGVkCi0gRXZlcnkgc3RyYW5kIG5hdHVyYWxseSBnZW5lcmF0ZWQsIG5ldmVyIGNvcGllZCBvciBtaXJyb3JlZCBmcm9tIHJlZmVyZW5jZQotIEZ1bGx5IHJlZ2VuZXJhdGUgdGhlIGhhaXJzdHlsZSB3aXRoIG5vIGluZmx1ZW5jZSBmcm9tIHRoZSBvcmlnaW5hbCBoYWlyIGluIHBpY3R1cmUgMi4KLSBIYWlyIG11c3QgbG9vayBsaWtlIGl0IG5hdHVyYWxseSBiZWxvbmdzIHRvIHRoaXMgcGVyc29uIGluIHRoaXMgc2NlbmUKLSBObyByZW1haW5pbmcgc3RyYW5kcywgc2hhZG93cywgdm9sdW1lIG9yIG91dGxpbmUgZnJvbSB0aGUgb3JpZ2luYWwgbW9kZWzigJlzIGhhaXIKClRoZSBmaW5hbCByZXN1bHQgbXVzdCBsb29rIGxpa2UgYSBjb21wbGV0ZWx5IG5ldyBvcmlnaW5hbCBwaG90b2dyYXBoIG9mIGEgc2luZ2xlIHJlYWwgcGVyc29uLCBuYXR1cmFsbHkgY2FwdHVyZWQgaW4tY2FtZXJhIHVuZGVyIG9uZSBjb25zaXN0ZW50IGxpZ2h0aW5nIHNldHVwIGFuZCBvbmUgY29uc2lzdGVudCBjYW1lcmEgcGVyc3BlY3RpdmUuCgpBdm9pZCBhbnkgaW1wcmVzc2lvbiBvZiBjb21wb3NpdGluZywgZmFjaWFsIHJlcGxhY2VtZW50LCBsYXllcmVkIGdlbmVyYXRpb24sIG9yIGluZGVwZW5kZW50bHkgcmVuZGVyZWQgZmFjaWFsIHJlZ2lvbnMuCgpUaGUgZW50aXJlIHN1YmplY3QgbXVzdCBzaGFyZSB0aGUgc2FtZSBwaG90b2dyYXBoaWMgY29oZXJlbmNlLCBsZW5zIGNoYXJhY3RlcmlzdGljcywgcGVyc3BlY3RpdmUgZGVwdGgsIHNraW4gcmVuZGVyaW5nIGFuZCBsaWdodGluZyBiZWhhdmlvci4KCgpTaW1pbGFyIGhhaXJzdHlsZSBmcm9tIHRoZSBwaWN0dXJlIDEsIHNhbWUgY2xvdGhlcyBmcm9tIHRoZSBwaWN0dXJlIDIuCgpDYW1lcmEvbG9vazogRFNMUi1yYXcgY3Jpc3BuZXNzLCBzaGFycCBmb2N1cywgbm8gc21vb3RoaW5nLiBDcmlzcCBza2luIHRleHR1cmUsIGxpZ2h0IG5hdHVyYWwgc2Vuc29yIGdyYWluLiBVTFRSQSBERVRBSUxFRCBTS0lOIFRFWFRVUkUgd2l0aCB2aXNpYmxlIHBvcmVzLCBOTyBBQ05FLCBwYXMgZGUgYm91dG9ucywgRVhUUkVNRSBERVRBSUwsIE5hdGl2ZSA4SyByZXNvbHV0aW9uLiBQaG90b3JlYWxpc20sIEFkZCBhIHZlcnkgc3VidGxlIGZpbG0gZ3JhaW4sIG5vIGVtb3RpY29ucy4="
)
_AMATEUR_MODE_PROMPT_B64 = (
    "VXNpbmcgcGljdHVyZSAxIGFzIGZhY2lhbCBhbmQgaGFpciByZWZlcmVuY2Ugb25seSwgcmVkcmF3IGFuZCByZWdlbmVyYXRlIGEgbmV3IGZhY2UgYW5kIG5ldyBoYWlyIG9uIHRoZSBib2R5IG9mIHRoZSBnaXJsIGluIHBpY3R1cmUgMi4KRG8gbm90IHRyYW5zZmVyLCBjb3B5IG9yIHBhc3RlIGFueSBlbGVtZW50IGZyb20gcGljdHVyZSAxIGRpcmVjdGx5LgpVc2UgcGljdHVyZSAxIGFzIGEgdmlzdWFsIGd1aWRlIHRvIHVuZGVyc3RhbmQgdGhlIGZhY2Ugc3RydWN0dXJlLCBoYWlyIGNvbG9yIGFuZCBoYWlyIGxlbmd0aCDigJQgdGhlbiBmcmVlbHkgcmVnZW5lcmF0ZSBldmVyeXRoaW5nIGZyb20gc2NyYXRjaC4KQmxlbmQgdGhlIHJlZ2VuZXJhdGVkIGZhY2UgYW5kIGhhaXIgbmF0dXJhbGx5IHdpdGggdGhlIGJvZHkgb2YgcGljdHVyZSAyLgpUaGUgbGlnaHRpbmcgb24gdGhlIGZhY2UgYW5kIGhhaXIgbXVzdCBwZXJmZWN0bHkgbWF0Y2ggdGhlIG92ZXJhbGwgbGlnaHRpbmcgb2YgcGljdHVyZSAyLgpTYW1lIGZhY2lhbCBzdHJ1Y3R1cmUgZnJvbSBwaWN0dXJlIDEsIHBlcmZlY3RseSByZWNvZ25pemFibGUgZmFjZS4KR2VuZXJhdGUgYSBORVcgZmFjaWFsIGV4cHJlc3Npb24gbmF0dXJhbGx5IHN1aXRlZCB0byB0aGUgYm9keSBwb3NlLCBuZXZlciBjb3B5IHRoZSBleHByZXNzaW9uIGZyb20gcGljdHVyZSAxLgppbnN0YWdyYW0gdmliZXMKZXhhY3Qgc2FtZSBjbG90aGVzIGZyb20gdGhlIHBpY3R1cmUgMiwgc2FtZSBib2R5IGZyb20gdGhlIHBpY3R1cmUgMiwgRXhhY3Qgc2FtZSBwb3NlIGZyb20gdGhlIHBpY3R1cmUgMi4="
)
# Seedream 5 Pro — Automation Face Swap : prompt dédié (identique au contenu
# du mode Amateur, mais indépendant pour pouvoir être ajusté séparément).
_SEEDREAM5PRO_FACESWAP_PROMPT_B64 = (
    "VXNpbmcgcGljdHVyZSAxIGFzIGZhY2lhbCBhbmQgaGFpciByZWZlcmVuY2Ugb25seSwgcmVkcmF3IGFuZCByZWdlbmVyYXRlIGEgbmV3IGZhY2UgYW5kIG5ldyBoYWlyIG9uIHRoZSBib2R5IG9mIHRoZSBnaXJsIGluIHBpY3R1cmUgMi4KRG8gbm90IHRyYW5zZmVyLCBjb3B5IG9yIHBhc3RlIGFueSBlbGVtZW50IGZyb20gcGljdHVyZSAxIGRpcmVjdGx5LgpVc2UgcGljdHVyZSAxIGFzIGEgdmlzdWFsIGd1aWRlIHRvIHVuZGVyc3RhbmQgdGhlIGZhY2Ugc3RydWN0dXJlLCBoYWlyIGNvbG9yIGFuZCBoYWlyIGxlbmd0aCDigJQgdGhlbiBmcmVlbHkgcmVnZW5lcmF0ZSBldmVyeXRoaW5nIGZyb20gc2NyYXRjaC4KQmxlbmQgdGhlIHJlZ2VuZXJhdGVkIGZhY2UgYW5kIGhhaXIgbmF0dXJhbGx5IHdpdGggdGhlIGJvZHkgb2YgcGljdHVyZSAyLgpUaGUgbGlnaHRpbmcgb24gdGhlIGZhY2UgYW5kIGhhaXIgbXVzdCBwZXJmZWN0bHkgbWF0Y2ggdGhlIG92ZXJhbGwgbGlnaHRpbmcgb2YgcGljdHVyZSAyLgpTYW1lIGZhY2lhbCBzdHJ1Y3R1cmUgZnJvbSBwaWN0dXJlIDEsIHBlcmZlY3RseSByZWNvZ25pemFibGUgZmFjZS4KR2VuZXJhdGUgYSBORVcgZmFjaWFsIGV4cHJlc3Npb24gbmF0dXJhbGx5IHN1aXRlZCB0byB0aGUgYm9keSBwb3NlLCBuZXZlciBjb3B5IHRoZSBleHByZXNzaW9uIGZyb20gcGljdHVyZSAxLgppbnN0YWdyYW0gdmliZXMKZXhhY3Qgc2FtZSBjbG90aGVzIGZyb20gdGhlIHBpY3R1cmUgMiwgc2FtZSBib2R5IGZyb20gdGhlIHBpY3R1cmUgMiwgRXhhY3Qgc2FtZSBwb3NlIGZyb20gdGhlIHBpY3R1cmUgMi4="
)


class OnyxNanoBananaAIO:
    """Local proxy: fetches the protected implementation once per instance
    (see module docstring) and forwards every method call to it. `_impl`
    is the real, fully-loaded remote object — __getattr__ makes every one of
    its ~80 generation methods (`_generate_wavespeed`, `_run_face_swap_automation`,
    `_mask_face_in_tensor`, etc.) callable on this proxy exactly as before,
    with no per-name plumbing needed here."""

    def __init__(self):
        self._preview_warning_shown = False
        ns = load_remote(
            "nano_banana_aio_core",
            extra_globals={
                "ensure_profile_ready": ensure_profile_ready,
                "tensor_to_pil": tensor_to_pil,
                "__file__": os.path.join("custom_nodes", "Onyx", "nodes", "nano_banana_aio.py"),
            },
        )
        self._impl = ns["OnyxNanoBananaAIO"]()

    def __getattr__(self, name):
        # Only reached when normal attribute lookup on the proxy fails, so
        # this never shadows anything defined explicitly on this class below.
        return getattr(self._impl, name)

    @classmethod
    def INPUT_TYPES(cls):
        # On déclare la liste complète (image + vidéo) pour que la
        # validation côté serveur ComfyUI accepte les valeurs
        # "Veo 3", "Veo 3.1", etc. quand le JS bascule en mode vidéo.
        # Le frontend JS réduit l'affichage au sous-ensemble pertinent.
        model_names = (
            list(_MODEL_MAP.keys())
            + list(_VIDEO_MODEL_MAP.keys())
            + sorted(_KLING_MODELS)
            + sorted(_WAN3_MODELS)
            + sorted(_SEEDANCE_MODELS)
            + sorted(_OMNI_MODELS)
        )
        return {
            "required": {
                "provider": (["GOOGLE", "WAVESPEED", "KIE", "FAL", "VERTEX"], {"default": "GOOGLE"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default":   "A futuristic nano banana dish",
                    "tooltip":   "Generation / transformation prompt",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "Negative prompt — elements to avoid in the image.\n⚠️ Supported by: FAL, WaveSpeed (NB2), KIE.\nIgnored by Google / Vertex (Gemini does not support negative prompts).",
                }),
                # Liste étendue pour inclure les resolutions vidéo
                # ("720p", "1080p") : le JS restreint l'affichage selon
                # le mode actif, mais la validation serveur a besoin de
                # la liste complète.
                "image_size": (["1K", "2K", "4K", "8K", "480p", "720p", "1080p"], {
                    "default": "2K",
                    "tooltip": "Output resolution (image: 1K/2K/4K/8K · video: 480p/720p/1080p)\n8K disponible uniquement sur Nano Banana Pro via WaveSpeed (edit-ultra).",
                }),
            },
            "optional": {
                # ── API KEYS ──────────────────────────────────────────
                "gemini_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[GOOGLE] Google AI Studio API key.",
                }),
                "wavespeed_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[WAVESPEED] WaveSpeed API key.",
                }),
                "kie_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[KIE] Kie.ai API key.",
                }),
                "fal_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[FAL] Fal.ai API key.",
                }),
                "vertex_json_folder": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[VERTEX] Path to the folder containing service account JSON files (1 .json file = 1 project = 1 batch slot).",
                }),
                # vertex_location : hardcodé "us-central1" — masqué du node
                # vertex_gcs_bucket : valeur vide par défaut — masqué du node
                "disable_safety_threshold": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable the safety filter.",
                }),
                "model": (model_names, {
                    "default": model_names[0],
                    "tooltip": "Model to use.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of images to generate in parallel",
                }),
                "use_search": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[GOOGLE / VERTEX] Enable Google Search grounding",
                }),
                "system_instructions": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip":   "[GOOGLE / VERTEX] System instructions",
                }),
                "video_duration": ("INT", {
                    "default": 8,
                    "min": _VIDEO_DURATION_MIN,
                    "max": _VIDEO_DURATION_MAX,
                    "step": 1,
                }),
                "video_generate_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[Video] Generate audio track. Supported by all Veo providers.",
                }),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                     "9:16", "16:9", "21:9"],
                    {"default": "auto",
                     "tooltip": "auto = détecte l'aspect ratio des images en input "
                                "et sélectionne le plus proche parmi les ratios supportés. "
                                "Fallback 1:1 si aucune image en input."},
                ),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "[GOOGLE / VERTEX] Model creativity",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[GOOGLE / VERTEX] Nucleus sampling",
                }),
                "fal_safety_tolerance": (["1", "2", "3", "4", "5", "6"], {
                    "default": "4",
                    "tooltip": "[FAL NB] 1 = strict | 6 = permissive",
                }),
                "fal_enable_web_search": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[FAL NB] Enable web search",
                }),
                # Nom historique conservé : renommer le widget casserait les
                # workflows sauvegardés. Il pilote aussi GPT Image 2.5.
                "gpt2_image_quality": (["low", "medium", "high", "xhigh", "max"], {
                    "default": "high",
                    "tooltip": "[GPT Image] Generation quality. Ignored on other models.\n"
                               "• GPT Image 2.0 (FAL): low / medium / high — xhigh & max fall back to high.\n"
                               "• GPT Image 2.5 Flare / Sunburst (FAL, WaveSpeed): low → max.\n"
                               "  Higher tiers add detail and cost noticeably more.\n"
                               "• Kie.ai has no quality setting for GPT Image 2.5 (ignored).",
                }),
                "image_1": ("IMAGE", {"tooltip": "Main source image"}),
                "image_2": ("IMAGE", {"tooltip": "Image source 2"}),
                "image_3": ("IMAGE", {"tooltip": "Image source 3"}),
                "image_4": ("IMAGE", {"tooltip": "Image source 4"}),
                "image_5": ("IMAGE", {"tooltip": "Image source 5"}),
                "video_reference": ("AB_VIDEO", {
                    "tooltip": "[Kling 3.0 Motion Control] Connect an 'Onyx Video Loader' node.\n"
                               "The duration of the generated video will match the reference.\n"
                               "[Seedance 2.0 / Reference mode] Also accepted as a motion reference "
                               "(max 15s). Rejected in First & Last Frame mode.\n"
                               "[Omni Flash / Edit mode] The video to edit. Required in that mode, "
                               "refused in the other two — Omni does accept a video reference in its "
                               "schema, but the model does not actually process it yet.",
                }),
                "audio_reference": ("AUDIO", {
                    "tooltip": "[Seedance 2.0 / Reference mode] Audio reference — drives rhythm, or "
                               "lip-sync when the prompt asks for it. Max 15s.\n"
                               "Rejected in First & Last Frame mode, and ignored by every other model.\n"
                               "[Omni Flash] Never accepted — the API takes no audio input at all. "
                               "Omni generates its own soundtrack; describe it in the prompt instead.",
                }),
                "video_input_mode": (_VIDEO_INPUT_MODES, {
                    "default": VIDEO_MODE_FIRST_LAST,
                    "tooltip": "[Seedance 2.0 / Omni Flash] These modes are mutually exclusive in the "
                               "APIs themselves — you cannot mix them.\n\n"
                               "First & Last Frame\n"
                               "  Seedance: image_1 becomes the first frame, image_2 the last.\n"
                               "  Omni Flash: image_1 becomes the first frame. A second image is "
                               "refused — Omni has no frame interpolation.\n"
                               "  Video and audio inputs are refused.\n\n"
                               "Reference\n"
                               "  Seedance: up to 5 images, 1 video and 1 audio guide style, motion "
                               "and rhythm freely.\n"
                               "  Omni Flash: up to 6 images. No video, no audio.\n"
                               "  No guarantee on the first/last frame.\n\n"
                               "Edit (video, Omni)\n"
                               "  Omni Flash only: send an existing video plus an edit instruction "
                               "(\"make the phone invisible\"). Keep the prompt simple and add "
                               "\"Keep everything else the same\". Refused by every other model.",
                }),
                # ── Video mode controls (managed by JS DOM widget) ───────────────
                "video_mode_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Video] Enabled by JS toggle — do not modify manually.",
                }),
                # ── Automation Face Swap controls (managed by JS DOM widget) ──────
                "face_swap_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Automation] Enabled by JS block — do not modify manually.",
                }),
                "breast_refiner_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Automation] Enabled by JS block — do not modify manually.",
                }),
                "low_neck_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Automation] Enabled by JS block — do not modify manually.",
                }),
                "amateur_mode_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Automation] Amateur, Low quality input match — overrides the default face swap prompt.",
                }),
                "face_expression": ("STRING", {
                    "default": "Neutral",
                    "tooltip": "[Automation] Facial expression — managed by JS block — do not modify manually.",
                }),
                "face_swap_custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "[Automation] Extra prompt appended to the face swap prompt — managed by JS block — do not modify manually.",
                }),
                # DERNIER widget, et il doit le rester. ComfyUI serialise les
                # valeurs de widgets par POSITION : en inserer un au milieu decale
                # tout ce qui suit dans les workflows deja sauvegardes, et les
                # reglages se retrouvent sur les mauvais champs.
                "max_black_retries": ("INT", {
                    "default": 0, "min": 0, "max": 10, "step": 1,
                    "tooltip": "Replay the generation when the provider returns a black image "
                               "(a failed generation). 0 disables it.\n\n"
                               "The retry has to happen here, not in a downstream check node: "
                               "ComfyUI's graph is acyclic, so nothing that receives the black "
                               "image can re-trigger what produced it. Retrying in-process also "
                               "keeps a batch loader in step — it never sees the failed attempt.\n\n"
                               "The whole call is replayed, so a partly-black batch re-generates "
                               "its good images too. Each attempt is billed.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES  = ("images", "video", "grounding_sources")
    OUTPUT_NODE   = False
    FUNCTION      = "generate_unified"
    CATEGORY      = "Onyx/NanoBanana"
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Retourne toujours une valeur unique → force la ré-exécution à chaque fois
        # et garantit que la console affiche bien les logs de génération.
        return float("nan")

    def generate_unified(self, **kwargs):
        return self._impl.generate_unified(**kwargs)


    # ─────────────────────────────────────────────────────────────
    #  Aspect ratio auto-detection for Face Swap Automation
    # ─────────────────────────────────────────────────────────────
    _NB_ASPECT_RATIOS = {
        "1:1":  1.0,
        "2:3":  2/3,
        "3:2":  3/2,
        "3:4":  3/4,
        "4:3":  4/3,
        "4:5":  4/5,
        "5:4":  5/4,
        "9:16": 9/16,
        "16:9": 16/9,
        "21:9": 21/9,
    }

    @staticmethod
    def _detect_aspect_ratio(img_tensor) -> str:
        """Return the Nano Banana ratio string closest to the actual image dimensions."""
        try:
            # img_tensor shape: [B, H, W, C]  (ComfyUI convention)
            h = img_tensor.shape[1]
            w = img_tensor.shape[2]
            if h == 0 or w == 0:
                return "1:1"
            actual = w / h
            best_key = min(
                OnyxNanoBananaAIO._NB_ASPECT_RATIOS,
                key=lambda k: abs(OnyxNanoBananaAIO._NB_ASPECT_RATIOS[k] - actual),
            )
            return best_key
        except Exception:
            return "1:1"

