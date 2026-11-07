# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""
Onyx API media — what the Onyx.AI site sends to a workflow run on its Comfy API deployment:

  Onyx LoRA from URL   the member's own character LoRA (stored by the site, short-lived signed link):
                       downloaded once per machine (cached by `cache_id`), applied to MODEL / CLIP.
  Onyx Image from URL  a reference picture (Onyx Studio "recreate"), as an IMAGE.

Only https links, size capped, a small on-disk cache (the 3 most recent LoRAs). These nodes are not
gated: they only load files; the gated Onyx nodes downstream still need the session.
"""
import hashlib
import io
import os
import re
import threading
import time

import requests

_MAX_LORA = 3 * 1024 ** 3          # 3 GB
_MAX_IMAGE = 40 * 1024 ** 2         # 40 MB
_KEEP_LORAS = 3
_lock = threading.Lock()


def _cache_dir():
    try:
        import folder_paths
        base = folder_paths.get_temp_directory()
    except Exception:  # noqa: BLE001
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
    path = os.path.join(base, "onyx_api_loras")
    os.makedirs(path, exist_ok=True)
    return path


def _check_url(url):
    url = (url or "").strip()
    if not url.lower().startswith("https://"):
        raise RuntimeError("Onyx API: an https link is needed.")
    return url


def _download(url, dest, cap):
    tmp = dest + ".part"
    got = 0
    with requests.get(url, stream=True, timeout=(15, 120)) as r:
        if r.status_code != 200:
            raise RuntimeError("Onyx API: the file could not be downloaded (HTTP %d)." % r.status_code)
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(4 * 1024 * 1024):
                got += len(chunk)
                if got > cap:
                    fh.close()
                    os.remove(tmp)
                    raise RuntimeError("Onyx API: the file is too large.")
                fh.write(chunk)
    os.replace(tmp, dest)


def _prune(folder, keep):
    files = sorted((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".safetensors")),
                   key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        try:
            os.remove(f)
        except OSError:
            pass


class OnyxLoraFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "url": ("STRING", {"default": "", "tooltip": "https link to a .safetensors LoRA."}),
                "cache_id": ("STRING", {"default": "", "tooltip": "Same id = same file: downloaded once."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "run"
    CATEGORY = "Onyx"
    DESCRIPTION = "Loads a LoRA from an https link (Onyx.AI site runs) and applies it to MODEL / CLIP."

    def run(self, model, clip, url="", cache_id="", strength_model=1.0, strength_clip=1.0):
        import comfy.sd
        import comfy.utils
        if not (url or "").strip():
            return (model, clip)                       # no LoRA: the workflow runs as is
        url = _check_url(url)
        name = re.sub(r"[^A-Za-z0-9_-]", "", cache_id or "")[:64] or hashlib.sha256(
            url.split("?", 1)[0].encode()).hexdigest()[:32]
        folder = _cache_dir()
        path = os.path.join(folder, name + ".safetensors")
        with _lock:
            if not os.path.isfile(path):
                t0 = time.time()
                _download(url, path, _MAX_LORA)
                print("[Onyx API] LoRA downloaded (%.0f MB, %.1f s)" % (os.path.getsize(path) / 1048576.0,
                                                                         time.time() - t0))
            else:
                os.utime(path, None)
            _prune(folder, _KEEP_LORAS)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, float(strength_model), float(strength_clip))
        return (model, clip)


class OnyxImageFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"url": ("STRING", {"default": "", "tooltip": "https link to a picture."})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Onyx"
    DESCRIPTION = "Loads a picture from an https link (Onyx.AI site runs)."

    def run(self, url=""):
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        url = _check_url(url)
        r = requests.get(url, timeout=(15, 60))
        if r.status_code != 200:
            raise RuntimeError("Onyx API: the picture could not be downloaded (HTTP %d)." % r.status_code)
        if len(r.content) > _MAX_IMAGE:
            raise RuntimeError("Onyx API: the picture is too large.")
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        return (torch.from_numpy(arr)[None, ...],)


NODE_CLASS_MAPPINGS = {"OnyxLoraFromUrl": OnyxLoraFromUrl, "OnyxImageFromUrl": OnyxImageFromUrl}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxLoraFromUrl": "Onyx LoRA from URL", "OnyxImageFromUrl": "Onyx Image from URL"}
