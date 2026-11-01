"""
ComfyUI custom nodes for Nano Banana Pro and Nano Banana 2 APIs.
- Nano Banana Pro Edit: edit with images (Gemini 3 Pro).
- Nano Banana 2 Edit: edit with images (Gemini 3.1 Flash).
Each node returns image as ComfyUI IMAGE tensor [B, H, W, C].
"""

import logging
import io
import base64
import random as _random

import torch
import numpy as np
from PIL import Image


from .nodes.onyx_render_profile import ensure_profile_ready
from .utils.onyx_http import post_json, get_json, download as _http_download, OnyxAPIError
from .utils.onyx_parallel import map_parallel
from .utils import onyx_api_cache as _cache
from .utils import onyx_cost as _cost

_SAFETY_THRESHOLDS = [
    "OFF",
    "BLOCK_NONE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE",
]

_HARM_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)

_ASPECT_RATIOS = [
    "auto", "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
]

_IMAGE_SIZES = ["1K", "2K", "4K"]

# Widget appended at the END of every node's optional inputs: ComfyUI saves
# widget values by position, so a widget inserted in the middle would shift
# every saved workflow's settings onto the wrong fields.
_CACHE_WIDGET = {"cache_mode": (_cache.CACHE_MODES, {"default": _cache.CACHE_OFF,
                                                     "tooltip": _cache.CACHE_TOOLTIP})}
_HIDDEN = {"unique_id": "UNIQUE_ID"}


def _run_with_cache_and_cost(node_name, namespace, params, cache_mode, unique_id, estimate, generate):
    """Shared wrapper: cache lookup (opt-in), cost badge + ledger, then generate().

    `generate()` must return the IMAGE tensor. The licence check has already
    run in the caller, so a cache hit is not a way around it."""
    _cost.notify_node(unique_id, estimate)
    key = _cache.make_key(namespace, params) if cache_mode == _cache.CACHE_REUSE else None
    hit = _cache.load(key)
    if hit is not None and hit["images"] is not None and not isinstance(hit["images"], list):
        print(f"♻️  [{node_name}] identical request already generated — reused from cache, not billed.")
        _cost.record(node_name, estimate, cache_hit=True)
        _cost.notify_node(unique_id, estimate, final=True, cache_hit=True)
        return (hit["images"],)
    images = generate()
    _cost.record(node_name, estimate)
    _cost.notify_node(unique_id, estimate, final=True)
    if key:
        _cache.store(key, images=images, info={"node": node_name})
    return (images,)


# ---------- Nano Banana Pro Edit ----------

class NanoBananaProEditAPINode:
    """Edit/Generate images via Gemini 3 Pro (gemimg.py reference implementation)."""

    MODEL_ID = "gemini-3-pro-image"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (_IMAGE_SIZES, {"default": "2K"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "safety_threshold": (_SAFETY_THRESHOLDS, {"default": "OFF"}),
                **_CACHE_WIDGET,
            },
            "hidden": _HIDDEN,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, images, api_key, prompt, aspect_ratio="auto", image_size="2K", temperature=1.0, seed=0,
             safety_threshold="OFF", cache_mode=_cache.CACHE_OFF, unique_id=None):
        ensure_profile_ready()
        if api_key.strip():
            NanoBananaProEditAPINode._cached_api_key = api_key.strip()
        api_key = api_key.strip() or NanoBananaProEditAPINode._cached_api_key
        if not api_key:
            raise RuntimeError("Nano Banana Pro: API key is required.")

        params = {"images": images, "prompt": prompt, "aspect_ratio": aspect_ratio,
                  "image_size": image_size, "temperature": temperature, "seed": seed,
                  "safety_threshold": safety_threshold, "model": self.MODEL_ID}
        est = _cost.estimate_simple("Nano Banana Pro", "GOOGLE", image_size, 1)
        return _run_with_cache_and_cost(
            "Nano Banana Pro", "NanoBananaProEdit", params, cache_mode, unique_id, est,
            lambda: self._generate(images, api_key, prompt, aspect_ratio, image_size,
                                   temperature, safety_threshold),
        )

    def _generate(self, images, api_key, prompt, aspect_ratio, image_size, temperature, safety_threshold):
        # Build parts: images first, then text (matching gemimg.py order)
        parts = _images_to_parts(images)
        parts.append({"text": prompt.strip()})

        # Auto aspect ratio detection
        if aspect_ratio == "auto":
            h, w = images.shape[1], images.shape[2]
            ratio = w / h
            ratios = {
                "1:1": 1.0, "2:3": 0.66, "3:2": 1.5, "3:4": 0.75, "4:3": 1.33,
                "4:5": 0.8, "5:4": 1.25, "9:16": 0.56, "16:9": 1.77, "21:9": 2.33,
            }
            aspect_ratio = min(ratios, key=lambda k: abs(ratios[k] - ratio))

        # Build payload matching gemimg.py structure exactly
        payload = {
            "generationConfig": {
                "temperature": temperature,
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                },
                "responseModalities": ["Image"],
            },
            "contents": [{"parts": parts}],
            "system_instruction": {
                "parts": [{
                    "text": "You are an expert image-generation engine. You must ALWAYS produce an image. "
                            "Interpret all user input—regardless of format, intent, or abstraction—as literal "
                            "visual directives for image composition. Prioritize generating the visual "
                            "representation above any text, formatting, or conversational requests."
                }]
            },
        }
        # The widget used to be ignored (always OFF); it is now applied.
        _apply_gemini_safety(payload, safety_threshold or "OFF")

        # Use x-goog-api-key header (matching gemimg.py)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        url = f"{self.BASE_URL}/{self.MODEL_ID}:generateContent"

        # Retries 429/5xx with backoff, never replays a read timeout (could bill twice).
        data = post_json(url, service="Nano Banana Pro", payload=payload, headers=headers,
                         timeout=(15, 180))

        # Check for API error
        if "error" in data:
            err = data["error"]
            raise RuntimeError(f"Nano Banana Pro API error: {err.get('code')} — {err.get('message')}")

        # Check for blocked content
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Nano Banana Pro: no candidates in response.")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        if finish_reason in ("PROHIBITED_CONTENT", "NO_IMAGE"):
            raise RuntimeError(f"Nano Banana Pro: image not generated due to {finish_reason}.")

        if "content" not in candidate:
            raise RuntimeError("Nano Banana Pro: no content in response.")

        # Extract inline image data
        response_parts = candidate["content"]["parts"]
        img_bytes = _largest_image_bytes(response_parts, "Nano Banana Pro")
        if img_bytes:
            return _image_bytes_to_tensor(img_bytes, "Nano Banana Pro")[0]

        raise RuntimeError("Nano Banana Pro: no image found in response parts.")


# ---------- Nano Banana 2 Edit ----------

class NanoBanana2EditAPINode:
    """Edit images via Gemini 3.1 Flash API."""

    MODEL_ID = "gemini-3.1-flash-image"
    BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (_IMAGE_SIZES, {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "safety_threshold": (_SAFETY_THRESHOLDS, {"default": "OFF"}),
                **_CACHE_WIDGET,
            },
            "hidden": _HIDDEN,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, images, api_key, prompt, aspect_ratio="auto", image_size="1K", temperature=1.0, seed=0,
             safety_threshold="OFF", cache_mode=_cache.CACHE_OFF, unique_id=None):
        ensure_profile_ready()
        if api_key.strip():
            NanoBanana2EditAPINode._cached_api_key = api_key.strip()
        api_key = api_key.strip() or NanoBanana2EditAPINode._cached_api_key
        if not api_key:
            raise RuntimeError("Nano Banana 2 Edit: API key is required.")
        if images.shape[0] == 0:
            raise RuntimeError("Nano Banana 2 Edit: at least one image is required.")

        params = {"images": images, "prompt": prompt, "aspect_ratio": aspect_ratio,
                  "image_size": image_size, "temperature": temperature, "seed": seed,
                  "safety_threshold": safety_threshold, "model": self.MODEL_ID}
        est = _cost.estimate_simple("Nano Banana 2", "GOOGLE", image_size or "1K", 1)
        return _run_with_cache_and_cost(
            "Nano Banana 2 Edit", "NanoBanana2Edit", params, cache_mode, unique_id, est,
            lambda: self._generate(images, api_key, prompt, aspect_ratio, image_size,
                                   temperature, seed, safety_threshold),
        )

    def _generate(self, images, api_key, prompt, aspect_ratio, image_size, temperature, seed, safety_threshold):
        parts = _images_to_parts(images)
        parts.append({"text": prompt})

        image_config = {}
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        if image_size:
            image_config["imageSize"] = image_size

        gen_config = {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": temperature,
            "seed": seed,
        }
        if image_config:
            gen_config["imageConfig"] = image_config

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "systemInstruction": {
                "parts": [{
                    "text": "You are an expert image-generation engine. You must ALWAYS produce an image. "
                            "Interpret all user input—regardless of format, intent, or abstraction—as literal "
                            "visual directives for image composition. Prioritize generating the visual "
                            "representation above any text, formatting, or conversational requests."
                }]
            },
            "generationConfig": gen_config,
        }
        # The widget used to be ignored (always OFF); it is now applied.
        _apply_gemini_safety(payload, safety_threshold or "OFF")

        # Key in a header, not in "?key=": requests copies the URL into its
        # exception messages, which put the key in the console and in screenshots.
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        data = post_json(self.BASE_URL, service="Nano Banana 2 Edit", payload=payload,
                         headers=headers, timeout=(15, 120))

        raw = _extract_image_from_gemini_response(data, "Nano Banana 2 Edit")
        if not raw:
            detail = _gemini_response_summary(data)
            raise RuntimeError(f"Nano Banana 2 Edit: no image in response. {detail}")
        return _image_bytes_to_tensor(raw, "Nano Banana 2 Edit")[0]


# ---------- Seedream 4.5 Edit (WaveSpeed) ----------

import time as _time

_SEEDREAM_SIZES = [
    "auto — match input (2K)",
    "1920*1920 — square (1:1)",
    "2560*1920 — landscape_4_3 (4:3)",
    "1920*2560 — portrait_4_3 (3:4)",
    "1536*1920 — portrait_4_5 (4:5)",
    "1920*1536 — landscape_5_4 (5:4)",
    "2560*1440 — landscape_16_9 (16:9)",
    "1440*2560 — portrait_16_9 (9:16)",
]


class SeedreamEditAPINode:
    """Edit images via ByteDance Seedream 4.5 Edit on WaveSpeed API.

    Supports 1-10 reference images for context-aware editing with
    facial feature, lighting, and color tone preservation up to 4K.
    """

    BASE_URL = "https://api.wavespeed.ai/api/v3/bytedance/seedream-v4.5/edit"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "wavespeed_apikey": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "size": (_SEEDREAM_SIZES, {"default": "auto — match input (2K)"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "use_custom_size": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                **_CACHE_WIDGET,
            },
            "hidden": _HIDDEN,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, images, wavespeed_apikey, prompt, size="auto — match input (2K)", num_images=1, use_custom_size=False,
             custom_width=2048, custom_height=2048, cache_mode=_cache.CACHE_OFF, unique_id=None):
        ensure_profile_ready()
        if wavespeed_apikey.strip():
            SeedreamEditAPINode._cached_api_key = wavespeed_apikey.strip()
        api_key = wavespeed_apikey.strip() or SeedreamEditAPINode._cached_api_key
        if not api_key:
            raise RuntimeError("Seedream 4.5 Edit: WaveSpeed API key is required.")
        if not prompt.strip():
            raise RuntimeError("Seedream 4.5 Edit: prompt is required.")
        if images.shape[0] == 0:
            raise RuntimeError("Seedream 4.5 Edit: at least one image is required.")

        params = {"images": images, "prompt": prompt, "size": size, "num_images": num_images,
                  "use_custom_size": use_custom_size, "custom_width": custom_width,
                  "custom_height": custom_height}
        est = _cost.estimate_simple("Seedream 4.5", "WAVESPEED", "*", num_images)
        return _run_with_cache_and_cost(
            "Seedream 4.5 Edit", "SeedreamEditWavespeed", params, cache_mode, unique_id, est,
            lambda: self._generate(images, api_key, prompt, size, num_images, use_custom_size,
                                   custom_width, custom_height),
        )

    def _generate(self, images, api_key, prompt, size, num_images, use_custom_size, custom_width, custom_height):
        # Resolve size
        if use_custom_size:
            api_size = f"{custom_width}*{custom_height}"
        elif size.startswith("auto"):
            # Use first image's exact aspect ratio, scale to meet minimum pixels
            h, w = images.shape[1], images.shape[2]
            min_pixels = 3686400
            # Scale up proportionally until we meet minimum pixels
            scale = max(1.0, (min_pixels / (w * h)) ** 0.5)
            new_w = int(round(w * scale / 64) * 64)
            new_h = int(round(h * scale / 64) * 64)
            # Clamp to API limits
            new_w = max(512, min(8192, new_w))
            new_h = max(512, min(8192, new_h))
            api_size = f"{new_w}*{new_h}"
            logging.info("Seedream auto size: input %dx%d -> output %s", w, h, api_size)
        else:
            api_size = size.split(" ")[0]  # e.g. "2048*2048" from "2048*2048 — square_hd (1:1)"

        # Convert images to data URIs (max 10)
        image_uris = _images_to_data_uris(images, 10)

        payload = {
            "images": image_uris,
            "prompt": prompt,
            "size": api_size,
            "num_images": num_images,
            "enable_sync_mode": True,
            "enable_base64_output": True,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        result = post_json(self.BASE_URL, service="Seedream 4.5 Edit", payload=payload,
                           headers=headers, timeout=(15, 180))

        data = result.get("data", {})

        # Async polling fallback if sync mode didn't complete immediately
        if data.get("status") not in ("completed",):
            poll_url = data.get("urls", {}).get("get", "")
            if not poll_url:
                raise RuntimeError("Seedream 4.5 Edit: no poll URL and task not completed.")
            for _ in range(90):
                _time.sleep(2)
                try:
                    # Polling is free and idempotent: get_json retries everything.
                    data = get_json(poll_url, service="Seedream 4.5 Edit poll", headers=headers,
                                    timeout=(10, 30), max_retries=2).get("data", {})
                except OnyxAPIError as e:
                    logging.warning("Seedream 4.5 Edit poll error: %s", e)
                    continue
                if data.get("status") == "completed":
                    break
                if data.get("status") == "failed":
                    raise RuntimeError(f"Seedream 4.5 Edit failed: {data.get('error', 'unknown error')}")
            else:
                raise RuntimeError("Seedream 4.5 Edit timed out after 3 minutes.")

        outputs = data.get("outputs", [])
        if not outputs:
            raise RuntimeError("Seedream 4.5 Edit: no images in API response.")

        def _decode_output(output):
            if output.startswith("data:"):
                raw = base64.b64decode(output.split(",", 1)[-1])
            elif output.startswith("http"):
                raw = _http_download(output, service="Seedream 4.5 Edit")
            else:
                raw = base64.b64decode(output)  # Assume raw base64
            return _image_bytes_to_tensor(raw, "Seedream 4.5 Edit")[0]

        # Up to 4 outputs: download/decode them in parallel, order preserved.
        tensors = map_parallel(_decode_output, outputs, max_workers=4, label="Seedream 4.5 Edit")
        if not tensors:
            raise RuntimeError("Seedream 4.5 Edit: no valid images decoded.")
        return _batch_tensors(tensors)


# ---------- Seedream 4.5 Edit (fal.ai) ----------

_FAL_SEEDREAM_SIZES = [
    "auto — match input (2K)",
    "1920*1920 — square (1:1)",
    "2560*1920 — landscape (4:3)",
    "1920*2560 — portrait (3:4)",
    "1536*1920 — portrait (4:5)",
    "1920*1536 — landscape (5:4)",
    "2560*1440 — landscape (16:9)",
    "1440*2560 — portrait (9:16)",
]

_FAL_SIZE_MAP = {
    "1920*1920": "square_hd",
    "2560*1920": "landscape_4_3",
    "1920*2560": "portrait_4_3",
    "1536*1920": {"width": 1536, "height": 1920},
    "1920*1536": {"width": 1920, "height": 1536},
    "2560*1440": "landscape_16_9",
    "1440*2560": "portrait_16_9",
}


class SeedreamEditFalAPINode:
    """Edit images via ByteDance Seedream 4.5 Edit on fal.ai.

    Supports 1-10 reference images for context-aware editing with
    facial feature, lighting, and color tone preservation up to 4K.
    """

    BASE_URL = "https://fal.run/fal-ai/bytedance/seedream/v4.5/edit"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fal_apikey": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "size": (_FAL_SEEDREAM_SIZES, {"default": "auto — match input (2K)"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "use_custom_size": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                **_CACHE_WIDGET,
            },
            "hidden": _HIDDEN,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, images, fal_apikey, prompt, size="auto — match input (2K)", num_images=1, use_custom_size=False,
             custom_width=2048, custom_height=2048, cache_mode=_cache.CACHE_OFF, unique_id=None):
        ensure_profile_ready()
        if fal_apikey.strip():
            SeedreamEditFalAPINode._cached_api_key = fal_apikey.strip()
        api_key = fal_apikey.strip() or SeedreamEditFalAPINode._cached_api_key
        if not api_key:
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): fal.ai API key is required.")
        if not prompt.strip():
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): prompt is required.")
        if images.shape[0] == 0:
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): at least one image is required.")

        # fal gets a random seed on every call, so the cache key has no seed:
        # "reuse identical request" deliberately freezes that randomness.
        params = {"images": images, "prompt": prompt, "size": size, "num_images": num_images,
                  "use_custom_size": use_custom_size, "custom_width": custom_width,
                  "custom_height": custom_height}
        est = _cost.estimate_simple("Seedream 4.5", "FAL", "*", num_images)
        return _run_with_cache_and_cost(
            "Seedream 4.5 Edit (fal.ai)", "SeedreamEditFal", params, cache_mode, unique_id, est,
            lambda: self._generate(images, api_key, prompt, size, num_images, use_custom_size,
                                   custom_width, custom_height),
        )

    def _generate(self, images, api_key, prompt, size, num_images, use_custom_size, custom_width, custom_height):
        # Resolve size
        if use_custom_size:
            fal_size = {"width": custom_width, "height": custom_height}
        elif size.startswith("auto"):
            # Use first image's exact aspect ratio, scale to ~2K
            h, w = images.shape[1], images.shape[2]
            target_pixels = 1920 * 1920  # 2K target
            current_pixels = w * h
            scale = (target_pixels / current_pixels) ** 0.5
            new_w = int(round(w * scale / 64) * 64)
            new_h = int(round(h * scale / 64) * 64)
            new_w = max(512, min(4096, new_w))
            new_h = max(512, min(4096, new_h))
            fal_size = {"width": new_w, "height": new_h}
            logging.info("Seedream (fal.ai) auto size: input %dx%d -> output %dx%d", w, h, new_w, new_h)
        else:
            dimension_str = size.split(" ")[0]
            fal_size = _FAL_SIZE_MAP.get(dimension_str, "square_hd")

        # Convert images to data URIs (max 10)
        image_urls = _images_to_data_uris(images, 10)

        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": fal_size,
            "num_images": num_images,
            "seed": _random.randint(0, 2147483647),
            "enable_safety_checker": False,
        }
        headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }

        result = post_json(self.BASE_URL, service="Seedream 4.5 Edit (fal.ai)", payload=payload,
                           headers=headers, timeout=(15, 180))

        fal_images = result.get("images", [])
        if not fal_images:
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): no images in API response.")

        urls = []
        for idx, fal_img in enumerate(fal_images):
            img_url = fal_img.get("url", "")
            if not img_url:
                logging.warning("Seedream 4.5 Edit (fal.ai): no URL for image %d, skipping.", idx)
                continue
            urls.append(img_url)

        def _fetch(url):
            raw = _http_download(url, service="Seedream 4.5 Edit (fal.ai)")
            return _image_bytes_to_tensor(raw, "Seedream 4.5 Edit (fal.ai)")[0]

        # Download the (up to 4) results in parallel instead of one after the other.
        tensors = map_parallel(_fetch, urls, max_workers=4, label="Seedream 4.5 Edit (fal.ai)")
        if not tensors:
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): no valid images downloaded.")
        return _batch_tensors(tensors)


# ---------- Helpers ----------

def _png_b64(img):
    img_np = (255.0 * img.cpu().numpy()).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_batch(images, limit=None):
    """PNG+base64 of each image of an IMAGE batch, in parallel (zlib releases
    the GIL, so 10 references at 2K encode several times faster)."""
    count = images.shape[0] if limit is None else min(images.shape[0], limit)
    return map_parallel(lambda i: _png_b64(images[i]), range(count),
                        max_workers=4, progress=False)


def _images_to_parts(images):
    """Convert ComfyUI IMAGE tensor [B, H, W, C] to list of inlineData parts."""
    return [{"inlineData": {"mimeType": "image/png", "data": b64}} for b64 in _encode_batch(images)]


def _images_to_data_uris(images, limit):
    return [f"data:image/png;base64,{b64}" for b64 in _encode_batch(images, limit)]


def _batch_tensors(tensors):
    """[1,H,W,C] tensors -> one batch. Mixed sizes are resized to the first one
    (torch.cat used to crash the node outright on them)."""
    first = tensors[0]
    same = [t if t.shape[1:3] == first.shape[1:3] else
            torch.nn.functional.interpolate(t.permute(0, 3, 1, 2), size=tuple(first.shape[1:3]),
                                            mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            for t in tensors]
    return torch.cat(same, dim=0)


def _apply_gemini_safety(payload, threshold):
    """Apply safety threshold to all harm categories."""
    payload["safetySettings"] = [
        {"category": cat, "threshold": threshold} for cat in _HARM_CATEGORIES
    ]


def _gemini_response_summary(data):
    """Return a human-readable summary of why a Gemini response contained no image."""
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "")
            if block_reason:
                return f"Prompt was blocked: {block_reason}"
            return "No candidates returned."
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        parts = candidate.get("content", {}).get("parts") or []
        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        if text_parts:
            snippet = " | ".join(text_parts)[:300]
            return f"Model returned text only (finishReason={finish_reason}): {snippet}"
        return f"No image part in response (finishReason={finish_reason})."
    except Exception:
        return ""



def _largest_image_bytes(parts, log_name):
    """Image to keep among Gemini response parts: ignore "thought" drafts and keep
    the LARGEST remaining image (Google can return a default 1K image next to the
    requested 2K/4K one, in no guaranteed order). All thoughts -> largest of them."""
    found = []  # (pixels, index, thought, bytes, "WxH")
    for idx, part in enumerate(parts or []):
        inline = part.get("inlineData") or part.get("inline_data")
        b64 = inline.get("data") if inline else None
        if not b64:
            continue
        try:
            raw = base64.b64decode(b64)
        except Exception:
            continue
        try:
            w, h = Image.open(io.BytesIO(raw)).size
        except Exception:
            w = h = 0
        found.append((w * h, idx, bool(part.get("thought")), raw, f"{w}x{h}"))
    if not found:
        return None
    pool = [f for f in found if not f[2]] or found
    best = max(pool, key=lambda f: (f[0], f[1]))
    if len(found) > 1:
        desc = ", ".join(f"{f[4]} {len(f[3]) // 1024} KB" + (" (thought)" if f[2] else "") for f in found)
        print(f"🖼️  [{log_name}] {len(found)} images in the answer: {desc} -> kept {best[4]}")
    return best[3]

def _extract_image_from_gemini_response(data, log_name):
    """Extract the image to keep (largest non-thought) from a Gemini generateContent response."""
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            logging.warning("%s: response has no candidates. Full response: %s", log_name, data)
            return None
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        parts = candidate.get("content", {}).get("parts") or []
        logging.info("%s: finishReason=%s, part count=%d", log_name, finish_reason, len(parts))
        img = _largest_image_bytes(parts, log_name)
        if img:
            return img
        text_parts = []
        for part in parts:
            if part.get("thought"):
                continue
            if "text" in part:
                text_parts.append(part["text"])
        if text_parts:
            logging.warning(
                "%s: API returned text instead of image (finishReason=%s). Text: %s",
                log_name, finish_reason, " | ".join(text_parts)[:500],
            )
        else:
            logging.warning(
                "%s: no image or text in response. finishReason=%s. Full response: %s",
                log_name, finish_reason, str(data)[:1000],
            )
    except (KeyError, TypeError, ValueError) as e:
        logging.warning("%s parse response: %s — raw: %s", log_name, e, str(data)[:500])
    return None


def _image_bytes_to_tensor(raw, log_name):
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        logging.error("%s: failed to decode image: %s", log_name, e)
        raise RuntimeError(f"Failed to decode image from API: {e}") from e
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...]
    return (tensor,)


# ---------- ComfyUI registration ----------

NODE_CLASS_MAPPINGS = {
    "NanoBananaProEditAPINode": NanoBananaProEditAPINode,
    "NanoBanana2EditAPINode": NanoBanana2EditAPINode,
    "SeedreamEditAPINode": SeedreamEditAPINode,
    "SeedreamEditFalAPINode": SeedreamEditFalAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaProEditAPINode": "Onyx Nano Banana Pro Edit",
    "NanoBanana2EditAPINode": "Onyx Nano Banana 2 Edit",
    "SeedreamEditAPINode": "Onyx Seedream 4.5 Edit(Wavespeed)",
    "SeedreamEditFalAPINode": "Onyx Seedream 4.5 Edit(fal.ai)",
}
