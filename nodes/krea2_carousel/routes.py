"""Private HTTP routes of the Onyx Krea2 Carousel module (port of NodoForgeLab2 Beta)."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import traceback

import requests
from aiohttp import web
from PIL import Image, ImageOps, UnidentifiedImageError
from server import PromptServer

from . import local_vlm
from .director import _local_endpoint, _read_state, generate_directed_prompt
from .pose_generator import generate_pose_list


MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
_GENERATION_LOCK = asyncio.Lock()


def _decode_image(value, label):
    if not value:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a base64 image.")
    if "," in value:
        prefix, value = value.split(",", 1)
        if prefix and not prefix.lower().startswith("data:image/"):
            raise ValueError(f"{label} has an invalid data URL.")
    if len(value) > ((MAX_IMAGE_BYTES * 4) // 3) + 8:
        raise ValueError(f"{label} exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MiB.")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{label} contains invalid base64.") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"{label} exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MiB.")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            width, height = opened.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"{label} has dimensions that are not allowed ({width}×{height}).")
            opened.load()
            return ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{label} is not a valid image.") from exc


def _ollama_models(endpoint):
    base = _local_endpoint(endpoint, "http://127.0.0.1:11434")
    url = base if base.endswith("/api/tags") else base + "/api/tags"
    try:
        response = requests.get(url, timeout=15, allow_redirects=False)
    except requests.RequestException as exc:
        raise ValueError(f"Could not connect to Ollama at {base}.") from exc
    if response.status_code != 200:
        raise ValueError(f"Ollama answered HTTP {response.status_code}: {response.text[:300]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("Ollama returned invalid JSON.") from exc
    models = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model") or item.get("name") or "").strip()
        if model_id:
            models.append({"id": model_id})
    return models


def _grok_models(api_key):
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("The Grok API key is missing, so the available models cannot be listed.")
    try:
        response = requests.get(
            "https://api.x.ai/v1/models",
            headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
            timeout=20,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise ValueError("Could not fetch the Grok/xAI model list.") from exc
    if response.status_code != 200:
        raise ValueError(f"Grok models HTTP {response.status_code}: {response.text[:500]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("Grok returned an invalid model list.") from exc
    excluded = ("imagine", "image", "video", "voice", "audio", "tts", "transcribe")
    models = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        lowered = model_id.lower()
        if not lowered.startswith("grok-") or any(token in lowered for token in excluded):
            continue
        models.append(
            {
                "id": model_id,
                "context_length": item.get("context_length"),
                "owned_by": item.get("owned_by") or "xai",
            }
        )
    if not models:
        raise ValueError("The account returned no Grok models compatible with Chat Completions.")
    return models


routes = PromptServer.instance.routes

if not getattr(PromptServer.instance, "_onyx_nf_routes_registered", False):
    PromptServer.instance._onyx_nf_routes_registered = True

    @routes.get("/onyx/nf/local/status")
    async def local_status(_request):
        return web.json_response(dict(local_vlm.STATUS, loaded=local_vlm.is_loaded()))

    @routes.get("/onyx/nf/local/models")
    async def local_models(_request):
        try:
            models = await asyncio.to_thread(local_vlm.list_cached_models)
            return web.json_response(
                {"models": models},
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @routes.post("/onyx/nf/local/unload")
    async def local_unload(_request):
        await asyncio.to_thread(local_vlm.unload)
        return web.json_response({"ok": True})

    @routes.post("/onyx/nf/ollama/models")
    async def ollama_models(request):
        try:
            data = await request.json()
            models = await asyncio.to_thread(_ollama_models, data.get("host"))
            return web.json_response({"models": models})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/onyx/nf/runtime_key")
    async def runtime_key(request):
        try:
            data = await request.json()
            from .pose_generator import RUNTIME_KEYS
            backend = str(data.get("backend") or "")
            if backend in ("Grok API", "Gemini API"):
                value = str(data.get("api_key") or "").strip()[:500]
                if value:
                    RUNTIME_KEYS[backend] = value
                else:
                    RUNTIME_KEYS.pop(backend, None)
            return web.json_response({"ok": True}, headers={"Cache-Control": "no-store"})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/onyx/nf/grok/models")
    async def grok_models(request):
        try:
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError("The request body must be a JSON object.")
            models = await asyncio.to_thread(_grok_models, data.get("api_key"))
            return web.json_response({"models": models}, headers={"Cache-Control": "no-store"})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/onyx/nf/director/generate")
    async def director_generate(request):
        try:
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError("The request body must be a JSON object.")
            state = _read_state(data.get("director_state") or "{}")
            pil_1 = _decode_image(data.get("image_1_b64"), "IMAGE 1")
            pil_2 = _decode_image(data.get("image_2_b64"), "IMAGE 2")
            pil_3 = _decode_image(data.get("image_3_b64"), "IMAGE 3")
            async with _GENERATION_LOCK:
                result = await asyncio.to_thread(
                    generate_directed_prompt,
                    state,
                    pil_1,
                    pil_2,
                    pil_3,
                )
            return web.json_response(result)
        except Exception as exc:
            traceback.print_exc()
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/onyx/nf/krea2/poses/generate")
    async def krea2_pose_generate(request):
        try:
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError("The request body must be a JSON object.")
            pil_image = _decode_image(data.get("image_b64"), "MASTER IMAGE")
            async with _GENERATION_LOCK:
                result = await asyncio.to_thread(generate_pose_list, data, pil_image)
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except Exception as exc:
            traceback.print_exc()
            return web.json_response({"error": str(exc)}, status=400)
