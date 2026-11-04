# -*- coding: utf-8 -*-
"""
Onyx LLM — one call, five back-ends, for prompt writing.

    text = chat(provider, model, system=..., user=..., images=[PIL.Image, ...],
                api_key=..., temperature=0.7, max_tokens=2048)

Providers:
  ollama       local Ollama server (default http://127.0.0.1:11434). Vision
               models (qwen2.5vl, qwen3-vl, gemma3, llava...) accept images.
  local_vlm    the pack's own Qwen3-VL loaded with transformers (the Krea2
               Carousel's local VLM): no server, no key.
  xai          xAI Grok (api.x.ai), OpenAI-style chat completions.
  openrouter   OpenRouter (any model id, e.g. x-ai/grok-4.1-fast).
  gemini       Google AI Studio generateContent.
  vertex       Gemini on Vertex AI with service-account JSON files: every JSON in the folder is one
               Google project; calls rotate over them and a key that fails (quota, permission) passes
               the call to the next one. The "key" of this provider is the folder path.

Each request is sent once (no automatic retry, same rule as utils/onyx_http).
Errors come back as OnyxAPIError with the provider's message and the key redacted.
"""

from __future__ import annotations

import base64
import io
import re

from .onyx_http import OnyxAPIError, get_json, post_json
from . import onyx_registry as _reg

PROVIDERS = ["ollama", "local_vlm", "xai", "openrouter", "gemini", "vertex"]
PROVIDER_LABELS = {
    "ollama": "Ollama (local)",
    "local_vlm": "Qwen3-VL local (transformers)",
    "xai": "Grok (xAI API)",
    "openrouter": "OpenRouter API",
    "gemini": "Gemini API",
    "vertex": "Gemini on Vertex (JSON keys)",
}
DEFAULT_VERTEX_FOLDER = r"C:\Comfyui\json"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
NEEDS_KEY = {"xai", "openrouter", "gemini", "vertex"}

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def _png_b64(img, max_side: int = 1536) -> str:
    """PIL image -> base64 PNG, longest side capped (vision APIs bill per pixel)."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ollama_models(url: str = DEFAULT_OLLAMA_URL) -> list:
    data = get_json(url.rstrip("/") + "/api/tags", service="Ollama", timeout=(3, 10))
    return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))


def chat(provider: str, model: str, *, system: str = "", user: str = "", images=(),
         api_key: str = "", temperature: float = 0.7, max_tokens: int = 2048,
         seed: int | None = None, ollama_url: str = DEFAULT_OLLAMA_URL, timeout: int = 240) -> str:
    images = [i for i in (images or []) if i is not None]
    if provider in NEEDS_KEY and provider != "vertex" and not (api_key or "").strip():
        raise OnyxAPIError(provider, f"{PROVIDER_LABELS.get(provider, provider)}: no API key. "
                                     f"Save one in Onyx Studio (Settings tab) or connect it.")
    if provider == "ollama":
        text = _ollama(model, system, user, images, temperature, max_tokens, seed, ollama_url, timeout)
    elif provider == "local_vlm":
        text = _local_vlm(model, system, user, images, temperature, max_tokens, seed)
    elif provider in ("xai", "openrouter"):
        text = _openai_style(provider, model, system, user, images, api_key, temperature, max_tokens,
                             seed, timeout)
    elif provider == "gemini":
        text = _gemini(model, system, user, images, api_key, temperature, max_tokens, seed, timeout)
    elif provider == "vertex":
        text = _vertex(model, system, user, images, api_key, temperature, max_tokens, seed, timeout)
    else:
        raise OnyxAPIError(provider, f"Unknown provider '{provider}'.")
    return _THINK.sub("", text or "").strip()


# ─────────────────────────────────────────────────────────────────────────────

def _ollama(model, system, user, images, temperature, max_tokens, seed, url, timeout):
    if not model:
        raise OnyxAPIError("Ollama", "Ollama: pick a model (Settings tab lists the ones installed).")
    msg = {"role": "user", "content": user}
    if images:
        msg["images"] = [_png_b64(i) for i in images]
    messages = ([{"role": "system", "content": system}] if system else []) + [msg]
    options = {"temperature": temperature, "num_predict": max_tokens}
    if seed:
        options["seed"] = int(seed) % (2 ** 31)
    data = post_json(url.rstrip("/") + "/api/chat", service="Ollama",
                     payload={"model": model, "messages": messages, "stream": False, "options": options},
                     timeout=(5, timeout))
    return (data.get("message") or {}).get("content", "")


def _local_vlm(model, system, user, images, temperature, max_tokens, seed):
    try:
        from ..nodes.krea2_carousel import local_vlm
    except Exception as e:
        raise OnyxAPIError("Local VLM", f"Local Qwen3-VL unavailable ({e}). Install transformers>=4.57 "
                                        f"and accelerate, or use Ollama.") from None
    return local_vlm.generate_text(system, user, pil_images=list(images), model_id=model or None,
                                   temperature=temperature, max_new_tokens=max_tokens,
                                   seed=int(seed or 0), unload_after=True)


def _openai_style(provider, model, system, user, images, api_key, temperature, max_tokens, seed, timeout):
    url = ("https://api.x.ai/v1/chat/completions" if provider == "xai"
           else "https://openrouter.ai/api/v1/chat/completions")
    content = [{"type": "text", "text": user}]
    for img in images:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _png_b64(img)}})
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": content if images else user}]
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if seed:
        payload["seed"] = int(seed) % (2 ** 31)
    data = post_json(url, service=PROVIDER_LABELS[provider], payload=payload,
                     headers={"Authorization": f"Bearer {api_key.strip()}"}, timeout=(15, timeout))
    choices = data.get("choices") or []
    if not choices:
        raise OnyxAPIError(provider, f"{PROVIDER_LABELS[provider]}: empty answer.")
    return (choices[0].get("message") or {}).get("content") or ""


def _gemini(model, system, user, images, api_key, temperature, max_tokens, seed, timeout):
    parts = [{"text": user}]
    for img in images:
        parts.append({"inlineData": {"mimeType": "image/png", "data": _png_b64(img)}})
    gen = {"maxOutputTokens": max_tokens}
    if model not in _reg.no_sampling_models("gemini_text"):
        gen["temperature"] = temperature
    if seed:
        gen["seed"] = int(seed) % (2 ** 31)
    payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = post_json(url, service="Gemini", payload=payload,
                     headers={"x-goog-api-key": api_key.strip()}, timeout=(15, timeout))
    cands = data.get("candidates") or []
    if not cands:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise OnyxAPIError("Gemini", f"Gemini: no answer ({reason}).")
    texts = [p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", [])
             if "text" in p and not p.get("thought")]
    return "".join(texts)


# ── Vertex (several service-account JSONs) ───────────────────────────────────
import threading as _threading

_VX_LOCK = _threading.Lock()
_VX_CLIENTS: dict = {}
_VX_CURSOR = [0]


def vertex_keys(folder: str = "") -> list:
    """Service-account JSON files of the folder (or a single JSON path), sorted."""
    import glob
    import json as _json
    import os
    folder = (folder or "").strip().strip('"') or DEFAULT_VERTEX_FOLDER
    files = [folder] if folder.lower().endswith(".json") else sorted(glob.glob(os.path.join(folder, "*.json")))
    out = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = _json.load(fh)
            if d.get("type") == "service_account" and d.get("project_id") and d.get("private_key"):
                out.append((f, d["project_id"]))
        except (OSError, ValueError):
            continue
    return out


def _vx_client(path: str, project: str, timeout: int):
    with _VX_LOCK:
        c = _VX_CLIENTS.get(path)
    if c is not None:
        return c
    from google import genai
    from google.genai import types
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(
        path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    c = genai.Client(vertexai=True, project=project, location="global", credentials=creds,
                     http_options=types.HttpOptions(timeout=int(timeout * 1000)))
    with _VX_LOCK:
        _VX_CLIENTS[path] = c
    return c


def _vertex(model, system, user, images, folder, temperature, max_tokens, seed, timeout):
    try:
        from google.genai import types
    except ImportError:
        raise OnyxAPIError("Vertex", "Vertex: the google-genai package is missing "
                                     "(python_embeded\\python.exe -m pip install google-genai).") from None
    keys = vertex_keys(folder)
    if not keys:
        raise OnyxAPIError("Vertex", f"Vertex: no service-account JSON in {(folder or '').strip() or DEFAULT_VERTEX_FOLDER} "
                                     f"(Onyx Studio > Settings).")
    parts = [types.Part.from_text(text=user)]
    for img in images:
        parts.append(types.Part.from_bytes(data=base64.b64decode(_png_b64(img)), mime_type="image/png"))
    cfg = {"max_output_tokens": max_tokens}
    if model not in _reg.no_sampling_models("gemini_text"):
        cfg["temperature"] = temperature
    if seed:
        cfg["seed"] = int(seed) % (2 ** 31)
    if system:
        cfg["system_instruction"] = system
    config = types.GenerateContentConfig(**cfg)
    with _VX_LOCK:
        start = _VX_CURSOR[0] % len(keys)
        _VX_CURSOR[0] += 1
    errors = []
    import time as _time
    deadline = _time.time() + max(60, int(timeout))          # all keys together, not each
    for k in range(len(keys)):
        if _time.time() > deadline:
            errors.append("time budget used up")
            break
        path, project = keys[(start + k) % len(keys)]
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        try:
            resp = _vx_client(path, project, min(int(timeout), 150)).models.generate_content(
                model=model, contents=[types.Content(role="user", parts=parts)], config=config)
        except Exception as e:  # noqa: BLE001 - quota / permission / outage: the next key takes the call
            code = getattr(e, "code", None)
            errors.append(f"{name}: {str(e)[:160]}")
            if code == 400:            # a bad request is the same for every key
                break
            continue
        texts = []
        for cand in (getattr(resp, "candidates", None) or [])[:1]:
            for part in (getattr(getattr(cand, "content", None), "parts", None) or []):
                if getattr(part, "text", None) and not getattr(part, "thought", False):
                    texts.append(part.text)
        if texts:
            return "".join(texts)
        block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        raise OnyxAPIError("Vertex", f"Vertex: no answer ({block or 'empty'}).")
    raise OnyxAPIError("Vertex", "Vertex: every key failed - " + " | ".join(errors[:4]))


# ─────────────────────────────────────────────────────────────────────────────
# Answer parsing (skills answer "chat style": a code block + a NOTES line)
# ─────────────────────────────────────────────────────────────────────────────

_FENCE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.S)
_NOTES = re.compile(r"^\s*NOTES:\s*(.+?)\s*$", re.M | re.I)


def parse_answer(text: str) -> dict:
    """-> {"prompt", "notes", "ratio", "post"}. Works with a fenced answer, a raw
    answer, and drops a stray READY line."""
    text = _THINK.sub("", text or "").strip()
    notes_m = _NOTES.search(text)
    notes = notes_m.group(1).strip() if notes_m else ""
    fence = _FENCE.search(text)
    if fence:
        prompt = fence.group(1).strip()
    else:
        prompt = _NOTES.sub("", text).strip()
        prompt = re.sub(r"^\s*READY\b.*$", "", prompt, flags=re.M).strip()
    ratio = re.search(r"ratio\s*[:=]?\s*(\d+\s*:\s*\d+)", notes, re.I)
    post = re.search(r"post\s*[:=]\s*([a-z_]+)", notes, re.I)
    return {"prompt": prompt, "notes": notes,
            "ratio": ratio.group(1).replace(" ", "") if ratio else "",
            "post": (post.group(1).lower() if post else "")}
