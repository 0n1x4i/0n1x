# -*- coding: utf-8 -*-
"""
Onyx API cache — never pay twice for the exact same request.

Typical case: a 12-node workflow fails at node 10, you fix it and re-queue.
ComfyUI re-runs the API nodes too (they declare IS_CHANGED = NaN on purpose,
so that re-queueing the same prompt gives a NEW image). With the cache turned
on, an identical request is answered from disk in milliseconds, for free.

It is OPT-IN per node (`cache_mode` widget, default off): for most of these
nodes, re-running the same prompt *is* how people get variations, and a cache
on by default would silently stop that.

Key = sha256 of: a namespace (node), every parameter that changes the result
(prompt, model, size, seed, ...) and the PIXELS of every input image. API keys
and other credentials are excluded (and never written to disk). Any parameter
of a type it cannot fingerprint reliably makes the key unique, so a doubtful
case is a miss, never a wrong hit.

Storage: <pack>/_cache/api_cache/<key>/ — images as lossless PNG (providers
return 8-bit images, so nothing is lost), text outputs in meta.json, a video
output copied next to them. Least-recently-used entries are evicted beyond
ONYX_API_CACHE_MAX_GB (default 3 GB).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_PACK_ROOT, "_cache", "api_cache")

CACHE_OFF = "off"
CACHE_REUSE = "reuse identical request"
CACHE_MODES = [CACHE_OFF, CACHE_REUSE]
CACHE_TOOLTIP = (
    "off (default): every run calls the provider and is billed — re-running the same "
    "prompt gives a new variation.\n"
    "reuse identical request: if this exact request (same prompt, settings and input "
    "pixels) was already generated, the saved result is returned instantly and nothing "
    "is billed. Change anything — even one word or the seed — and it generates again.\n"
    "Results are kept in the pack's _cache/api_cache folder (oldest removed past 3 GB)."
)

# Parameter names that never enter the key (credentials, UI-only state).
SKIP_KEYS = frozenset({
    "api_key", "gemini_api_key", "wavespeed_api_key", "kie_api_key", "fal_api_key",
    "atlascloud_api_key", "grok_api_key", "openrouter_api_key", "wavespeed_apikey",
    "fal_apikey", "vertex_json_folder", "cache_mode", "unique_id", "prompt_id",
    "extra_pnginfo", "max_black_retries",
})

_lock = threading.Lock()


class _Unhashable(Exception):
    pass


def _max_bytes() -> int:
    try:
        gb = float(os.environ.get("ONYX_API_CACHE_MAX_GB", "3"))
    except ValueError:
        gb = 3.0
    return int(gb * 1024 ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprinting
# ─────────────────────────────────────────────────────────────────────────────

def _feed(h, value, depth=0):
    if depth > 8:
        raise _Unhashable("too deep")
    if value is None or isinstance(value, (bool, int, float, str)):
        h.update(b"J" + json.dumps(value, ensure_ascii=False).encode("utf-8"))
        if isinstance(value, str) and len(value) < 1024 and os.path.isfile(value):
            _feed_file(h, value)       # a path input: the CONTENT matters, not the name
        return
    if isinstance(value, bytes):
        h.update(b"B" + hashlib.sha256(value).digest())
        return
    if isinstance(value, dict):
        h.update(b"D")
        for k in sorted(value, key=str):
            h.update(str(k).encode("utf-8") + b"=")
            _feed(h, value[k], depth + 1)
        return
    if isinstance(value, (list, tuple)):
        h.update(b"L%d" % len(value))
        for v in value:
            _feed(h, v, depth + 1)
        return
    if hasattr(value, "shape") and hasattr(value, "cpu"):      # torch tensor
        t = value.detach().cpu().contiguous()
        h.update(b"T" + str(tuple(t.shape)).encode() + str(t.dtype).encode())
        h.update(t.numpy().tobytes())
        return
    if hasattr(value, "shape") and hasattr(value, "tobytes"):  # numpy array
        h.update(b"N" + str(value.shape).encode() + str(value.dtype).encode())
        h.update(value.tobytes())
        return
    # Objects with a stable path attribute (video wrappers, etc.)
    for attr in ("path", "file", "filename", "video_path"):
        p = getattr(value, attr, None)
        if isinstance(p, str) and os.path.isfile(p):
            h.update(b"P" + type(value).__name__.encode())
            _feed_file(h, p)
            return
    raise _Unhashable(type(value).__name__)


def _feed_file(h, path):
    """Size + head + tail of the file: catches any re-encode or trim without
    reading a 2 GB video entirely."""
    st = os.stat(path)
    h.update(b"F%d" % st.st_size)
    chunk = 4 * 1024 * 1024
    with open(path, "rb") as fh:
        h.update(fh.read(chunk))
        if st.st_size > 2 * chunk:
            fh.seek(-chunk, os.SEEK_END)
            h.update(fh.read(chunk))


def make_key(namespace: str, params: dict) -> str | None:
    """sha256 key of a request, or None when some input cannot be fingerprinted
    (the caller then simply does not use the cache for this run)."""
    h = hashlib.sha256(b"onyx-api-cache-v1|" + namespace.encode("utf-8"))
    try:
        for k in sorted(params):
            if k in SKIP_KEYS:
                continue
            h.update(b"|" + k.encode("utf-8") + b"=")
            _feed(h, params[k])
    except _Unhashable as e:
        print(f"[Onyx Cache] {namespace}: input '{e}' cannot be fingerprinted — cache skipped this run.")
        return None
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Load / store
# ─────────────────────────────────────────────────────────────────────────────

def load(key: str | None):
    """Return {"images": tensor|None, "video": path, "texts": [...], "meta": {...}} or None."""
    if not key:
        return None
    d = os.path.join(CACHE_DIR, key)
    meta_path = os.path.join(d, "meta.json")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    images = None
    if meta.get("image_count"):
        import numpy as np
        import torch
        from PIL import Image
        frames = []
        for i in range(meta["image_count"]):
            p = os.path.join(d, f"img_{i:03d}.png")
            if not os.path.isfile(p):
                return None
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr))
        if len({tuple(f.shape) for f in frames}) == 1:
            images = torch.stack(frames, 0)
        else:
            images = [f[None] for f in frames]     # caller decides how to batch mixed sizes
    video = ""
    if meta.get("video"):
        video = os.path.join(d, meta["video"])
        if not os.path.isfile(video):
            return None
    now = time.time()
    try:
        os.utime(meta_path, (now, now))            # LRU touch
    except OSError:
        pass
    return {"images": images, "video": video, "texts": meta.get("texts", []), "meta": meta}


def store(key: str | None, *, images=None, video_path: str = "", texts=(), info: dict | None = None) -> None:
    """Save one result. Failures are logged and swallowed: the cache must never
    break a generation that already succeeded (and was billed)."""
    if not key:
        return
    try:
        import numpy as np
        from PIL import Image
        d = os.path.join(CACHE_DIR, key)
        tmp = d + f".tmp{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        count = 0
        if images is not None:
            batch = images if isinstance(images, (list, tuple)) else [images[i] for i in range(images.shape[0])]
            for i, img in enumerate(batch):
                if img.dim() == 4:
                    img = img[0]
                arr = (img.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(tmp, f"img_{i:03d}.png"), compress_level=4)
                count += 1
        video_name = ""
        if video_path and os.path.isfile(video_path):
            video_name = "video" + os.path.splitext(video_path)[1]
            shutil.copy2(video_path, os.path.join(tmp, video_name))
        meta = {"created": time.time(), "image_count": count, "video": video_name,
                "texts": list(texts), "info": info or {}}
        with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        with _lock:
            shutil.rmtree(d, ignore_errors=True)
            os.replace(tmp, d)
        _evict()
    except Exception as e:
        print(f"[Onyx Cache] could not save result: {e}")


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _evict() -> None:
    limit = _max_bytes()
    try:
        entries = []
        for name in os.listdir(CACHE_DIR):
            p = os.path.join(CACHE_DIR, name)
            meta = os.path.join(p, "meta.json")
            if os.path.isdir(p) and os.path.isfile(meta):
                entries.append((os.path.getmtime(meta), _dir_size(p), p))
    except OSError:
        return
    total = sum(e[1] for e in entries)
    for _mtime, size, path in sorted(entries):
        if total <= limit:
            break
        shutil.rmtree(path, ignore_errors=True)
        total -= size


def stats() -> dict:
    try:
        names = [n for n in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, n))]
    except OSError:
        names = []
    return {"entries": len(names), "bytes": _dir_size(CACHE_DIR) if names else 0,
            "limit_bytes": _max_bytes(), "path": CACHE_DIR}
