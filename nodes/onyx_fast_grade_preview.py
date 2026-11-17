# -*- coding: utf-8 -*-
# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""Onyx Fast Video Grade — live frame preview (routes used by js/onyx_fast_grade.js)."""

import io
import os
import subprocess
import threading
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from . import onyx_fast_grade as _fg

_VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif")
_PARAM_KEYS = ("exposure", "contrast", "highlights", "shadows", "whites", "blacks", "temperature", "tint",
               "saturation", "vibrance", "lut_strength", "grain_amount", "grain_size", "grain_color", "seed")

_lock = threading.Lock()
_FRAMES = OrderedDict()      # (src, mtime, index) -> uint8 (h, w, 3)
_INFO = {}                   # (src, mtime) -> dict
_LUTS = {}                   # name -> tensor
_LAST_SOURCE = {}            # node id -> source path of its last run
_MEM = {}                    # node id -> (IMAGE batch, fps, version) graded by its last run


def _clean_name(p: str) -> str:
    p = str(p or "").strip().strip('"')
    for tag in (" [input]", " [output]", " [temp]"):
        if p.endswith(tag):
            p = p[: -len(tag)]
    return p


def _resolve(path: str, node_id=None) -> str:
    p = _clean_name(path)
    if p:
        if not os.path.isfile(p):
            try:
                import folder_paths
                for base in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
                    q = os.path.join(base, p)
                    if os.path.isfile(q):
                        return os.path.abspath(q)
            except Exception:
                pass
            raise FileNotFoundError(f"video not found: {p}")
        return os.path.abspath(p)
    if node_id is not None and str(node_id) in _MEM:
        return "mem:" + str(node_id)
    last = _LAST_SOURCE.get(str(node_id)) if node_id is not None else None
    if last and os.path.isfile(last):
        return last
    raise LookupError("no video")


def _stamp(src: str):
    return _MEM[src[4:]][2] if src.startswith("mem:") else os.path.getmtime(src)


def _info(src: str) -> dict:
    key = (src, _stamp(src))
    if key in _INFO:
        return _INFO[key]
    if src.startswith("mem:"):
        images, fps, _v = _MEM[src[4:]]
        h, w = int(images.shape[1]), int(images.shape[2])
        info = {"width": w - w % 2, "height": h - h % 2, "fps": float(fps), "frames": int(images.shape[0]),
                "name": "images"}
        _INFO[key] = info
        return info
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read the video: {src}")
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    h, w = frame.shape[:2]
    info = {"width": w - w % 2, "height": h - h % 2, "fps": float(fps), "frames": max(1, frames),
            "name": os.path.basename(src)}
    _INFO[key] = info
    return info


def _frame(src: str, index: int) -> np.ndarray:
    info = _info(src)
    if src.startswith("mem:"):
        images = _MEM[src[4:]][0]
        f = images[index, : info["height"], : info["width"], :3]
        return (f.float().clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
    key = (src, _stamp(src), index)
    with _lock:
        if key in _FRAMES:
            _FRAMES.move_to_end(key)
            return _FRAMES[key]
    w, h, fps = info["width"], info["height"], info["fps"]
    t = index / fps
    r = subprocess.run([_fg._ffmpeg(), "-v", "error", "-ss", f"{t:.6f}", "-i", src, "-frames:v", "1",
                        "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, timeout=60)
    fb = w * h * 3
    if len(r.stdout) < fb:
        raise RuntimeError("cannot decode this frame: " + r.stderr.decode(errors="ignore")[-300:])
    arr = np.frombuffer(r.stdout[:fb], dtype=np.uint8).reshape(h, w, 3).copy()
    with _lock:
        _FRAMES[key] = arr
        while len(_FRAMES) > 24:
            _FRAMES.popitem(last=False)
    return arr


def _lut(name: str):
    if not name or name == "none":
        return None
    if name not in _LUTS:
        _LUTS[name] = _fg._load_cube(os.path.join(_fg._LUT_DIR, name))
    return _LUTS[name]


def _params(raw: dict) -> dict:
    p = {}
    for k in _PARAM_KEYS:
        v = raw.get(k, 0)
        if k == "grain_color":
            p[k] = bool(v)
        elif k == "seed":
            p[k] = int(v or 0)
        else:
            p[k] = float(v or 0.0)
    if "grain_size" not in raw:
        p["grain_size"] = 1.2
    if "lut_strength" not in raw:
        p["lut_strength"] = 1.0
    return p


def render(path, node_id, index: int, raw: dict, original: bool, size: int, zoom: bool, cx: float, cy: float,
           view_w: int = 0, view_h: int = 0):
    from PIL import Image
    src = _resolve(path, node_id)
    info = _info(src)
    index = max(0, min(int(index), info["frames"] - 1))
    arr = _frame(src, index)
    h, w = arr.shape[:2]
    if not original:
        try:
            import comfy.model_management as mm
            device = mm.get_torch_device()
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        with torch.inference_mode():
            x = torch.from_numpy(arr).to(device).permute(2, 0, 1).unsqueeze(0).to(dtype) / 255.0
            y = _fg.grade(x, _params(raw), _lut(raw.get("lut")), frame0=index)
            arr = (y[0] * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
    if zoom:                                            # 1:1 window around (cx, cy)
        vw, vh = (int(view_w), int(view_h)) if view_w and view_h else (size, size)
        cw, ch = max(16, min(w, vw)), max(16, min(h, vh))
        x0 = int(round(min(max(cx * w - cw / 2, 0), w - cw)))
        y0 = int(round(min(max(cy * h - ch / 2, 0), h - ch)))
        arr = arr[y0:y0 + ch, x0:x0 + cw]
    im = Image.fromarray(np.ascontiguousarray(arr))
    if not zoom and max(w, h) > size:
        s = size / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# ── extra inputs: IMAGE batch (+ AUDIO), and VIDEO objects that have no file ──────────────────
# An IMAGE batch (the end of a workflow) is graded straight from memory on the GPU and piped to
# the encoder, in one pass: no intermediate file. The batch is kept (by reference, it already
# lives in ComfyUI's cache) so the viewer can show its frames after the run.

def _work_dir() -> str:
    try:
        import folder_paths
        d = os.path.join(folder_paths.get_temp_directory(), "onyx_grade_src")
    except Exception:
        import tempfile
        d = os.path.join(tempfile.gettempdir(), "onyx_grade_src")
    os.makedirs(d, exist_ok=True)
    return d


def _write_wav(audio: dict, path: str) -> bool:
    import wave
    wf = audio.get("waveform") if isinstance(audio, dict) else None
    if wf is None:
        return False
    sr = int(audio.get("sample_rate") or 44100)
    a = wf[0] if wf.dim() == 3 else wf                       # (channels, samples)
    a = a.float().clamp(-1, 1).cpu().numpy()
    if a.ndim == 1:
        a = a[None]
    pcm = (a.T * 32767.0).astype("<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(pcm.shape[1])
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())
    return True


def _grade_images(images, audio, fps, a: dict, node_id):
    import queue as _queue
    import folder_paths
    import comfy.utils
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    if n == 0:
        raise ValueError("[Onyx Fast Grade] the image batch is empty.")
    w, h = w - w % 2, h - h % 2
    ff = _fg._ffmpeg()
    params = {k: a[k] for k in _PARAM_KEYS}
    lut_t = _lut(a.get("lut"))

    out_dir, name, counter, sub, _ = folder_paths.get_save_image_path(
        a["filename_prefix"], folder_paths.get_output_directory(), w, h)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{name}_{counter:05}_.mp4"
    out = os.path.join(out_dir, fname)
    enc_name = str(a.get("encoder", ""))
    use_gpu = enc_name.startswith("GPU") or (enc_name.startswith("auto") and _fg._has_nvenc(ff))
    q = str(int(a.get("quality", 18)))
    venc = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", q, "-b:v", "0"]
            if use_gpu else ["-c:v", "libx264", "-preset", "veryfast", "-crf", q])

    wav = os.path.join(_work_dir(), f"audio_{node_id}_{os.getpid()}.wav")
    has_audio = _write_wav(audio, wav)
    rate = f"{float(fps):.6f}"
    cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", rate, "-i", "-"]
    if has_audio:
        cmd += ["-i", wav, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += venc + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    out_q, errors = _queue.Queue(maxsize=3), []

    def writer():
        try:
            while True:
                arr = out_q.get()
                if arr is None:
                    break
                enc.stdin.write(arr.tobytes())
        except Exception as e:
            errors.append(e)

    tw = threading.Thread(target=writer, daemon=True)
    tw.start()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    bar = comfy.utils.ProgressBar(n)
    try:
        with torch.inference_mode():
            for k in range(0, n, _fg._CHUNK):
                x = images[k:k + _fg._CHUNK, :h, :w, :3].to(device, non_blocking=True).permute(0, 3, 1, 2).to(dtype)
                y = _fg.grade(x, params, lut_t, frame0=k)
                y = (y * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
                out_q.put(y)
                bar.update(y.shape[0])
                if errors:
                    break
    finally:
        out_q.put(None)
        tw.join()
        try:
            enc.stdin.close()
        except Exception:
            pass
        err = enc.stderr.read().decode(errors="ignore")
        enc.wait()
        if has_audio:
            try:
                os.remove(wav)
            except OSError:
                pass
    if errors:
        raise RuntimeError(f"[Onyx Fast Grade] {errors[0]}")
    if enc.returncode != 0:
        raise RuntimeError("[Onyx Fast Grade] the encoder failed: " + err[-600:])

    print(f"🎨 [Onyx Fast Grade] {n} frames -> {out} ({'NVENC' if use_gpu else 'x264'})")
    result_video = None
    try:
        from comfy_api.input_impl import VideoFromFile
        result_video = VideoFromFile(out)
    except Exception:
        pass
    return {"ui": {"images": [{"filename": fname, "subfolder": sub, "type": "output"}], "animated": (True,)},
            "result": (result_video, out)}


def _video_to_file(video, node_id) -> str:
    """Path of a VIDEO input; a VIDEO with no file behind it is saved once to the work folder."""
    try:
        src = video.get_stream_source()
        if isinstance(src, str) and os.path.isfile(src):
            return src
    except Exception:
        pass
    out = os.path.join(_work_dir(), f"vid_{node_id}_{os.getpid()}.mp4")
    video.save_to(out)
    return out


_orig_input_types = _fg.OnyxFastVideoGrade.__dict__["INPUT_TYPES"].__func__ \
    if isinstance(_fg.OnyxFastVideoGrade.__dict__.get("INPUT_TYPES"), classmethod) else None


def _input_types(cls):
    d = _orig_input_types(cls) if _orig_input_types else {}
    opt = dict(d.get("optional") or {})
    d["optional"] = {"images": ("IMAGE",), "audio": ("AUDIO",), **opt,
                     "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01})}
    return d


if _orig_input_types and not getattr(_fg.OnyxFastVideoGrade, "_onyx_images", False):
    _fg.OnyxFastVideoGrade.INPUT_TYPES = classmethod(_input_types)
    _fg.OnyxFastVideoGrade._onyx_images = True


# Remember which file each node graded, so a source coming from an upstream node shows after one run.
_orig_run = _fg.OnyxFastVideoGrade.run


def _run(self, *args, **kwargs):
    images = kwargs.pop("images", None)
    audio = kwargs.pop("audio", None)
    fps = float(kwargs.pop("frame_rate", 24.0) or 24.0)
    nid = None
    try:
        from server import PromptServer
        nid = PromptServer.instance.last_node_id
    except Exception:
        pass
    has_path = bool(str(kwargs.get("video_path") or "").strip())
    if not has_path and kwargs.get("video") is not None:
        kwargs["video_path"] = _video_to_file(kwargs["video"], nid)
    elif not has_path and images is not None:
        info = kwargs.get("video_info")
        if isinstance(info, dict):            # images decoded from a video: keep its frame rate
            fps = float(info.get("loaded_fps") or info.get("source_fps") or fps)
        if nid is not None:
            prev = _MEM.get(str(nid))
            _MEM[str(nid)] = (images, fps, (prev[2] + 1) if prev else 1)
            _LAST_SOURCE.pop(str(nid), None)
        return _grade_images(images, audio, fps, kwargs, nid)
    if nid is not None:
        _MEM.pop(str(nid), None)
    try:
        src = _fg._resolve_source(kwargs.get("video_path", ""), kwargs.get("video"), kwargs.get("video_info"))
        if nid is not None and src:
            _LAST_SOURCE[str(nid)] = src
    except Exception:
        pass
    return _orig_run(self, *args, **kwargs)


if not getattr(_fg.OnyxFastVideoGrade.run, "_onyx_preview", False):
    _run._onyx_preview = True
    _fg.OnyxFastVideoGrade.run = _run


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    inst = PromptServer.instance
    if getattr(inst, "_onyx_fast_grade_routes", False):
        return
    inst._onyx_fast_grade_routes = True
    import asyncio

    def err(e, status=400):
        return web.json_response({"ok": False, "error": str(e)}, status=status)

    @inst.routes.post("/onyx/fast_grade/info")
    async def _info_route(request):
        try:
            b = await request.json()
            res = await asyncio.get_running_loop().run_in_executor(
                None, lambda: _info(_resolve(b.get("path", ""), b.get("node_id"))))
            return web.json_response({"ok": True, **res})
        except LookupError as e:
            return err(e, 404)
        except Exception as e:
            return err(e)

    @inst.routes.post("/onyx/fast_grade/frame")
    async def _frame_route(request):
        try:
            b = await request.json()
            data = await asyncio.get_running_loop().run_in_executor(
                None, lambda: render(b.get("path", ""), b.get("node_id"), int(b.get("index", 0)),
                                     b.get("params") or {}, bool(b.get("original")),
                                     max(256, min(4096, int(b.get("size", 960)))), bool(b.get("zoom")),
                                     float(b.get("cx", 0.5)), float(b.get("cy", 0.5)),
                                     int(b.get("view_w", 0) or 0), int(b.get("view_h", 0) or 0)))
            return web.Response(body=data, content_type="image/jpeg", headers={"Cache-Control": "no-store"})
        except LookupError as e:
            return err(e, 404)
        except Exception as e:
            return err(e)
