"""
Onyx Image and Video Batch Loader
Sequential batch loader with drag-and-drop UI.
Loads images and videos one at a time in order, cycling through the uploaded list.
"""

import os
import json
import uuid
import hashlib
import torch
import numpy as np
from PIL import Image
from aiohttp import web
import folder_paths
from server import PromptServer

# ─────────────────────────────────────────────────────────────────────────────
from .onyx_render_profile import ensure_profile_ready
_POOL_SUBDIR = "Onyx_ImagePool"
_THUMB_PREFIX = "thumb_"

_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}

_EMPTY_IMAGE_SIDE = 64

# Mode "MiniMax 15s limit" : MiniMax H3 ne prend pas plus de 15 s de reference.
# Un clip qui depasse de peu (jusqu'a 16.5 s) est coupe a 15 s pile — on perd au
# plus 1.5 s de fin, ce qui ne change pas le clip. Au-dela, couper retirerait un
# morceau reel du contenu : le clip est ecarte plutot que mutile en silence.
_MINIMAX_LIMIT_S = 15.0
_MINIMAX_HARD_S = 16.5


def _pool_dir() -> str:
    d = os.path.join(folder_paths.get_input_directory(), _POOL_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _is_video(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in _VIDEO_EXTS


class _AnyType(str):
    """A type string that ComfyUI's link check accepts against any socket.

    Needed for `path`. LoadVideoUI declares its `video` input as a list, which the
    frontend presents as a COMBO socket, and a STRING output simply cannot be
    linked to it — the check is a plain type comparison. Returning False from
    __ne__ makes that comparison pass while the value stays an ordinary string,
    which is exactly what LoadVideoUI wants: it resolves `video` by testing the
    raw value as a filesystem path before anything else.

    The trade is that this output can be dropped onto sockets where a path makes
    no sense; ComfyUI will accept the link and the receiving node will fail on the
    value instead of at connection time.
    """

    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


# ─────────────────────────────────────────────────────────────────────────────
# Video decoding
#
# Trois backends essayes dans l'ordre plutot qu'un seul suppose : ce pack tourne
# sur des installations tres differentes, et une dependance absente doit donner
# un message lisible, pas un ImportError au milieu d'un run. PyAV en premier
# parce que c'est ce que ComfyUI utilise nativement pour la video, donc celui qui
# a le plus de chances d'etre deja la et de decoder les memes fichiers.
# ─────────────────────────────────────────────────────────────────────────────

def _decode_with_av(path, max_side):
    import av
    frames, fps = [], 0.0
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for frame in container.decode(stream):
            pil = frame.to_image()
            frames.append(_fit(pil, max_side))
    return frames, fps


def _decode_with_cv2(path, max_side):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cv2 could not open the file")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(_fit(Image.fromarray(rgb), max_side))
    cap.release()
    return frames, fps


def _decode_with_imageio(path, max_side):
    import imageio.v3 as iio
    meta = iio.immeta(path)
    fps = float(meta.get("fps") or 0.0)
    frames = [_fit(Image.fromarray(f), max_side) for f in iio.imiter(path)]
    return frames, fps


def _fit(pil: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the long edge is at most max_side. 0 keeps native size.

    A 15 s 1080p clip is 360 frames; held as float32 RGB that is over 2 GB of
    RAM before anything else runs. The default cap exists so dropping a normal
    phone video into the node does not take the machine down.
    """
    if not max_side:
        return pil.convert("RGB")
    w, h = pil.size
    longest = max(w, h)
    if longest <= max_side:
        return pil.convert("RGB")
    scale = max_side / float(longest)
    return pil.convert("RGB").resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS
    )


def _decode_video(path: str, max_side: int = 1024):
    """Return (frames_tensor (N,H,W,3), fps). Raises with the list of failures."""
    errors = []
    for name, fn in (("av", _decode_with_av), ("cv2", _decode_with_cv2),
                     ("imageio", _decode_with_imageio)):
        try:
            frames, fps = fn(path, max_side)
            if not frames:
                errors.append(f"{name}: decoded 0 frames")
                continue
            arr = np.stack([np.asarray(f, dtype=np.float32) / 255.0 for f in frames])
            print(f"[Onyx Batch] 🎞️  decoded with {name}: {len(frames)} frames "
                  f"@ {fps:.2f} fps, {frames[0].size[0]}x{frames[0].size[1]}")
            return torch.from_numpy(arr), (fps or 24.0)
        except ImportError:
            errors.append(f"{name}: not installed")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(
        "[Onyx Batch] Could not decode the video. Tried:\n  "
        + "\n  ".join(errors)
        + "\n-> Install one of: av (recommended, same decoder as ComfyUI), "
          "opencv-python, imageio[ffmpeg]."
    )


def _probe_video(path: str):
    """(duration_s, fps, frame_count) read from the container, without decoding.

    Cheap enough to run at upload for every file, which is what lets the node
    show each clip's length — and flag the ones over the MiniMax limit — before
    anything is queued. Returns zeros when no backend can tell.
    """
    try:
        import av
        with av.open(path) as container:
            st = container.streams.video[0]
            fps = float(st.average_rate) if st.average_rate else 0.0
            n = int(st.frames or 0)
            dur = 0.0
            if n and fps:
                dur = n / fps
            elif st.duration is not None and st.time_base is not None:
                dur = float(st.duration * st.time_base)
            elif container.duration:
                dur = container.duration / 1_000_000.0
            if not n and dur and fps:
                n = int(round(dur * fps))
            return float(dur), float(fps), int(n)
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return (n / fps if fps else 0.0), fps, n
    except Exception:
        return 0.0, 0.0, 0


def _effective_duration(dur, fps, trim_mode, trim_start, trim_end):
    """Length the clip will have once the node's trim rule is applied.

    Mirrors _apply_trim so the limit is judged on what actually leaves the node,
    not on the raw file: a 20 s clip trimmed to its first 12 s is fine.
    """
    if dur <= 0:
        return 0.0
    if trim_mode == "frames":
        if fps <= 0:
            return dur
        start = trim_start / fps
        end = (trim_end / fps) if trim_end > 0 else dur
    else:
        start = trim_start
        end = trim_end if trim_end > 0 else dur
    return max(0.0, min(end, dur) - min(start, dur))


def _slice_audio(audio, start_s: float, dur_s: float):
    """Cut an AUDIO dict to [start_s, start_s + dur_s] so it matches the frames.

    Before this, trimming a clip trimmed only its pictures: the audio output
    still ran from 0 to the end of the file, out of sync with the video from the
    first frame whenever trim_start was set.
    """
    if audio is None:
        return None
    sr = int(audio["sample_rate"])
    wav = audio["waveform"]
    total = int(wav.shape[-1])
    a = max(0, min(total, int(round(start_s * sr))))
    b = max(a, min(total, a + int(round(dur_s * sr))))
    if (a, b) == (0, total):
        return audio
    return {"waveform": wav[..., a:b].contiguous(), "sample_rate": sr}


def _frame_at(path: str, t: float = 0.0) -> Image.Image:
    """The single frame shown at time t (seconds), decoded on its own.

    Used by the "video first frame" mode. Decoding the whole clip to keep one
    frame is what made that mode fail: a 10 s 1080x1920 phone video is ~300
    frames, ~7.5 GB as float32 RGB — enough to kill the process or raise a
    MemoryError before anything reached the `image` output. Here we seek to
    the nearest keyframe before t and decode forward to t: a few frames at
    most, whatever the clip length or resolution.
    """
    errors = []
    try:
        import av
        with av.open(path) as container:
            stream = container.streams.video[0]
            if t > 0 and stream.time_base:
                try:
                    container.seek(int(t / float(stream.time_base)), stream=stream,
                                   backward=True, any_frame=False)
                except Exception:
                    container.seek(0)
            last = None
            for frame in container.decode(stream):
                last = frame
                ft = frame.time if frame.time is not None else 0.0
                if ft + 1e-3 >= t:
                    return frame.to_image().convert("RGB")
            if last is not None:      # t past the end: last frame
                return last.to_image().convert("RGB")
        errors.append("av: no video frame")
    except ImportError:
        errors.append("av: not installed")
    except Exception as e:
        errors.append(f"av: {e}")
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if t > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, bgr = cap.read()
        cap.release()
        if ok:
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        errors.append("cv2: could not read a frame")
    except ImportError:
        errors.append("cv2: not installed")
    except Exception as e:
        errors.append(f"cv2: {e}")
    raise RuntimeError("no video backend could read the frame — " + "; ".join(errors))


def _first_frame(path: str) -> Image.Image:
    """First frame of a video, for the gallery thumbnail.

    Decodes one frame and stops, rather than reusing _decode_video: pulling a
    whole 15 s clip into memory just to make a 300 px preview would stall the
    upload for every file dropped on the node.
    """
    errors = []
    try:
        import av
        with av.open(path) as container:
            for frame in container.decode(container.streams.video[0]):
                return frame.to_image().convert("RGB")
        errors.append("av: no video frame")
    except ImportError:
        errors.append("av: not installed")
    except Exception as e:
        errors.append(f"av: {e}")

    try:
        import cv2
        cap = cv2.VideoCapture(path)
        ok, bgr = cap.read()
        cap.release()
        if ok:
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        errors.append("cv2: could not read a frame")
    except ImportError:
        errors.append("cv2: not installed")
    except Exception as e:
        errors.append(f"cv2: {e}")

    raise RuntimeError("no video backend could read a first frame — " + "; ".join(errors))


def _resize_frames(frames, method: str, target_w: int, target_h: int):
    """Resize a (N,H,W,C) batch to target_w x target_h.

    Runs on the whole batch at once through torch rather than frame by frame with
    PIL: 360 frames is 360 round trips otherwise, and the tensor is already in
    memory in the right layout apart from the channel axis.
    """
    import torch.nn.functional as F

    if method == "none" or (target_w <= 0 and target_h <= 0):
        return frames

    n, h, w, c = frames.shape

    # Une seule dimension fournie : l'autre suit le rapport d'origine, ce qui
    # evite d'avoir a saisir les deux quand on veut juste "toutes en 480 de haut".
    if target_w <= 0:
        target_w = max(1, round(w * target_h / h))
    if target_h <= 0:
        target_h = max(1, round(h * target_w / w))

    chw = frames.permute(0, 3, 1, 2)  # (N,C,H,W) pour interpolate

    if method == "stretch to fit":
        out = F.interpolate(chw, size=(target_h, target_w), mode="bilinear", align_corners=False)

    elif method == "crop":
        # Couvre la cible puis recadre au centre : aucune deformation, on perd
        # les bords du plan le plus long.
        scale = max(target_w / w, target_h / h)
        rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
        res = F.interpolate(chw, size=(rh, rw), mode="bilinear", align_corners=False)
        top, left = (rh - target_h) // 2, (rw - target_w) // 2
        out = res[:, :, top:top + target_h, left:left + target_w]

    elif method == "pad":
        # Tient dans la cible puis complete en noir : rien n'est perdu ni
        # deforme, on ajoute des bandes.
        scale = min(target_w / w, target_h / h)
        rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
        res = F.interpolate(chw, size=(rh, rw), mode="bilinear", align_corners=False)
        pad_t, pad_l = (target_h - rh) // 2, (target_w - rw) // 2
        out = F.pad(res, (pad_l, target_w - rw - pad_l, pad_t, target_h - rh - pad_t))

    else:  # "maintain aspect ratio" — tient dans la boite, sans completer
        scale = min(target_w / w, target_h / h)
        rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
        out = F.interpolate(chw, size=(rh, rw), mode="bilinear", align_corners=False)

    out = out.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)
    print(f"[Onyx Batch] 📐 resized {w}x{h} -> {out.shape[2]}x{out.shape[1]} ({method})")
    return out


def _extract_audio(path: str):
    """Return a ComfyUI AUDIO dict, or None when the file carries no audio.

    Missing audio is a normal case, not an error: plenty of clips are silent,
    and refusing to load them would be worse than returning nothing.
    """
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(path) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            sr = int(stream.rate)
            chunks = [f.to_ndarray() for f in container.decode(stream)]
        if not chunks:
            return None
        data = np.concatenate(chunks, axis=-1) if chunks[0].ndim > 1 else np.concatenate(chunks)
        if data.ndim == 1:
            data = data[None, :]
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        wav = torch.from_numpy(np.ascontiguousarray(data)).float().unsqueeze(0)
        print(f"[Onyx Batch] 🔊 audio: {wav.shape[1]} channel(s) @ {sr} Hz")
        return {"waveform": wav, "sample_rate": sr}
    except Exception as e:
        print(f"[Onyx Batch] audio not extracted: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
class OnyxImageBatchLoader:
    """
    Onyx Image and Video Batch Loader
    Upload images and videos via the node UI and process them sequentially.
    Each execution loads the next item in the list.
    Click «Queue All» to process every image.
    """

    _states: dict = {}  # state_key → {"current_index": int}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Managed entirely by the JS widget — hidden from the user via JS.
                # socketless partout : ces reglages sont pilotes par l'UI du node et
                # caches. Sans ca, le frontend leur cree quand meme une entree
                # chacun, et ces entrees de widgets caches s'empilent au meme
                # endroit en haut du node.
                "batch_data": ("STRING", {"default": "{}", "socketless": True}),
                "video_max_side": ("INT", { "socketless": True,
                    "default": 1024, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Longest edge of the decoded video frames. 0 keeps native size.\n"
                               "A 15 s 1080p clip is 360 frames, over 2 GB held as float32 RGB. "
                               "The cap keeps a normal phone video from taking the machine down.\n"
                               "Irrelevant for images, which are never resized.",
                }),
                # Le decoupage est une REGLE, pas une valeur : la meme pour tous
                # les clips de la file, alors que leur duree differe. D'ou la
                # convention 0 = fin de la video, qui rend la regle applicable
                # a des longueurs quelconques sans rien recalculer a la main.
                "trim_mode": (["seconds", "frames"], {"socketless": True,
                    "default": "seconds",
                    "tooltip": "Whether the trim below is expressed in seconds or in frames.",
                }),
                "trim_start": ("FLOAT", { "socketless": True,
                    "default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01,
                    "tooltip": "Where each clip starts. 0 keeps the beginning.",
                }),
                "trim_end": ("FLOAT", { "socketless": True,
                    "default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01,
                    "tooltip": "Where each clip ends. **0 means the end of the video**, so the "
                               "same setting works across clips of different lengths — which is "
                               "the whole point of trimming in a batch.\n"
                               "Set 8 with trim_mode=seconds to keep the first 8 seconds of every "
                               "clip, whatever its duration.",
                }),
                "resize_method": (
                    ["none", "maintain aspect ratio", "stretch to fit", "pad", "crop"], {"socketless": True,
                        "default": "none",
                        "tooltip": "How each clip is fitted to custom_width x custom_height.\n"
                                   "maintain aspect ratio: fits inside the box, output size varies "
                                   "with each source's ratio.\n"
                                   "stretch to fit: exact size, distorts.\n"
                                   "pad: exact size, black bars, nothing lost.\n"
                                   "crop: exact size, no distortion, edges lost.\n\n"
                                   "Use pad or crop when every clip must come out the same size — "
                                   "'maintain aspect ratio' does not guarantee that across a batch "
                                   "of mixed sources.",
                    }),
                "custom_width": ("INT", { "socketless": True,
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "0 derives the width from the height and the source ratio.",
                }),
                "custom_height": ("INT", { "socketless": True,
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "0 derives the height from the width and the source ratio.",
                }),
                "force_fps": ("FLOAT", { "socketless": True,
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Resample every clip to this frame rate. 0 keeps each clip's native "
                               "rate.\nSet 24 to normalise a batch of mixed 30 and 60 fps sources — "
                               "MiniMax H3 works at 24, and a clip left at 30 plays back at the "
                               "wrong speed.",
                }),
                # Les deux widgets ci-dessous sont ajoutes EN DERNIER : ComfyUI
                # serialise les valeurs par POSITION, donc les inserer plus haut
                # decalerait tous les reglages des workflows deja enregistres.
                "consume_on_load": ("BOOLEAN", { "socketless": True,
                    "default": False,
                    "label_on": "remove each item once loaded",
                    "label_off": "keep the whole list",
                    "tooltip": "ON: an item disappears from the list as soon as it has been "
                               "loaded. A crash three quarters of the way through a batch then "
                               "costs you only what had not run yet — re-queue and it carries on "
                               "instead of starting over.\n"
                               "The file itself is NOT deleted from disk, only removed from this "
                               "node's list.\n"
                               "Pair it with queue_batch_size = 1: an item is consumed when it is "
                               "LOADED, not when the graph finishes, so anything still queued "
                               "behind a failure is lost from the list.",
                }),
                "queue_batch_size": ("INT", { "socketless": True,
                    "default": 50, "min": 1, "max": 50, "step": 1,
                    "tooltip": "How many runs 'Queue All' submits at a time.\n"
                               "50 is fastest. 1 is the safe setting: a failure then costs one "
                               "item instead of everything already sitting in the queue.\n"
                               "With consume_on_load ON, the button waits for the queue to empty "
                               "between batches — it has to, because each run rewrites the list "
                               "the next submission is built from.",
                }),
            },
            "optional": {
                # OPTIONNEL, pas required : un workflow enregistre avant cet ajout,
                # ou un onglet pas encore recharge apres la mise a jour, n'envoie pas
                # ce reglage — en required, ComfyUI refusait alors TOUT le run avec
                # 'Required input is missing: minimax_15s_limit', meme pour une
                # simple photo. Absent = off. Reste le dernier widget : l'ordre des
                # valeurs enregistrees ne bouge pas.
                "minimax_15s_limit": ("BOOLEAN", {
                    "default": False, "socketless": True,
                    "label_on": "on", "label_off": "off",
                    "tooltip": "MiniMax H3 takes at most 15 s of reference video.\n"
                               "ON: a clip between 15 s and 16.5 s (after trim) is cut to 15 s "
                               "by dropping its end. A clip longer than 16.5 s is skipped — "
                               "cutting it would remove real content — and the next item loads "
                               "in its place. Images are not affected.",
                }),
                # Optionnel et en dernier, pour les memes raisons que le precedent.
                "video_first_frame": ("BOOLEAN", {
                    "default": False, "socketless": True,
                    "label_on": "on", "label_off": "off",
                    "tooltip": "ON: for each video, ONLY its first frame is output, through `image` "
                               "(photos keep coming out of `image` too, in sequence). The video is "
                               "not decoded as a whole: one frame is read at the trim start, then "
                               "Max side / resize are applied. `video` and `audio` stay empty.\n"
                               "OFF: normal video loading; `image` is a 64x64 placeholder for videos.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    # `video` is a separate output on purpose. It is an IMAGE batch like `image`,
    # so ComfyUI would happily let them be swapped, and a frame sequence wired
    # into a single-image input fails much later with a confusing error.
    # `path` is the absolute path of the current item in the pool. LoadVideoUI
    # resolves its `video` argument with `video_path = video` before trying any
    # ComfyUI lookup, and its VALIDATE_INPUTS returns True unconditionally, so an
    # absolute path fed there loads the file directly — which is what turns a
    # batch of videos into a dynamic source for a workflow built around it.
    # duration / frame_count / video_fps are OUTPUTS, not refreshed widgets.
    # LoadVideoUI needs its browser to probe the file and fill its widgets because
    # they are editable trim controls; that refresh only fires on a manual click,
    # which is precisely what a queue cannot do. Reading the real values in Python
    # at execution time gives the correct figures for every item in the batch,
    # with no round trip through the frontend.
    RETURN_TYPES  = ("IMAGE", "IMAGE", "FLOAT", "FLOAT", "INT", "AUDIO", _ANY)
    RETURN_NAMES  = ("image", "video", "video_fps", "duration", "frame_count", "audio", "path")
    OUTPUT_NODE   = False
    FUNCTION      = "load_next"
    CATEGORY      = "Onyx"
    DESCRIPTION   = (
        "Sequential batch loader for images and videos.\n"
        "Upload via the node UI, then click Queue All.\n"
        "Each queue run outputs the next item: images leave through `image`, "
        "videos through `video` + `video_fps` + `duration` + `frame_count` + `audio`.\n"
        "Trim and frame rate are applied per clip, using each clip's own real values."
    )

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_trim(frames, fps, trim_mode, trim_start, trim_end, force_fps):
        """Trim then optionally resample. Returns (frames, fps, start_seconds).

        Resampling is nearest-neighbour on the time axis: it drops or repeats
        whole frames rather than blending them. Blending would invent motion that
        never happened, which is the opposite of what a motion reference is for.
        """
        total = int(frames.shape[0])
        if total == 0:
            return frames, fps, 0.0

        if trim_mode == "frames":
            start = int(trim_start)
            end = int(trim_end) if trim_end > 0 else total
        else:
            start = int(round(trim_start * fps))
            end = int(round(trim_end * fps)) if trim_end > 0 else total

        start = max(0, min(start, total - 1))
        end = max(start + 1, min(end, total))
        start_s = start / fps if fps else 0.0

        if (start, end) != (0, total):
            frames = frames[start:end]
            print(f"[Onyx Batch] ✂️  trimmed to frames {start}-{end} "
                  f"({(end - start) / fps:.3f}s of {total / fps:.3f}s)")

        if force_fps > 0 and abs(force_fps - fps) > 1e-6:
            n_in = int(frames.shape[0])
            n_out = max(1, int(round(n_in * force_fps / fps)))
            idx = torch.linspace(0, n_in - 1, n_out).round().long().clamp(0, n_in - 1)
            frames = frames[idx]
            print(f"[Onyx Batch] ⏱️  resampled {fps:.2f} -> {force_fps:.2f} fps "
                  f"({n_in} -> {n_out} frames)")
            fps = float(force_fps)

        return frames, fps, start_s

    # ─────────────────────────────────────────────────────────────────────────
    def load_next(self, batch_data: str = "{}", video_max_side: int = 1024,
                  trim_mode: str = "seconds", trim_start: float = 0.0,
                  trim_end: float = 0.0, resize_method: str = "none",
                  custom_width: int = 0, custom_height: int = 0,
                  force_fps: float = 0.0, consume_on_load: bool = False,
                  queue_batch_size: int = 50, minimax_15s_limit: bool = False,
                  video_first_frame: bool = False, unique_id=None):
        # ── Parse data ────────────────────────────────────────────────────────
        ensure_profile_ready()
        try:
            data = json.loads(batch_data) if batch_data else {}
        except (json.JSONDecodeError, TypeError):
            data = {}

        images_meta = [
            x for x in data.get("images", [])
            if isinstance(x, dict) and x.get("id")
        ]
        order = [x for x in data.get("order", []) if isinstance(x, str)]

        empty = torch.zeros((1, _EMPTY_IMAGE_SIDE, _EMPTY_IMAGE_SIDE, 3), dtype=torch.float32)
        # Le placeholder image reste 64x64 noir : grok_prompt.py le detecte par
        # cette taille exacte pour ne pas facturer une analyse vision sur du vide.
        # Ne pas changer sans mettre a jour _is_placeholder_frame la-bas.
        blank = (empty, empty, 0.0, 0.0, 0, None, "")

        if not images_meta or not order:
            return blank

        # Build ordered flat list
        flat = []
        for img_id in order:
            meta = next((m for m in images_meta if m["id"] == img_id), None)
            if meta:
                flat.append(meta)

        total = len(flat)
        if total == 0:
            return blank

        # ── Sequential state ──────────────────────────────────────────────────
        state_key = f"{unique_id}_{hashlib.md5(batch_data.encode()).hexdigest()[:8]}"
        if state_key not in self._states:
            self._states[state_key] = {"current_index": 0}

        idx   = self._states[state_key]["current_index"] % total
        pool  = _pool_dir()

        # Mode MiniMax : un clip trop long est saute, et l'item suivant prend sa
        # place dans CE run. Le bouton Queue All ne compte deja pas ces clips,
        # donc N runs couvrent exactement les N items acceptables.
        if minimax_15s_limit:
            for _ in range(total):
                m = flat[idx]
                if not _is_video(m["filename"]):
                    break
                p = os.path.join(pool, m["filename"])
                dur, vfps = float(m.get("duration") or 0.0), float(m.get("fps") or 0.0)
                if dur <= 0 and os.path.exists(p):
                    dur, vfps, _n = _probe_video(p)
                eff = _effective_duration(dur, vfps, trim_mode,
                                          float(trim_start), float(trim_end))
                if eff <= _MINIMAX_HARD_S:
                    break
                print(f"[Onyx Batch] ⏭️  skipped '{m.get('original_name', m['filename'])}' — "
                      f"{eff:.2f}s after trim, over the {_MINIMAX_HARD_S}s MiniMax limit.")
                idx = (idx + 1) % total
            else:
                print(f"[Onyx Batch] ⚠️  every item is over the {_MINIMAX_HARD_S}s MiniMax "
                      f"limit — nothing to load.")
                return blank

        meta  = flat[idx]
        _next = (idx + 1) % total
        self._states[state_key]["current_index"] = _next

        # ── Load ──────────────────────────────────────────────────────────────
        path = os.path.join(pool, meta["filename"])
        name = meta.get("original_name", meta["filename"])

        if not os.path.exists(path):
            print(f"[Onyx Batch] File not found: {path}")
            self._notify(unique_id, idx, total)
            return blank

        if _is_video(meta["filename"]) and video_first_frame:
            # Mode "first frame" : UNE image et rien d'autre. Pas de decodage
            # de la video entiere (voir _frame_at) ; les sorties video restent
            # vides. La frame est prise au debut de la decoupe et passe par le
            # meme plafond / redimensionnement que la video l'aurait fait.
            try:
                dur, vfps, _nf = _probe_video(path)
                if trim_mode == "frames":
                    t0 = float(trim_start) / vfps if vfps else 0.0
                else:
                    t0 = float(trim_start)
                pil = _fit(_frame_at(path, max(0.0, t0)), int(video_max_side))
                frame = torch.from_numpy(
                    np.asarray(pil, dtype=np.float32) / 255.0).unsqueeze(0)
                frame = _resize_frames(frame, resize_method,
                                       int(custom_width), int(custom_height))
                frame = frame.contiguous()
                print(f"[Onyx Batch] ✅ [{idx + 1}/{total}] 🖼️  first frame of {name} "
                      f"at {t0:.2f}s — {frame.shape[2]}x{frame.shape[1]}")
                self._notify(unique_id, idx, total)
                if consume_on_load:
                    self._consume(unique_id, meta["id"], name, total - 1)
                return (frame, empty, float(vfps), 0.0, 0, None, path)
            except Exception as e:
                print(f"[Onyx Batch] ❌ first frame of {name}: {e}")
                self._notify(unique_id, idx, total)
                return blank

        if _is_video(meta["filename"]):
            try:
                # Le plafond memoire ne doit pas descendre sous la cible de
                # redimensionnement, sinon on decoderait en dessous puis on
                # reagrandirait — une perte de detail invisible dans les reglages.
                cap = int(video_max_side)
                if resize_method != "none":
                    cap = max(cap, int(custom_width), int(custom_height))
                frames, fps = _decode_video(path, cap)
                frames = _resize_frames(
                    frames, resize_method, int(custom_width), int(custom_height)
                )
                frames, fps, start_s = self._apply_trim(
                    frames, fps, trim_mode, float(trim_start), float(trim_end), float(force_fps)
                )
                n = int(frames.shape[0])
                if minimax_15s_limit and fps and n / fps > _MINIMAX_LIMIT_S + 1e-6:
                    keep = max(1, int(_MINIMAX_LIMIT_S * fps + 1e-6))
                    print(f"[Onyx Batch] ✂️  MiniMax limit: {n / fps:.3f}s -> "
                          f"{keep / fps:.3f}s (end dropped)")
                    frames = frames[:keep]
                    n = keep
                duration = n / fps if fps else 0.0
                audio = _slice_audio(_extract_audio(path), start_s, duration)
                print(f"[Onyx Batch] ✅ [{idx + 1}/{total}] 🎬 {name} — "
                      f"{n} frames, {duration:.3f}s @ {fps:.2f} fps")
                self._notify(unique_id, idx, total)
                if consume_on_load:
                    self._consume(unique_id, meta["id"], name, total - 1)
                # Par defaut `image` reste le placeholder : renvoyer la premiere
                # frame ferait passer une video pour une photo dans un graphe
                # branche sur les deux. video_first_frame le demande explicitement.
                return (empty, frames, float(fps), float(duration), n, audio, path)
            except Exception as e:
                print(f"[Onyx Batch] ❌ {name}: {e}")
                self._notify(unique_id, idx, total)
                return blank

        try:
            pil_img = Image.open(path).convert("RGB")
            arr     = np.array(pil_img).astype(np.float32) / 255.0
            tensor  = torch.from_numpy(arr).unsqueeze(0)
            print(f"[Onyx Batch] ✅ [{idx + 1}/{total}] 🖼️  {name}")
        except Exception as e:
            print(f"[Onyx Batch] ❌ Error loading {meta['filename']}: {e}")
            self._notify(unique_id, idx, total)
            return blank

        self._notify(unique_id, idx, total)
        # Consomme UNIQUEMENT apres un chargement reussi : un fichier illisible
        # doit rester dans la liste, sinon il disparait sans avoir rien produit
        # et sans qu'on sache lequel c'etait.
        if consume_on_load:
            self._consume(unique_id, meta["id"], name, total - 1)
        return (tensor, empty, 0.0, 0.0, 0, None, path)

    @staticmethod
    def _consume(node_id, img_id: str, name: str, left: int):
        """Tell the frontend to drop this item from the list.

        It has to go through the frontend: the list lives in the batch_data
        widget, which only the browser owns. Python cannot edit a widget, but it
        can say what happened and let the JS rewrite it.

        The already-queued prompts each carry their OWN snapshot of batch_data,
        taken when they were submitted, so removing an entry here never disturbs
        a run that is already in flight. It only shortens the list the NEXT
        submission is built from — which is exactly what makes a batch
        resumable.
        """
        try:
            PromptServer.instance.send_sync(
                "onyx_batch_loader_consume",
                {"node_id": str(node_id), "image_id": str(img_id),
                 "name": name, "left": int(left)},
            )
            print(f"[Onyx Batch] 🗑️  consumed '{name}' — {left} left in the list.")
        except Exception as e:
            # Le rendu est deja fait : perdre la notification coute une entree
            # en trop dans la liste, pas la generation.
            print(f"[Onyx Batch] ⚠️  could not notify the frontend ({e}) — "
                  f"'{name}' stays in the list.")

    @staticmethod
    def _notify(node_id, current_index: int, total: int):
        try:
            PromptServer.instance.send_sync(
                "onyx_batch_loader_update",
                {"node_id": str(node_id), "current_index": current_index, "total": total},
            )
        except Exception:
            pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always re-execute


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
#
# Guard : evite le crash "Cannot register a resource into frozen router"
# quand ComfyUI re-importe le module au restart. Sans ce guard, les
# decorateurs ci-dessous sont re-executes -> aiohttp leve une exception ->
# shutdown bloque -> port 8190 + DB lock restent detenus -> nouveau ComfyUI
# ne peut pas demarrer. Pattern identique a celui utilise dans nano_banana_aio.
# ─────────────────────────────────────────────────────────────────────────────

if not getattr(PromptServer.instance, "_onyx_batch_routes_registered", False):
    PromptServer.instance._onyx_batch_routes_registered = True

    @PromptServer.instance.routes.post("/onyx/batch_upload")
    async def _onyx_batch_upload(request):
        """Upload one or more images to the shared pool."""
        try:
            reader  = await request.multipart()
            pool    = _pool_dir()
            results = []

            async for field in reader:
                if field.name != "files":
                    continue
                raw_name = field.filename or "image.jpg"
                data     = await field.read()

                file_id  = str(uuid.uuid4())
                ext      = os.path.splitext(raw_name)[1].lower() or ".jpg"
                safe     = f"{file_id}{ext}"
                path     = os.path.join(pool, safe)

                with open(path, "wb") as fh:
                    fh.write(data)

                try:
                    if _is_video(safe):
                        # Vignette = premiere frame. Le thumbnail est toujours un
                        # PNG, quelle que soit l'extension source, pour que la
                        # galerie du widget n'ait pas a savoir lire de la video.
                        img = _first_frame(path)
                        is_video = True
                    else:
                        img = Image.open(path)
                        is_video = False

                    w, h = img.size

                    thumb      = img.convert("RGB").copy()
                    thumb.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    thumb_name = f"{_THUMB_PREFIX}{os.path.splitext(safe)[0]}.png"
                    thumb.save(os.path.join(pool, thumb_name))

                    item = {
                        "id":            file_id,
                        "filename":      safe,
                        "original_name": raw_name,
                        "thumbnail":     thumb_name,
                        "width":         w,
                        "height":        h,
                        "is_video":      is_video,
                        "size":          len(data),
                    }
                    if is_video:
                        dur, vfps, nfr = _probe_video(path)
                        item.update({"duration": round(dur, 3), "fps": round(vfps, 3),
                                     "frame_count": nfr})
                    results.append(item)
                except Exception as e:
                    print(f"[Onyx Batch] Error processing {raw_name}: {e}")
                    if os.path.exists(path):
                        os.remove(path)

            return web.json_response({"success": True, "images": results})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)


    @PromptServer.instance.routes.delete("/onyx/batch_delete/{image_id}")
    async def _onyx_batch_delete(request):
        """Delete an image (and its thumbnail) from the pool."""
        image_id = request.match_info["image_id"]
        try:
            pool    = _pool_dir()
            deleted = []
            for fn in os.listdir(pool):
                if image_id in fn:
                    os.remove(os.path.join(pool, fn))
                    deleted.append(fn)
            return web.json_response({"success": True, "deleted": deleted})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)


    @PromptServer.instance.routes.get("/onyx/batch_probe/{filename}")
    async def _onyx_batch_probe(request):
        """Duration / fps of a pool video. Lets the UI fill in lengths for items
        uploaded before durations were recorded at upload time."""
        filename = os.path.basename(request.match_info["filename"])
        path = os.path.join(_pool_dir(), filename)
        if not os.path.exists(path) or not _is_video(filename):
            return web.json_response({"success": False}, status=404)
        dur, vfps, nfr = _probe_video(path)
        return web.json_response({"success": True, "duration": round(dur, 3),
                                  "fps": round(vfps, 3), "frame_count": nfr})


    @PromptServer.instance.routes.get("/onyx/view/{filename}")
    async def _onyx_view(request):
        """Serve an image from the pool via ComfyUI's standard /view endpoint."""
        filename = request.match_info["filename"]
        pool     = _pool_dir()
        if os.path.exists(os.path.join(pool, filename)):
            return web.HTTPFound(
                f"/view?filename={filename}&type=input&subfolder={_POOL_SUBDIR}"
            )
        return web.Response(status=404, text=f"Image not found: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "OnyxImageBatchLoader": OnyxImageBatchLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    # La cle reste OnyxImageBatchLoader : elle est inscrite dans chaque
    # workflow sauvegarde, seul le libelle affiche change.
    "OnyxImageBatchLoader": "Onyx Image and Video Batch Loader",
}
