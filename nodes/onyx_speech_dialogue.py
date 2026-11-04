# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""
Onyx Speech Transcribe + Onyx H3 Window Dialogue — the real words in the H3 prompt.

H3 lip-syncs better when the prompt says WHAT is said and WHO says it. Gemini's
analysis of a muffled soundtrack often gives "[unclear]"; Whisper (large-v3) is far
more accurate on speech.

1. Onyx Speech Transcribe (outside the loop): Whisper turns the source audio into
   timed segments with word timings, then (optional) Gemini listens to the audio and
   looks at frames to say who speaks each segment: the woman on camera, or an
   off-camera voice (and its gender). Output: a TRANSCRIPT (JSON text) + a readable
   version. Cached per audio, so re-runs and rerolls cost nothing.

2. Onyx H3 Window Dialogue (inside the Relay loop, between the prompt and the H3
   conditioning): keeps only the words spoken inside the current window, times them
   from the window's own start, and inserts them into that window's prompt:
       At 00:00.400, <Subject 1> (S1) says: <d>[English] ...</d>
       At 00:02.100, an off-camera male voice (S2) says: <d>[English] ...</d> She does not mouth these words.
   Every window gets exactly its own lines, nothing repeated.

Needs `faster-whisper` in ComfyUI's Python:
    python_embeded\\python.exe -m pip install faster-whisper
The Whisper model is downloaded on first use (large-v3 ≈ 3 GB) into ComfyUI/models/whisper.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(os.path.dirname(_HERE), "_cache", "transcripts")
_MODELS = ["large-v3", "large-v3-turbo", "medium", "small"]
_LANGS = ["auto", "en", "fr", "es", "pt", "de", "it", "nl", "ru", "ja", "ko", "zh", "ar", "tr", "pl"]
_LANG_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "pt": "Portuguese", "de": "German",
               "it": "Italian", "nl": "Dutch", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
               "zh": "Chinese", "ar": "Arabic", "tr": "Turkish", "pl": "Polish"}
_SPEAKER_MODES = ["gemini (voices + frames)", "one speaker (all hers)", "off"]
_lock = threading.Lock()
_whisper = {"name": None, "model": None}


def _models_dir():
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "whisper")
    except Exception:
        return os.path.join(os.path.dirname(_HERE), "_cache", "whisper")


def _audio_16k_mono(audio) -> np.ndarray:
    import torch
    wav = audio["waveform"]
    if wav.ndim == 3:
        wav = wav[0]
    wav = wav.float().mean(0, keepdim=True)                      # mono
    rate = int(audio["sample_rate"])
    if rate != 16000:
        try:
            import torchaudio
            wav = torchaudio.functional.resample(wav, rate, 16000)
        except Exception:
            n = int(round(wav.shape[-1] * 16000 / rate))
            wav = torch.nn.functional.interpolate(wav[None], size=n, mode="linear", align_corners=False)[0]
    return wav[0].cpu().numpy().astype(np.float32)


# faster-whisper (CTranslate2) is built for CUDA 12 / its own cuDNN, while ComfyUI's torch may run
# CUDA 13 with another cuDNN. Loading both in the same process mixes the cuDNN sub-libraries
# (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH in every later torch conv). So Whisper always runs in a
# separate Python process: its DLLs never enter ComfyUI, and its VRAM is freed when it exits.
_WORKER = r'''
import json, os, sys
import numpy as np
args = json.loads(sys.stdin.read())

def add_cuda12_dlls():
    # Private folder first (models/whisper/cuda12/nvidia/...): torch never looks there, while CUDA 12
    # packages installed in site-packages/nvidia are picked up by torch's own DLL loader and break cuDNN.
    roots = [os.path.join(args["root"], "cuda12", "nvidia")]
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    roots += list(spec.submodule_search_locations or []) if spec else []
    for root in roots:
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            for d in (os.path.join(root, sub, "bin"), os.path.join(root, sub, "lib")):
                if os.path.isdir(d):
                    try:
                        os.add_dll_directory(d)
                    except (OSError, AttributeError):
                        pass
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

def run(device):
    if device == "cuda":
        add_cuda12_dlls()
    from faster_whisper import WhisperModel
    model = WhisperModel(args["model"], device=device, compute_type="float16" if device == "cuda" else "int8",
                         download_root=args["root"])
    lang = args["language"]
    segs, info = model.transcribe(
        np.load(args["pcm"]), language=None if lang == "auto" else lang, beam_size=5, word_timestamps=True,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
        condition_on_previous_text=False, temperature=0.0)
    out = []
    for s in segs:
        text = (s.text or "").strip()
        if not text or (s.no_speech_prob or 0) > 0.75:
            continue
        words = [{"start": round(float(w.start), 3), "end": round(float(w.end), 3), "word": w.word}
                 for w in (s.words or [])]
        out.append({"start": round(float(s.start), 3), "end": round(float(s.end), 3), "text": text,
                    "words": words})
    return out, (info.language or "en")

try:
    import faster_whisper  # noqa: F401
except ImportError:
    print("@@ONYX@@" + json.dumps({"error": "missing"})); sys.exit(0)
device = args["device"]
try:
    segs, lang = run(device)
except RuntimeError as e:
    if device != "cuda":
        raise
    sys.stderr.write("[Onyx Speech] GPU failed (%s) -> CPU\n" % e)
    segs, lang = run("cpu")
    device = "cpu"
print("@@ONYX@@" + json.dumps({"segments": segs, "language": lang, "device": device}))
'''


def _transcribe(pcm: np.ndarray, model_name: str, language: str) -> tuple:
    import subprocess, sys, tempfile
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with _lock, tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pcm.npy")
        np.save(path, pcm.astype(np.float32))
        args = {"model": model_name, "root": _models_dir(), "language": language, "device": device, "pcm": path}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run([sys.executable, "-c", _WORKER], input=json.dumps(args), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", creationflags=flags)
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("@@ONYX@@")), None)
    if line is None:
        tail = "\n".join((p.stderr or "").strip().splitlines()[-15:])
        raise RuntimeError(f"[Onyx Speech] Whisper process failed:\n{tail}")
    res = json.loads(line[len("@@ONYX@@"):])
    if res.get("error") == "missing":
        raise RuntimeError("[Onyx Speech] faster-whisper is not installed. Run once:\n"
                           "  python_embeded\\python.exe -m pip install faster-whisper\n"
                           "then restart ComfyUI.")
    if device == "cuda" and res.get("device") == "cpu":
        print("⚠️  [Onyx Speech] Whisper ran on the CPU (CUDA 12 libraries missing). For the GPU, install them "
              "OUTSIDE site-packages (there they break torch's cuDNN):\n"
              f"  python_embeded\\python.exe -m pip install --target \"{os.path.join(_models_dir(), 'cuda12')}\" "
              "nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\"")
    return res["segments"], res["language"]


def _frames_b64(images, times, fps) -> list:
    if images is None or not len(times):
        return []
    from PIL import Image
    out = []
    for t in times[:8]:
        i = min(images.shape[0] - 1, max(0, int(round(t * fps))))
        arr = (images[i, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        im = Image.fromarray(arr)
        im.thumbnail((640, 640))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        out.append((t, buf.getvalue()))
    return out


def _wav_bytes(pcm: np.ndarray) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


_SPEAKER_SYSTEM = """You label who speaks each line of a transcript. You get the audio of a short video,
a few frames of it (each with its time), and the transcript lines with their times.
For every line decide:
- speaker: "woman" when it is the voice of the woman shown on camera, otherwise who it is
  ("man", "woman 2", "child", "narrator"...), judged from the VOICE;
- on_camera: true only when the speaker is the person visible in the frames AND her lips move
  with these words; an off-camera voice is false.
Never change the words. Return ONLY JSON: {"lines": [{"id": 1, "speaker": "woman", "on_camera": true}, ...]}"""


def _vertex_speakers(segs, pcm, frames, folder, model_id) -> list:
    from google import genai
    from google.genai import types
    from google.oauth2 import service_account
    folder = (folder or "").strip().strip('"')
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json"))
    if not files:
        raise RuntimeError(f"no service-account .json in {folder}")
    path = files[hash(threading.get_ident()) % len(files)]
    project = json.load(open(path, encoding="utf-8")).get("project_id", "")
    creds = service_account.Credentials.from_service_account_file(
        path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    client = genai.Client(vertexai=True, project=project, location="global", credentials=creds)
    parts = [types.Part.from_bytes(data=_wav_bytes(pcm), mime_type="audio/wav")]
    for t, jpg in frames:
        parts.append(types.Part.from_text(text=f"Frame at {t:.2f} s:"))
        parts.append(types.Part.from_bytes(data=jpg, mime_type="image/jpeg"))
    lines = "\n".join(f"{i + 1}. [{s['start']:.2f}-{s['end']:.2f} s] {s['text']}" for i, s in enumerate(segs))
    parts.append(types.Part.from_text(text="Transcript lines:\n" + lines))
    resp = client.models.generate_content(
        model=model_id or "gemini-3.6-flash", contents=parts,
        config=types.GenerateContentConfig(system_instruction=_SPEAKER_SYSTEM, temperature=0.1,
                                           response_mime_type="application/json"))
    data = json.loads(re.sub(r"^```(?:json)?|```$", "", (resp.text or "").strip()))
    return data.get("lines") or []


def _readable(t: dict) -> str:
    rows = [f"language: {t.get('language', '?')} · {len(t.get('segments', []))} line(s)"]
    for s in t.get("segments", []):
        who = s.get("speaker", "woman") + ("" if s.get("on_camera", True) else " (off camera)")
        rows.append(f"[{s['start']:6.2f}-{s['end']:6.2f}] {who}: {s['text']}")
    return "\n".join(rows)


class OnyxSpeechTranscribe:
    """Whisper transcript of the source audio, with who speaks each line."""

    DESCRIPTION = ("Accurate speech-to-text of the source audio (Whisper large-v3, word timings), then who "
                   "speaks each line: the woman on camera or an off-camera voice (Gemini, from the voices and a "
                   "few frames). Feed `transcript` to Onyx H3 Window Dialogue inside the Relay loop. "
                   "Needs faster-whisper (pip install faster-whisper).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The source audio, on the same timeline as the source video."}),
                "whisper_model": (_MODELS, {"default": "large-v3",
                                            "tooltip": "large-v3 = most accurate · large-v3-turbo = faster."}),
                "language": (_LANGS, {"default": "auto"}),
                "speakers": (_SPEAKER_MODES, {"default": "gemini (voices + frames)",
                                              "tooltip": "gemini: Vertex AI labels each line (her, or an "
                                                         "off-camera voice) · one speaker: every line is hers "
                                                         "· off: no speaker labels."}),
                "vertex_json_folder": ("STRING", {"default": "C:\\Comfyui\\json"}),
                "gemini_model": ("STRING", {"default": "gemini-3.6-flash"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "The source frames (helps Gemini see whose lips move)."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("transcript", "text")
    OUTPUT_TOOLTIPS = ("transcript — timed lines + speakers (JSON) → Onyx H3 Window Dialogue.",
                       "text — the transcript, readable (→ Show Text).")
    FUNCTION = "run"
    CATEGORY = "Onyx/H3"

    def run(self, audio, whisper_model, language, speakers, vertex_json_folder, gemini_model,
            images=None, fps=24.0):
        from .onyx_render_profile import ensure_profile_ready
        ensure_profile_ready()
        pcm = _audio_16k_mono(audio)
        key = hashlib.sha256(pcm[::7].tobytes() + f"{whisper_model}|{language}|{speakers}".encode()).hexdigest()[:20]
        path = os.path.join(_CACHE_DIR, key + ".json")
        if os.path.isfile(path):
            t = json.load(open(path, encoding="utf-8"))
            print(f"♻️  [Onyx Speech] transcript from cache ({len(t['segments'])} line(s))")
            return (json.dumps(t, ensure_ascii=False), _readable(t))
        segs, lang = _transcribe(pcm, whisper_model, language)
        for s in segs:
            s["speaker"], s["on_camera"] = "woman", True
        if segs and speakers.startswith("gemini"):
            try:
                frames = _frames_b64(images, [(s["start"] + s["end"]) / 2 for s in segs], float(fps or 24))
                labels = {int(x.get("id", 0)): x for x in _vertex_speakers(segs, pcm, frames, vertex_json_folder,
                                                                           gemini_model)}
                for i, s in enumerate(segs, 1):
                    lab = labels.get(i) or {}
                    s["speaker"] = str(lab.get("speaker") or "woman")[:30]
                    s["on_camera"] = bool(lab.get("on_camera", s["speaker"] == "woman"))
            except Exception as e:
                print(f"⚠️  [Onyx Speech] speaker labels failed ({e}) — every line is given to her.")
        t = {"language": lang, "segments": segs}
        os.makedirs(_CACHE_DIR, exist_ok=True)
        json.dump(t, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"🎙️  [Onyx Speech] {len(segs)} line(s), language {lang}\n" + _readable(t))
        return (json.dumps(t, ensure_ascii=False), _readable(t))


def _ts(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60):02d}:{sec % 60:06.3f}"


class OnyxH3WindowDialogue:
    """Inserts the lines spoken inside the current H3 Relay window into that window's prompt."""

    DESCRIPTION = ("Inside the H3 Relay loop: takes the transcript, keeps the words spoken in THIS window "
                   "(history included), times them from the window's start and inserts them into the prompt "
                   "in H3's dialogue format — her lines lip-synced, off-camera lines not mouthed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "tooltip": "The window's prompt (e.g. from H3 Context-IR)."}),
                "transcript": ("STRING", {"forceInput": True, "tooltip": "From Onyx Speech Transcribe."}),
                "window": ("H3_RELAY_WINDOW", {"tooltip": "H3 Relay Window Inputs → window."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "include_history": ("BOOLEAN", {"default": True,
                                                "tooltip": "The window's latent starts with the history frames "
                                                           "(reference history on): time the lines from there."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    OUTPUT_TOOLTIPS = ("prompt — the window's prompt with its own dialogue → H3 conditioning prompt.",)
    FUNCTION = "run"
    CATEGORY = "Onyx/H3"

    def run(self, prompt, transcript, window, fps=24.0, include_history=True):
        try:
            t = json.loads(transcript or "{}")
        except ValueError:
            return (prompt,)
        w = window
        if isinstance(w, dict) and "windows" in w and "source_start" not in w:
            return (prompt,)
        start, size = int(w["source_start"]), int(w["window_frames"])
        hist = int(w.get("history_frames", 0) or 0)
        past = max(0, hist - 1) if include_history else 0
        t0, t1 = (start - past) / fps, (start + size) / fps
        lang = _LANG_NAMES.get(t.get("language", "en"), (t.get("language") or "English").capitalize())
        lines, others = [], {}
        for s in t.get("segments", []):
            words = [x for x in s.get("words", []) if x["end"] > t0 and x["start"] < t1]
            if not words and not (s["end"] > t0 and s["start"] < t1):
                continue
            text = "".join(x["word"] for x in words).strip() if words else s["text"]
            if not text:
                continue
            at = (words[0]["start"] if words else s["start"]) - t0
            if s.get("on_camera", True) and s.get("speaker", "woman") == "woman":
                lines.append(f"At {_ts(at)}, <Subject 1> (S1) says: <d>[{lang}] {text}</d>")
            else:
                who = s.get("speaker", "man")
                sid = others.setdefault(who, f"S{len(others) + 2}")
                lines.append(f"At {_ts(at)}, an off-camera {who} voice ({sid}) says: <d>[{lang}] {text}</d> "
                             f"<Subject 1> does not mouth these words.")
        if not lines:
            return (prompt,)
        block = ("Dialogue in this clip, timed from its first frame, spoken exactly once each: "
                 + " ".join(lines))
        print(f"🗣️  [Onyx H3 Dialogue] window {w.get('index', '?')}: {len(lines)} line(s)")
        m = re.search(r"\n\s*overall_soundscape\s*:", prompt or "")
        if m:
            return (prompt[:m.start()].rstrip() + "\n" + block + "\n" + prompt[m.start():],)
        return ((prompt or "").rstrip() + "\n" + block,)


NODE_CLASS_MAPPINGS = {"OnyxSpeechTranscribe": OnyxSpeechTranscribe, "OnyxH3WindowDialogue": OnyxH3WindowDialogue}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxSpeechTranscribe": "Onyx Speech Transcribe (Whisper)",
                              "OnyxH3WindowDialogue": "Onyx H3 Window Dialogue"}
