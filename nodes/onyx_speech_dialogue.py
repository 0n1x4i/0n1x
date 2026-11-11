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

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('h3_tools_bundle', 'onyx_speech_dialogue', __package__)
    return ns


_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(os.path.dirname(_HERE), "_cache", "transcripts")
_MODELS = ["large-v3", "large-v3-turbo", "medium", "small"]
_LANGS = ["auto", "en", "fr", "es", "pt", "de", "it", "nl", "ru", "ja", "ko", "zh", "ar", "tr", "pl"]
_SPEAKER_MODES = ["gemini (voices + frames)", "one speaker (all hers)", "off"]
_ENGINES = ["elevenlabs scribe v2 (API)", "whisper (local)"]
_lock = threading.Lock()
_whisper = {"name": None, "model": None}


def _models_dir(*args, **kwargs):
    return _onyx_core()['_models_dir'](*args, **kwargs)



def _audio_16k_mono(*args, **kwargs):
    return _onyx_core()['_audio_16k_mono'](*args, **kwargs)





def _transcribe(*args, **kwargs):
    return _onyx_core()['_transcribe'](*args, **kwargs)



def _frames_b64(*args, **kwargs):
    return _onyx_core()['_frames_b64'](*args, **kwargs)



def _wav_bytes(*args, **kwargs):
    return _onyx_core()['_wav_bytes'](*args, **kwargs)





def _vertex_speakers(*args, **kwargs):
    return _onyx_core()['_vertex_speakers'](*args, **kwargs)



def _scribe_key(*args, **kwargs):
    return _onyx_core()['_scribe_key'](*args, **kwargs)



def _scribe(*args, **kwargs):
    return _onyx_core()['_scribe'](*args, **kwargs)



def _readable(*args, **kwargs):
    return _onyx_core()['_readable'](*args, **kwargs)



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
                "engine": (_ENGINES, {"default": _ENGINES[0],
                                      "tooltip": "elevenlabs scribe v2: ElevenLabs' speech-to-text API — very "
                                                 "accurate words and timings, and it separates the voices "
                                                 "(paid per minute of audio). whisper: local, free, less exact."}),
                "elevenlabs_api_key": ("STRING", {"default": "", "multiline": False,
                                                  "tooltip": "Paste it once: it is then kept on this computer "
                                                             "(Onyx keys.json) and the field can be emptied, so "
                                                             "the key never travels with the workflow. "
                                                             "ELEVENLABS_API_KEY also works."}),
                "num_speakers": ("INT", {"default": 0, "min": 0, "max": 32,
                                         "tooltip": "Scribe: how many different voices at most (0 = auto). "
                                                    "1 for a single person talking to camera."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("transcript", "text")
    OUTPUT_TOOLTIPS = ("transcript — timed lines + speakers (JSON) → Onyx H3 Window Dialogue.",
                       "text — the transcript, readable (→ Show Text).")
    FUNCTION = "run"
    CATEGORY = "Onyx/H3"

    def run(self, *args, **kwargs):
        return _onyx_core()['OnyxSpeechTranscribe'].run(self, *args, **kwargs)



def _ts(*args, **kwargs):
    return _onyx_core()['_ts'](*args, **kwargs)



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

    def run(self, *args, **kwargs):
        return _onyx_core()['OnyxH3WindowDialogue'].run(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {"OnyxSpeechTranscribe": OnyxSpeechTranscribe, "OnyxH3WindowDialogue": OnyxH3WindowDialogue}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxSpeechTranscribe": "Onyx Speech Transcribe (Whisper)",
                              "OnyxH3WindowDialogue": "Onyx H3 Window Dialogue"}
