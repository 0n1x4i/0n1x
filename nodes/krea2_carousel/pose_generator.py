"""Influencer-style pose-list generation for the Krea2 carousel UI.

API credentials are supplied at request time (or by environment variables) and
are never written into the ComfyUI workflow.  Remote endpoints are fixed to the
official Gemini and xAI hosts; local endpoints are restricted by director.py.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from urllib.parse import quote

import requests
from PIL import Image

from . import local_vlm
from .director import _llamacpp_generate, _local_endpoint, _ollama_generate

from ..onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('krea2_bundle', 'pose_generator', __package__)
    ns['RUNTIME_KEYS'] = RUNTIME_KEYS
    return ns



BACKENDS = ("Transformers local", "Ollama", "llama.cpp", "Gemini API", "Grok API", "Vertex AI")
VERTEX_DEFAULT_MODEL = "gemini-3.6-flash"

# Cles poussees par l'UI du Pose List Prep (route /onyx/nf/runtime_key), pour le
# mode One-step qui genere les poses pendant le run. Memoire du process
# uniquement : rien sur disque, rien dans le workflow.
RUNTIME_KEYS = {}
MAX_POSES = 8
DEFAULT_VIBE = (
    "Confident lifestyle influencer photo carousel: natural candid energy, varied hand gestures, "
    "one strong selfie, one over-the-shoulder turn, and believable social-media posing."
)



def _bounded(value, limit, default=""):
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value or "")).strip()
    return (value or default)[:limit]


def _safe_cloud_model(value, default):
    value = _bounded(value, 160, default)
    value = value.removeprefix("models/")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
        raise ValueError("The model identifier contains characters that are not allowed.")
    return value


def _image_b64(image: Image.Image, max_edge=1280):
    image = image.convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _pose_prompts(*args, **kwargs):
    return _onyx_core()['_pose_prompts'](*args, **kwargs)



def _extract_poses(*args, **kwargs):
    return _onyx_core()['_extract_poses'](*args, **kwargs)



def _gemini_generate(*args, **kwargs):
    return _onyx_core()['_gemini_generate'](*args, **kwargs)



def _grok_generate(*args, **kwargs):
    return _onyx_core()['_grok_generate'](*args, **kwargs)



# Rotation entre les service accounts du dossier, comme les autres nodes Onyx :
# chaque appel prend le JSON suivant, pour repartir la charge sur les projets.
_VERTEX_ROTATION = {"offset": 0}


def _vertex_json_files(*args, **kwargs):
    return _onyx_core()['_vertex_json_files'](*args, **kwargs)



def _vertex_generate(*args, **kwargs):
    return _onyx_core()['_vertex_generate'](*args, **kwargs)



def generate_pose_list(*args, **kwargs):
    return _onyx_core()['generate_pose_list'](*args, **kwargs)



__all__ = ["BACKENDS", "DEFAULT_VIBE", "FALLBACK_POSES", "generate_pose_list", "_extract_poses"]
