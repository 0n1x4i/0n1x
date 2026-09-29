# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
ComfyUI node - Onyx Grok Prompt Generator.
Calls the xAI Grok API directly (api.x.ai) - no OpenRouter middleman.
Get your API key at: https://console.x.ai
Same image+text message format as OnyxGeminiPromptNode._call_grok() in gemini_prompt.py,
kept consistent so both nodes talk to the API the same way.

The actual API call (message building, request, usage logging) lives server-side
— see nodes/onyx_remote_exec.py. This file keeps the ComfyUI scaffolding
(INPUT_TYPES, model list, the presets REST routes) local.
"""

import json
import os
import logging
import random
import re

from .onyx_remote_exec import load_remote


# Tirage aleatoire {a|b|c} dans le prompt, resolu ICI avant l'envoi a Grok.
# Pourquoi cote node et pas dans le prompt : chaque appel a Grok est
# independant (aucune memoire des images precedentes), donc "ne repete pas le
# meme lieu" ne peut pas marcher — le modele retombe toujours sur son choix le
# plus probable. Le hasard doit venir de l'exterieur. Tirage cale sur la seed :
# seed fixe = memes tirages, seed randomize = nouveaux tirages a chaque run.
# Seuls les blocs contenant au moins un "|" sont touches, donc un JSON ou des
# accolades normales dans le prompt restent intacts.
_CHOICE_RE = re.compile(r"\{([^{}]*\|[^{}]*)\}")


def _resolve_choices(text, seed):
    rng = random.Random(int(seed or 0))
    picks = []

    def _pick(match):
        options = [o.strip() for o in match.group(1).split("|")]
        options = [o for o in options if o] or [""]
        choice = rng.choice(options)
        picks.append(choice)
        return choice

    # Plusieurs passes pour les blocs imbriques ({a|{b|c}}), de l'interieur vers l'exterieur.
    for _ in range(5):
        new_text = _CHOICE_RE.sub(_pick, text)
        if new_text == text:
            break
        text = new_text
    return text, picks


# Vision-capable models (support image input) first, text-only models after.
# Same list as _GROK_MODELS in gemini_prompt.py for consistency across the pack.
_GROK_MODELS = [
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-2-vision-1212",
    "grok-3",
    "grok-3-fast",
    "grok-3-mini",
    "grok-3-mini-fast",
]

# ─────────────────────────────────────────────────────────────────────────────
# Presets — meme mecanique que OnyxNanoBananaAIO : un JSON a cote du node, deux
# routes REST, et le JS qui appelle. La cle API n'est JAMAIS sauvegardee.
# ─────────────────────────────────────────────────────────────────────────────
_PRESETS_FILE = os.path.join(os.path.dirname(__file__), "grok_prompt_presets.json")

_PRESET_KEYS = [
    "prompt",
    "trigger_word",
    "model",
    "temperature",
    "max_tokens",
]


def _load_presets() -> dict:
    if not os.path.isfile(_PRESETS_FILE):
        return {}
    try:
        with open(_PRESETS_FILE, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        print(f"⚠️  [Grok Presets] Unable to load: {e}")
        return {}


def _save_presets(presets: dict) -> bool:
    try:
        with open(_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in presets.items()}, f,
                      indent=2, ensure_ascii=False)
        print(f"✅ [Grok Presets] Saved → {_PRESETS_FILE}")
        return True
    except Exception as e:
        print(f"❌ [Grok Presets] Write failed: {e}")
        return False


try:
    from server import PromptServer
    from aiohttp import web

    # Garde contre le double enregistrement : une copie de sauvegarde du .py
    # dans le meme dossier ferait lever aiohttp sur une route deja prise.
    if not getattr(PromptServer.instance, "_onyx_grok_prompt_routes_registered", False):
        PromptServer.instance._onyx_grok_prompt_routes_registered = True

        @PromptServer.instance.routes.post("/onyx_grok_prompt/save_preset")
        async def _grok_save_preset(request):
            try:
                body = await request.json()
                slot = int(body.get("slot", 1))
                if slot < 1 or slot > 5:
                    return web.json_response(
                        {"success": False, "error": "Invalid slot (1-5)"}, status=400)
                filtered = {k: v for k, v in (body.get("params") or {}).items()
                            if k in _PRESET_KEYS}
                presets = _load_presets()
                presets[slot] = filtered
                if not _save_presets(presets):
                    return web.json_response(
                        {"success": False, "error": "File write failed"}, status=500)
                print(f"💾 [Grok Presets] Slot {slot} saved")
                return web.json_response({"success": True, "slot": slot,
                                          "params": filtered})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.get("/onyx_grok_prompt/load_presets")
        async def _grok_load_presets(request):
            try:
                return web.json_response({
                    "success": True,
                    "presets": {str(k): v for k, v in _load_presets().items()},
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=500)
except Exception:
    logging.warning("Onyx Grok: could not register preset routes (server not available)")


class OnyxGrokPromptNode:

    _cached_api_key = ""

    # Nombre d'appels API depuis le demarrage de ComfyUI.
    #
    # Ajoute pour une raison precise : cette fonction ne fait qu'UN POST par
    # execution, donc si la facture xAI montre N requetes pour ce qui semble
    # etre un seul run, c'est que ComfyUI a appele generate() N fois. Le
    # compteur rend ca visible dans la console au lieu de le laisser deviner
    # depuis les logs de facturation, des heures plus tard.
    #
    # Volontairement PAS de IS_CHANGED ici, contrairement a OnyxGeminiPromptNode qui
    # renvoie float("nan") : NaN != NaN, donc ce node-la se re-execute a CHAQUE
    # queue meme a entrees identiques. Sur une API facturee au token, c'est un
    # gouffre. Sans IS_CHANGED, ComfyUI hache les entrees et reutilise le cache
    # tant que rien ne change — ce qui est le comportement voulu ici.
    _api_call_count = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The prompt/instruction to send to the model.",
                }),
                "model": (_GROK_MODELS, {
                    "default": "grok-4.20-0309-reasoning",
                    "tooltip": "xAI Grok model. Vision-capable models are listed first - use one of those if you connect an image.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image (e.g. from the Onyx Image Batch Loader for batch runs). Sent alongside the prompt to vision-capable models.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "xAI API key (console.x.ai). Cached after first use.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Creativity of the response (0 = deterministic, 2 = very creative).",
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Maximum number of tokens in the response.",
                }),
                # Ajoute EN DERNIER volontairement : ComfyUI serialise les
                # valeurs par position, donc l'inserer plus haut decalerait tous
                # les widgets des workflows deja enregistres.
                "trigger_word": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Replaces every occurrence of TRIGGER in the prompt. "
                               "Left empty, TRIGGER is sent as-is.",
                }),
                # Aussi en dernier, pour la meme raison de position. N'est PAS
                # envoye a l'API : sa seule fonction est de changer les entrees
                # du node, donc de casser le cache de ComfyUI. En "randomize",
                # chaque run d'un batch rappelle Grok (nouveau prompt a chaque
                # image) ; en "fixed", le prompt est reutilise tant que rien ne
                # change, comme avant.
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "randomize = a new Grok prompt on every run (e.g. every image of a "
                               "batch). fixed = reuse the same prompt while nothing else changes. "
                               "Not sent to the API.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_prompt",)
    FUNCTION = "generate"
    OUTPUT_NODE = False
    CATEGORY = "Onyx/Prompt"
    DESCRIPTION = (
        "Send a text prompt (and optionally an image) to Grok and get a generated prompt back.\n"
        "Plug an Onyx Image Batch Loader into 'image' + Queue All to run one request per image in the batch."
    )

    def generate(
        self,
        prompt,
        model="grok-4.20-0309-reasoning",
        image=None,
        api_key="",
        temperature=0.7,
        max_tokens=1024,
        trigger_word="",
        seed=0,
    ):
        ensure_profile_ready()
        prompt, _picks = _resolve_choices(prompt or "", seed)
        if _picks:
            print(f"🎲 [Onyx Grok] Random picks (seed {seed}): " + " · ".join(p[:60] for p in _picks))
        ns = load_remote("grok_prompt_core")
        return ns["generate_impl"](
            OnyxGrokPromptNode, prompt, model, image, api_key,
            temperature, max_tokens, trigger_word,
        )
