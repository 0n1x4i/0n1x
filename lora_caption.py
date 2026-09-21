"""
ComfyUI node — Onyx Lora Caption Generator.
Captions every image in a folder through Gemini (API key), Vertex (service
account) or Grok, one request per image.
By default the images are left untouched and each caption is written to
<image name>.txt beside it. Switch keep_original_names off to renumber the
folder to <trigger>_0001.ext with a matching <trigger>_0001.txt.

The captioning logic itself lives server-side (see nodes/onyx_remote_exec.py):
only the ComfyUI scaffolding — INPUT_TYPES, widget lists, the REST routes'
plumbing — stays in this file.
"""

import os
import logging
import folder_paths


from .nodes.onyx_render_profile import ensure_profile_ready
from .nodes.onyx_remote_exec import load_remote


def _list_input_subdirs():
    """List subdirectories inside ComfyUI's input folder."""
    input_dir = folder_paths.get_input_directory()
    dirs = []
    for f in os.listdir(input_dir):
        full = os.path.join(input_dir, f)
        if os.path.isdir(full):
            dirs.append(f)
    dirs.sort(key=str.lower)
    if not dirs:
        dirs = ["(no folders found)"]
    return dirs


_PROVIDERS = ["gemini", "vertex", "grok"]

# Meme liste que gemini_prompt.py, pour que les deux nodes du pack proposent les
# memes modeles. L'ancienne s'arretait a la 2.0, qui ne repond plus.
_GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

_GROK_MODELS = [
    "grok-4-1-fast-non-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4.20-0309-reasoning",
]

_CAPTION_MODES = ["sdxl"]


# ---------- Serve prompts to frontend (so the JS can populate the textbox) ----------
#
# Note this route already sends the sdxl instruction text to the browser for
# preview, same as before this change — protecting it server-side stops the
# .py file from carrying it, it does not add anything against someone reading
# this route's response in devtools while using the node normally. Different
# problem than the one remote-exec solves.

try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/lora_caption/get_prompts")
    async def _lora_caption_get_prompts(request):
        try:
            ns = load_remote("lora_caption_core")
            return web.json_response({"sdxl": ns["_SDXL_CAPTION_INSTRUCTION"]})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @PromptServer.instance.routes.post("/lora_caption/rename_files")
    async def _lora_caption_rename_files(request):
        """Renumber a folder on demand, from the node's button.

        Not part of process_folder: renaming is destructive and must happen when
        the user asks for it, not as a side effect of running the graph.
        """
        try:
            data = await request.json()
            folder = (data.get("folder") or "").strip()
            trigger = (data.get("trigger") or "").strip()

            if not folder or folder == "(no folders found)":
                return web.json_response({"error": "no input folder selected"}, status=400)
            if not trigger:
                return web.json_response({"error": "trigger_word is empty"}, status=400)

            root = os.path.realpath(folder_paths.get_input_directory())
            directory = os.path.realpath(os.path.join(root, folder))
            # La route est joignable par n'importe qui sur le reseau : sans ce
            # test, un "../.." renommerait des fichiers hors de ComfyUI/input.
            if directory != root and not directory.startswith(root + os.sep):
                return web.json_response({"error": "folder outside ComfyUI/input"}, status=400)
            if not os.path.isdir(directory):
                return web.json_response({"error": f"not a directory: {folder}"}, status=400)

            ns = load_remote("lora_caption_core")
            renamed, captions = ns["rename_folder"](directory, trigger)
            return web.json_response({"renamed": renamed, "captions": captions,
                                      "folder": folder})
        except Exception as e:
            logging.exception("Lora Caption Generator: rename failed")
            return web.json_response({"error": str(e)}, status=500)
except Exception:
    logging.warning("Lora Caption Generator: could not register prompt route (server not available)")


class OnyxLoraCaptionGeneratorNode:
    """Generate LoRA training captions for all images in a folder.

    Images are sent one at a time. With keep_original_names on (default) nothing
    is renamed and each caption lands in <image name>.txt beside its image; with
    it off, images are renumbered to <trigger>_0001.ext with a matching .txt.
    """

    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": (_list_input_subdirs(), {"default": _list_input_subdirs()[0]}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "provider": (_PROVIDERS, {"default": "gemini"}),
                "trigger_word": ("STRING", {"default": "", "multiline": False}),
                "mode": (_CAPTION_MODES, {
                    "default": "sdxl",
                    "tooltip": (
                        "sdxl: caption style for SDXL person LoRA training "
                        "(no face/hair/identity, focus on outfit/scene/lighting/framing)."
                    ),
                }),
            },
            "optional": {
                "gemini_model": (_GEMINI_MODELS, {"default": "gemini-3.6-flash"}),
                "vertex_json_folder": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Folder of service-account .json files, for provider=vertex. "
                               "The first one is used; api_key is ignored in that mode."}),
                "keep_original_names": ("BOOLEAN", {
                    "default": True,
                    "label_on": "keep original filenames",
                    "label_off": "rename to <trigger>_0001",
                    "tooltip": "On: images are left untouched and each caption is written as "
                               "<image name>.txt beside it.\n"
                               "Off: the historical behaviour — images are renamed to "
                               "<trigger>_0001.ext with a matching .txt."}),
                "grok_model": (_GROK_MODELS, {"default": "grok-4-1-fast-non-reasoning"}),
                "caption_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional override. If empty, mode determines the prompt.",
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 8192, "step": 64}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "process_folder"
    OUTPUT_NODE = True
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def process_folder(self, input_folder, api_key, provider, trigger_word,
                       mode="sdxl",
                       gemini_model="gemini-3.6-flash",
                       vertex_json_folder="", keep_original_names=True,
                       grok_model="grok-4-1-fast-non-reasoning",
                       caption_instruction="", temperature=0.7,
                       max_tokens=8192):
        ensure_profile_ready()
        if not input_folder or input_folder == "(no folders found)":
            raise RuntimeError("Lora Caption Generator: no input folder selected. Create a subfolder in ComfyUI/input/.")

        directory = os.path.join(folder_paths.get_input_directory(), input_folder)
        if not os.path.isdir(directory):
            raise RuntimeError(f"Lora Caption Generator: '{directory}' is not a valid directory.")

        if provider != "vertex":
            if api_key.strip():
                OnyxLoraCaptionGeneratorNode._cached_api_key = api_key.strip()
            api_key = api_key.strip() or OnyxLoraCaptionGeneratorNode._cached_api_key
            if not api_key:
                raise RuntimeError("Lora Caption Generator: API key is required.")

        ns = load_remote("lora_caption_core")
        return ns["process_folder_impl"](
            directory, api_key, provider, trigger_word, mode, gemini_model,
            vertex_json_folder, keep_original_names, grok_model,
            caption_instruction, temperature, max_tokens,
        )
