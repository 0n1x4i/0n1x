"""
ComfyUI node for selecting fixed prompts or example prompts from text files.
Prompts are stored in the prompts/ directory, separated by '---'.
"""

import logging
import os


from .nodes.onyx_render_profile import ensure_profile_ready
_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

_FIXED_PROMPTS = {
    "Simple Pose and Style Transfer": "image 2 posing and dressing like image 1",
}


def _prompt_sources(wait: float = 0.0) -> dict:
    """{file name: text}. The factory library is protected data from the Worker
    (onyx_data_core PROMPTS); .txt files you add to prompts/ are merged on top."""
    sources = {}
    try:
        from .utils import onyx_remote_data as _rd
        sources.update({k: v for k, v in (_rd.get("PROMPTS", {}, wait=wait) or {}).items()
                        if isinstance(v, str)})
    except Exception:
        pass
    if os.path.isdir(_PROMPTS_DIR):
        for f in os.listdir(_PROMPTS_DIR):
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(_PROMPTS_DIR, f), "r", encoding="utf-8") as fh:
                        sources[f] = fh.read()
                except Exception as e:
                    logging.warning("PromptSelector: failed to read %s: %s", f, e)
    return sources


def _get_prompt_files(sources=None):
    """Prompt files, with Pose and Style Transfer Prompts first."""
    sources = _prompt_sources() if sources is None else sources
    return ["Pose and Style Transfer Prompts"] + sorted(sources)


_built = {"key": None, "maps": ({}, ["(no prompts found)"], {})}


def _load_all_prompts(wait: float = 0.0):
    """prompt_map {label: text}, all_labels, by_file {file_stem: [label, ...]}."""
    sources = _prompt_sources(wait)
    key = tuple(sorted((k, len(v)) for k, v in sources.items()))
    if key == _built["key"]:
        return _built["maps"]
    prompt_map, all_labels, by_file = {}, [], {}
    counter = 1
    for fname in sorted(sources):
        file_stem = os.path.splitext(fname)[0]
        file_labels = []
        for block in sources[fname].split("---"):
            text = block.strip()
            if text:
                preview = text[:60].replace("\n", " ")
                if len(text) > 60:
                    preview += "..."
                label = f"[{file_stem}] {counter}. {preview}"
                prompt_map[label] = text
                all_labels.append(label)
                file_labels.append(label)
                counter += 1
        if file_labels:
            by_file[file_stem] = file_labels
    if not all_labels:
        all_labels = ["(no prompts found)"]
    _built.update(key=key, maps=(prompt_map, all_labels, by_file))
    return _built["maps"]


# Build fixed prompt labels
_FIXED_LABELS = list(_FIXED_PROMPTS.keys())


def _all_labels(wait: float = 0.0):
    return _FIXED_LABELS + _load_all_prompts(wait)[1]


# Serve prompt data to frontend (unique routes for Monthly pack)
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/prompt_selector_monthly/get_prompts")
    async def _get_prompts_monthly_api(request):
        combined = dict(_FIXED_PROMPTS)
        combined.update(_load_all_prompts(wait=5.0)[0])
        return web.json_response(combined)

    @PromptServer.instance.routes.get("/prompt_selector_monthly/get_prompts_by_file")
    async def _get_prompts_by_file_monthly_api(request):
        result = {"Pose and Style Transfer Prompts": _FIXED_LABELS}
        result.update(_load_all_prompts(wait=5.0)[2])
        return web.json_response(result)
except Exception:
    logging.warning("PromptSelector: could not register API routes (server not available)")


class PromptSelectorNode:
    """Select from fixed prompts or pre-written example prompts stored in text files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_file": (_get_prompt_files(_prompt_sources(wait=3.0)),),
                "example_prompt": (_all_labels(wait=3.0),),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select_prompt"
    CATEGORY = "utils"

    def select_prompt(self, prompt_file, example_prompt, prompt_text=""):
        ensure_profile_ready()
        if prompt_text.strip():
            return (prompt_text.strip(),)
        # Check fixed prompts first
        if example_prompt in _FIXED_PROMPTS:
            return (_FIXED_PROMPTS[example_prompt],)
        full_text = _load_all_prompts(wait=15.0)[0].get(example_prompt, "")
        return (full_text,)
