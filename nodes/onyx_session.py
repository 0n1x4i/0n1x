"""
Onyx Session — the one place a user pastes their key.

OUTPUT_NODE = True on purpose: ComfyUI always executes an OUTPUT_NODE on every
queue, even with nothing wired to its output (exactly how Save Image / Preview
Image run without being connected to anything downstream). So this node just
needs to exist ANYWHERE in the workflow — no wiring to the other nodes needed —
and it refreshes the local sync cache every single run. Every other gated node
then reads that cache on its own; see onyx_render_profile.py.

The key is NOT kept in the workflow any more (a saved or shared .json used to
carry it in clear). The node's text box only takes it once: the front-end
(js/onyx_session.js) sends it to /onyx_session/key, it is written to a local
file and the box is emptied, and the widget never serializes a value. Where
the key is read from, first match wins:
  1. ONYX_LICENSE_KEY environment variable (servers, RunPod serverless)
  2. <ComfyUI user dir>/onyx_session.key   (or this pack's folder if unknown)
  3. a key still stored in an old workflow  -> saved to the file, then used
"""

import os

from .onyx_render_profile import sync_with_server

_ENV = "ONYX_LICENSE_KEY"
_FILE_NAME = "onyx_session.key"


def _key_path():
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:  # noqa: BLE001 - outside ComfyUI / old versions
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, _FILE_NAME)


def read_saved_key():
    env = (os.environ.get(_ENV) or "").strip()
    if env:
        return env
    try:
        with open(_key_path(), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def save_key(key):
    key = (key or "").strip()
    path = _key_path()
    if not key:
        try:
            os.remove(path)
        except OSError:
            pass
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(key)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def masked(key):
    key = (key or "").strip()
    return (key[:9] + "…" + key[-3:]) if len(key) > 14 else ("saved" if key else "")


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:  # noqa: BLE001 - not running inside ComfyUI
        return
    if getattr(PromptServer.instance, "_onyx_session_routes_registered", False):
        return
    PromptServer.instance._onyx_session_routes_registered = True

    @PromptServer.instance.routes.get("/onyx_session/key")
    async def _status(request):
        key = read_saved_key()
        return web.json_response({"saved": bool(key), "masked": masked(key),
                                  "from_env": bool((os.environ.get(_ENV) or "").strip())})

    @PromptServer.instance.routes.post("/onyx_session/key")
    async def _set(request):
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            data = {}
        key = str(data.get("key") or "").strip()
        if len(key) > 200:
            return web.json_response({"ok": False, "error": "too long"}, status=400)
        save_key(key)
        return web.json_response({"ok": True, "saved": bool(key), "masked": masked(key)})


_register_routes()


class OnyxSessionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Paste your Onyx key once: it is saved on this computer only and "
                               "never stored in the workflow. Leave empty afterwards.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Onyx"
    DESCRIPTION = "Paste your Onyx key here once (saved locally, never in the workflow). Runs on every queue."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Doit re-tourner a chaque queue pour rafraichir le cache, meme si le
        # widget key n'a pas change.
        return float("nan")

    def run(self, key=""):
        saved = read_saved_key()
        typed = (key or "").strip()
        if not saved and typed:
            # old workflow that still carries a key: move it to the local file once
            save_key(typed)
            saved = typed
        result = sync_with_server(saved)
        if result["valid"]:
            status = "✅ Onyx session active"
        elif not saved:
            status = "🔴 No Onyx key on this computer — paste it in this node"
        else:
            status = f"🔴 Onyx session inactive ({result['reason']})"
        print(f"[Onyx Session] {status}")
        return {"ui": {"text": [status]}, "result": (status,)}


NODE_CLASS_MAPPINGS = {"OnyxSessionNode": OnyxSessionNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxSessionNode": "Onyx Session"}
