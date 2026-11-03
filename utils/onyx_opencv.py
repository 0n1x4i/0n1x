# -*- coding: utf-8 -*-
"""
OpenCV duplicate-install check — warns, and fixes only when the user asks.

The problem: opencv-python, opencv-python-headless, opencv-contrib-python and
opencv-contrib-python-headless all install the same `cv2` folder. pip lets
several coexist, their files get mixed, and `import cv2` then fails or loads a
half-broken module.

The pack used to fix this on its own at every start (pip uninstall + install,
no question asked). That changed the user's Python environment behind their
back, and could remove the contrib build another custom node pack relies on.

Now:
  1. At startup: detection only (no pip call), and a clear console message.
  2. The frontend (js/onyx_opencv_check.js) shows a dialog with a "Fix" button.
  3. "Fix" schedules the repair for the NEXT ComfyUI start (marker file), where
     it runs from the very top of the pack's __init__ — before anything has
     imported cv2. On Windows a loaded .pyd is locked by the OS, so repairing
     inside a running process leaves orphan folders; next start is the only
     reliable moment. If cv2 is not loaded yet in this process, it can also
     run immediately.
  4. The repair keeps ONE build and the same features: a contrib build if any
     contrib build was installed (other packs may need cv2.face / ximgproc),
     headless otherwise, pinned to the newest version already installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_PACK_ROOT, "_cache")
_FIX_MARKER = os.path.join(_CACHE_DIR, "opencv_fix_requested")
_DISMISS_FILE = os.path.join(_CACHE_DIR, "opencv_warning_dismissed.json")

_last_result: dict = {}


def installed() -> dict:
    """{distribution name (lowercase): version} for every opencv* distribution."""
    try:
        import importlib.metadata as md
    except Exception:
        return {}
    found = {}
    try:
        for d in md.distributions():
            name = (d.metadata["Name"] or "").strip()
            if name.lower().startswith("opencv"):
                found[name.lower()] = d.version
    except Exception:
        return {}
    return found


def _version_key(v: str):
    parts = []
    for p in str(v).split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    return parts


def plan(dists: dict | None = None) -> dict:
    """What a repair would do. Empty 'uninstall' = nothing to fix."""
    dists = installed() if dists is None else dists
    if len(dists) <= 1:
        return {"uninstall": [], "install": None}
    keep = ("opencv-contrib-python-headless" if any("contrib" in n for n in dists)
            else "opencv-python-headless")
    newest = max(dists.values(), key=_version_key)
    return {"uninstall": sorted(dists), "install": f"{keep}=={newest}", "fallback": keep}


def manual_command(p: dict) -> str:
    if not p.get("uninstall"):
        return ""
    py = f'"{sys.executable}"' if " " in sys.executable else sys.executable
    return (f"{py} -m pip uninstall -y {' '.join(p['uninstall'])}\n"
            f"{py} -m pip install {p['install']}")


def _dismissed_for(dists: dict) -> bool:
    try:
        with open(_DISMISS_FILE, encoding="utf-8") as fh:
            return sorted(json.load(fh).get("dists", [])) == sorted(dists)
    except (OSError, ValueError):
        return False


def status() -> dict:
    dists = installed()
    p = plan(dists)
    return {
        "conflict": bool(p["uninstall"]),
        "installed": dists,
        "plan": p,
        "command": manual_command(p),
        "cv2_loaded": "cv2" in sys.modules,
        "fix_scheduled": os.path.exists(_FIX_MARKER),
        "dismissed": _dismissed_for(dists),
        "last_result": _last_result,
    }


def run_fix(p: dict | None = None) -> dict:
    """Uninstall every opencv build, install the single one from plan()."""
    global _last_result
    p = p or plan()
    if not p["uninstall"]:
        _last_result = {"ok": True, "message": "Nothing to fix."}
        return _last_result
    print(f"[Onyx] OpenCV: repairing duplicate installs ({', '.join(p['uninstall'])}) "
          f"-> {p['install']} ...")
    try:
        r1 = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *p["uninstall"]],
                            timeout=180, capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-m", "pip", "install", p["install"], "--quiet"],
                            timeout=300, capture_output=True, text=True)
        if r2.returncode != 0 and p.get("fallback"):
            # The pinned version may not exist for the kept variant: take the latest.
            r2 = subprocess.run([sys.executable, "-m", "pip", "install", p["fallback"], "--quiet"],
                                timeout=300, capture_output=True, text=True)
        ok = r2.returncode == 0
        msg = ("OpenCV repaired — restart ComfyUI if cv2 was already loaded."
               if ok else f"pip install failed: {(r2.stderr or r1.stderr or '')[-400:]}")
    except Exception as e:
        ok, msg = False, f"Repair failed: {e}"
    print(f"[Onyx] OpenCV: {msg}")
    _last_result = {"ok": ok, "message": msg}
    return _last_result


def startup_check() -> None:
    """Called at the very top of the pack's __init__. Never calls pip unless the
    user asked for the repair (marker written by the dialog's Fix button)."""
    p = plan()
    if os.path.exists(_FIX_MARKER):
        try:
            os.remove(_FIX_MARKER)
        except OSError:
            pass
        if p["uninstall"]:
            if "cv2" in sys.modules:
                print("[Onyx] OpenCV: repair requested, but another pack loaded cv2 before Onyx. "
                      "Run the commands below with ComfyUI closed:\n" + manual_command(p))
            else:
                run_fix(p)
        return
    if p["uninstall"]:
        print("\n[Onyx] ⚠️  Several OpenCV builds are installed at once "
              f"({', '.join(p['uninstall'])}).\n"
              "        They share the same cv2 folder and break each other. Nothing was changed.\n"
              "        Fix: click 'Fix at next start' in the dialog shown in ComfyUI, or run with "
              "ComfyUI closed:\n        " + manual_command(p).replace("\n", "\n        ") + "\n")


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    inst = PromptServer.instance
    if getattr(inst, "_onyx_opencv_routes", False):
        return
    inst._onyx_opencv_routes = True

    @inst.routes.get("/onyx/opencv/status")
    async def _status(_request):
        return web.json_response(status())

    @inst.routes.post("/onyx/opencv/fix")
    async def _fix(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        mode = body.get("mode", "next_start")
        if mode == "now" and "cv2" not in sys.modules:
            import asyncio
            result = await asyncio.get_running_loop().run_in_executor(None, run_fix)
            return web.json_response({"scheduled": False, **result})
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_FIX_MARKER, "w") as fh:
            fh.write("requested from the ComfyUI dialog\n")
        return web.json_response({"scheduled": True, "ok": True,
                                  "message": "The repair will run at the next ComfyUI start."})

    @inst.routes.post("/onyx/opencv/dismiss")
    async def _dismiss(_request):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_DISMISS_FILE, "w", encoding="utf-8") as fh:
            json.dump({"dists": sorted(installed())}, fh)
        return web.json_response({"ok": True})
