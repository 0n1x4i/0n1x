#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate NODES.md from the nodes' own INPUT_TYPES / RETURN_TYPES.

The hand-written NODES.md documented 12 nodes out of ~70 and had drifted (old
model lists, missing widgets). This builds it from the code, so it is always
complete and always matches what ComfyUI shows.

    python tools/gen_nodes_doc.py            # rewrite NODES.md
    python tools/gen_nodes_doc.py --check    # exit 1 if NODES.md is out of date (CI)

It runs WITHOUT ComfyUI, torch or the network: ComfyUI modules and missing
heavy packages are replaced by inert stubs, only to read the class
declarations. No node is executed and no protected code is fetched (the
registry refresh is disabled with ONYX_REGISTRY_URL=off).

Text sources, per node: DESCRIPTION attribute, else the class docstring (first
paragraph); per input: its "tooltip". A short hand-written intro for a node can
be put in docs/node_notes/<NodeClassName>.md — it is inserted under the title.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "NODES.md")
NOTES_DIR = os.path.join(ROOT, "docs", "node_notes")
PKG = "onyx_pack_doc"


# ─────────────────────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────────────────────

class _Stub:
    """Absorbs any use: attribute, call, decorator, subscript, iteration."""

    def __init__(self, name="stub"):
        self._name = name

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _Stub(f"{self._name}.{item}")

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs and not isinstance(args[0], _Stub):
            return args[0]                      # used as a decorator: keep the function
        return _Stub(f"{self._name}()")

    def __getitem__(self, item):
        return _Stub(f"{self._name}[]")

    def __setitem__(self, key, value):
        pass

    def __contains__(self, item):
        return False

    def keys(self):
        return []

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    def __mro_entries__(self, bases):
        return (object,)                        # `class X(torch.nn.Module)` -> plain class

    def __or__(self, other):
        return self

    __ror__ = __or__

    def __repr__(self):
        return f"<{self._name}>"

    __str__ = __repr__

    def __fspath__(self):
        return tempfile.gettempdir()


class _StubModule(types.ModuleType):
    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _Stub(f"{self.__name__}.{item}")


# Heavy or ComfyUI-only packages the pack imports. Only these are stubbed (when
# not installed): stubbing everything would also fake the optional imports of
# requests & co., which then misbehave.
STUBBED = {
    "torch", "torchvision", "torchaudio", "cv2", "comfy", "comfy_extras", "comfy_api",
    "comfy_execution", "nodes", "node_helpers", "latent_preview", "kornia", "av",
    "mediapipe", "transformers", "accelerate", "huggingface_hub", "google", "fal_client",
    "colour", "pywt", "imageio", "imageio_ffmpeg", "gallery_dl", "yt_dlp", "safetensors",
    "scipy", "skimage", "ultralytics", "onnxruntime", "impact", "aiohttp", "spandrel",
    "einops", "matplotlib", "sageattention", "bitsandbytes", "numba", "librosa", "soundfile",
}


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Last-resort finder for the STUBBED packages missing from this Python."""

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] not in STUBBED:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


def _install_comfy_stubs():
    # An EMPTY folder: dropdowns built from ComfyUI's input/output folders must
    # not list whatever happens to sit in this machine's temp directory.
    tmp = tempfile.mkdtemp(prefix="onyx_doc_")
    class _FolderPaths(types.ModuleType):
        def __getattr__(self, item):        # any other folder_paths helper: empty list
            if item.startswith("__"):
                raise AttributeError(item)
            return lambda *a, **k: []

    fp = _FolderPaths("folder_paths")
    fp.get_input_directory = lambda: tmp
    fp.get_output_directory = lambda: tmp
    fp.get_temp_directory = lambda: tmp
    fp.get_user_directory = lambda: tmp
    fp.get_filename_list = lambda *a, **k: []
    fp.get_full_path = lambda *a, **k: None
    fp.get_folder_paths = lambda *a, **k: []
    fp.models_dir = tmp
    fp.base_path = tmp
    fp.folder_names_and_paths = {}
    fp.supported_pt_extensions = {".safetensors", ".ckpt", ".pt"}
    fp.filter_files_content_types = lambda files, *a, **k: list(files)
    sys.modules["folder_paths"] = fp

    class _Routes:
        def __getattr__(self, _method):
            return lambda *a, **k: (lambda fn: fn)

    class _Instance:
        routes = _Routes()
        app = _Stub("app")

        def send_sync(self, *a, **k):
            pass

        def __getattr__(self, item):
            return _Stub(f"PromptServer.instance.{item}")

    srv = types.ModuleType("server")
    srv.PromptServer = type("PromptServer", (), {"instance": _Instance()})
    sys.modules["server"] = srv
    sys.meta_path.append(_StubFinder())


# ─────────────────────────────────────────────────────────────────────────────
# Loading the pack
# ─────────────────────────────────────────────────────────────────────────────

def load_pack():
    os.environ["ONYX_REGISTRY_URL"] = "off"
    _install_comfy_stubs()
    spec = importlib.util.spec_from_file_location(
        PKG, os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = mod
    import contextlib
    import io
    # ComfyUI's own top-level modules ("nodes", "server"...) must not resolve to
    # the pack's folders of the same name when the pack root is on sys.path
    # (running from the pack folder, or from the tests).
    saved_path = sys.path[:]
    root = os.path.normcase(os.path.abspath(ROOT))
    sys.path[:] = [p for p in sys.path
                   if os.path.normcase(os.path.abspath(p or os.getcwd())) != root]
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # the pack's startup banner
            spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _md(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>").strip()


def _type_and_details(spec):
    if not isinstance(spec, (tuple, list)) or not spec:
        return str(spec), {}
    t = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(t, _Stub):
        return "Dropdown (list filled by ComfyUI)", opts
    if isinstance(t, (list, tuple)):
        values = [str(v) for v in t]
        shown = ", ".join(f"`{v}`" for v in values[:12])
        if len(values) > 12:
            shown += f", … ({len(values)} options)"
        return f"Dropdown: {shown}" if values else "Dropdown", opts
    return f"`{t}`", opts


def _default(opts):
    if "default" not in opts:
        return "—"
    d = opts["default"]
    if isinstance(d, str):
        if not d:
            return '`""`'
        if len(d) > 40 or "\n" in d:
            return "(long text)"
        return f"`{d}`"
    return f"`{d}`"


def _range(opts):
    if "min" in opts or "max" in opts:
        lo, hi = opts.get("min", "…"), opts.get("max", "…")
        if isinstance(hi, int) and hi >= 2 ** 32:
            hi = "max"
        s = f"{lo} – {hi}"
        if "step" in opts:
            s += f" (step {opts['step']})"
        return s
    return ""


_INTERNAL_WORDS = ("protected", "proxy", "remote", "worker", "licence", "license", "_impl")


def _description(cls) -> str:
    desc = getattr(cls, "DESCRIPTION", None)
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    doc = inspect.getdoc(cls) or ""
    if not doc or cls.__doc__ is None:
        return ""
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    # Implementation notes are for maintainers, not for the user documentation.
    if any(w in first.lower() for w in _INTERNAL_WORDS):
        return ""
    return first


def _anchor(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else ("-" if ch in " -" else "") for ch in text)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def render(mappings: dict, names: dict) -> str:
    rows = []
    for key, cls in mappings.items():
        rows.append((str(getattr(cls, "CATEGORY", "misc")), names.get(key, key), key, cls))
    rows.sort(key=lambda r: (r[0].lower(), r[1].lower()))

    out = [
        "# Onyx pack — node reference",
        "",
        "<!-- GENERATED by tools/gen_nodes_doc.py from the nodes' INPUT_TYPES. "
        "Do not edit by hand: edit the tooltips in the code (or docs/node_notes/<Class>.md) "
        "and re-run the script. -->",
        "",
        f"{len(rows)} nodes. Inputs, defaults and ranges below are read from the code, so they "
        "match exactly what ComfyUI shows. Hover a widget in ComfyUI for the same tooltip.",
        "",
        "## Contents",
        "",
    ]
    current = None
    for cat, disp, key, _cls in rows:
        if cat != current:
            out.append(f"\n**{cat}**\n")
            current = cat
        out.append(f"- [{disp}](#{_anchor(disp)})")
    out.append("")

    for cat, disp, key, cls in rows:
        out += ["---", "", f"## {disp}", "", f"**Class:** `{key}` · **Category:** `{cat}`"]
        if getattr(cls, "OUTPUT_NODE", False):
            out[-1] += " · output node"
        out.append("")
        note = os.path.join(NOTES_DIR, f"{key}.md")
        if os.path.isfile(note):
            with open(note, encoding="utf-8") as fh:
                out += [fh.read().strip(), ""]
        desc = _description(cls)
        if desc:
            out += [_md(desc).replace("<br>", " "), ""]

        try:
            spec = cls.INPUT_TYPES()
        except Exception as e:
            if "Sync error" in str(e):
                out += ["_Inputs are provided by the Onyx server at run time._", ""]
            else:
                out += [f"_Inputs could not be read without ComfyUI ({type(e).__name__})._", ""]
            spec = {}
        lines = []
        for group in ("required", "optional"):
            for name, s in (spec.get(group) or {}).items():
                t, opts = _type_and_details(s)
                tip = opts.get("tooltip", "")
                lines.append(f"| `{name}` | {_md(t)} | {'yes' if group == 'required' else 'no'} | "
                             f"{_md(_default(opts))} | {_md(_range(opts))} | {_md(tip)} |")
        if lines:
            out += ["### Inputs", "",
                    "| Name | Type | Required | Default | Range | Description |",
                    "|------|------|----------|---------|-------|-------------|", *lines, ""]

        rtypes = list(getattr(cls, "RETURN_TYPES", ()) or ())
        rnames = list(getattr(cls, "RETURN_NAMES", ()) or ())
        tips = list(getattr(cls, "OUTPUT_TOOLTIPS", ()) or ())
        if rtypes:
            out += ["### Outputs", "", "| Name | Type | Description |", "|------|------|-------------|"]
            for i, t in enumerate(rtypes):
                n = rnames[i] if i < len(rnames) else str(t).lower()
                tip = tips[i] if i < len(tips) else ""
                tname = "Dropdown" if isinstance(t, (list, tuple)) else str(t)
                out.append(f"| `{n}` | `{tname}` | {_md(tip)} |")
            out.append("")
        else:
            out += ["_No outputs (saves files / side effects only)._", ""]
    return "\n".join(out).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if NODES.md is out of date")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    pack = load_pack()
    text = render(pack.NODE_CLASS_MAPPINGS, pack.NODE_DISPLAY_NAME_MAPPINGS)
    if args.check:
        try:
            with open(args.out, encoding="utf-8") as fh:
                same = fh.read() == text
        except OSError:
            same = False
        if not same:
            print("NODES.md is out of date — run: python tools/gen_nodes_doc.py")
            return 1
        print("NODES.md is up to date.")
        return 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"Wrote {args.out} ({len(pack.NODE_CLASS_MAPPINGS)} nodes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
