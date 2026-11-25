# -*- coding: utf-8 -*-
# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
"""Auto-pick for BFS Video Segment Picker (bfsnodes).

With nothing selected, the BFS H3 Image character swap stops silently after the Picker (its
conditioning returns an ExecutionBlocker), so a queue ends "idle" before any generation.
When nothing is selected and SAM 3 finds exactly ONE subject on the first frame, that subject
is selected as red (subject 1) for the run, so a single queue goes all the way. Two or more
subjects: unchanged, pick them in the panel. The node's own selection widget is not touched.

Applied to the BFS class at the first queue (bfsnodes loads after this pack), never edits the
bfsnodes files: a bfsnodes update keeps working, and any failure leaves the Picker as it is.
"""

from __future__ import annotations

import json
import sys

_PATCHED = {"done": False}


def _has_pick(parse_frames, selection: str) -> bool:
    for frame_state in parse_frames(selection).values():
        for key, sub in frame_state.items():
            if str(key).isdigit() and isinstance(sub, dict) and (sub.get("selected") or []):
                return True
    return False


def _auto_selection(module, kw: dict) -> str | None:
    selection = kw.get("selection") or "{}"
    try:
        meta = json.loads(selection) if selection.strip() else {}
    except ValueError:
        return None
    if not isinstance(meta, dict) or meta.get("segment_only"):
        return None
    if _has_pick(module.parse_frames, selection):
        return None
    image = kw.get("image")
    if image is None or getattr(image, "ndim", 0) != 4:
        return None
    sp = module.sp
    spec = {"concepts": ", ".join(sp.split_concepts(kw.get("concepts", ""))),
            "threshold": round(float(kw.get("threshold", 0.4)), 4),
            "max_per_concept": int(kw.get("max_per_concept", 8)),
            "min_area": round(float(kw.get("min_area", 0.002)), 5)}
    frame = image[0:1, ..., :3].float().cpu()
    # Same session key as the Picker's own first frame: SAM 3 runs once, the Picker reuses it.
    _sid, sess = sp.session_for((kw.get("sam3_model"), kw.get("sam3_clip")), frame, spec)
    found = [o for o in sess.get("auto", []) if not o.get("remove")]
    if len(found) != 1:
        if len(found) > 1:
            print(f"[Onyx Auto-pick] {len(found)} subjects found: select them in the Picker, then Apply fix.")
        return None
    frames = meta.get("frames") if isinstance(meta.get("frames"), dict) else {}
    frames["0"] = {"1": {"selected": [found[0]["id"]], "added": []}}
    meta["frames"] = frames
    print(f"[Onyx Auto-pick] one subject found ({found[0].get('label', 'subject')}): selected as red.")
    return json.dumps(meta)


def patch() -> bool:
    if _PATCHED["done"]:
        return True
    try:
        import nodes as comfy_nodes
        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("BFSVideoSegmentPicker")
        if cls is None:
            return False
        module = sys.modules.get(cls.__module__)
        if module is None or not hasattr(module, "parse_frames") or not hasattr(module, "sp"):
            return False
        original = cls.pick
        if getattr(original, "_onyx_autopick", False):
            _PATCHED["done"] = True
            return True

        def pick(self, *args, **kw):
            if not args:
                try:
                    new = _auto_selection(module, kw)
                    if new is not None:
                        kw["selection"] = new
                except Exception as e:  # never break the Picker
                    print(f"[Onyx Auto-pick] skipped ({e!r})")
            return original(self, *args, **kw)

        pick._onyx_autopick = True
        cls.pick = pick
        _PATCHED["done"] = True
        print("[Onyx Auto-pick] BFS Video Segment Picker: one subject = selected automatically.")
        return True
    except Exception as e:
        print(f"[Onyx Auto-pick] not applied ({e!r})")
        _PATCHED["done"] = True
        return False


def _on_prompt(json_data):
    if not _PATCHED["done"]:
        patch()
    return json_data


def register() -> None:
    try:
        from server import PromptServer
        PromptServer.instance.add_on_prompt_handler(_on_prompt)
    except Exception as e:
        print(f"[Onyx Auto-pick] not registered ({e!r})")
