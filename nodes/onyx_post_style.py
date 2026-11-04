# -*- coding: utf-8 -*-
"""
Onyx Post Style — choose the post-processing look once, upstream.

Outputs ONYX_POST: wire it to the `onyx_post` input of every Onyx
post-processing node (Post-Process Suite, FameGrid, Realism Injector, Camera
Look, Renoise, More JPEG, Camera Shake, LUT, Clarity, Dehaze, Skin Tone...).
With a look other than "none", each of them takes the preset's values instead
of its own widgets. Onyx Studio has the same choice in its Post tab and the
same output; this node is for workflows without the Studio.
"""

from __future__ import annotations

from ..utils import onyx_post as _post


class OnyxPostStyle:
    """Pick a post-processing look (+ motion) that overrides every Onyx post node it is wired to."""

    DESCRIPTION = ("One choice for the whole post-processing chain: crispy, clean, amateur, night, flash "
                   "(+ camera motion). Wire `post` to the onyx_post input of the post nodes.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "look": (_post.look_names(), {"default": "none",
                         "tooltip": "\n".join(f"{k}: {v}" for k, v in _post.LOOK_INFO.items())}),
                "motion": (_post.MOTIONS, {"default": "none",
                           "tooltip": "Camera Shake preset. none = no shake; manual = the Camera Shake node "
                                      "keeps its own settings; auto = the one given to `auto_motion` "
                                      "(e.g. Onyx Studio's post_preset, chosen by the skill)."}),
            },
            "optional": {
                "auto_motion": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = (_post.POST_TYPE, "STRING")
    RETURN_NAMES = ("post", "summary")
    FUNCTION = "build"
    CATEGORY = "Onyx/Post"

    def build(self, look, motion, auto_motion=None):
        post = _post.make(look, motion, auto_motion or "")
        summary = f"look: {post['look']} · motion: {post['motion']} · nodes: {', '.join(sorted(post['nodes'])) or '-'}"
        return (post, summary)


NODE_CLASS_MAPPINGS = {"OnyxPostStyle": OnyxPostStyle}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxPostStyle": "Onyx Post Style"}
