# Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
# Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
"""
Onyx H3 Lock Source Audio — generate the picture against the REAL audio.

MiniMax H3 generates video and audio together. A target-audio guide only
*suggests* the source sound: with turbo LoRAs and step caches the model drifts
(repeated words, missing syllables), and the lips follow that drifted audio,
not the real one.

This node writes the source audio, encoded with the H3 audio VAE, into the
audio half of the window latent and gives it a noise mask of 0: the sampler
re-imposes the real audio (at the current noise level) on every step, like
inpainting. The video is denoised against the exact original soundtrack, frame
for frame, in every window, and the window's audio output IS the source audio.

Wire it right after H3 Relay Apply History (its LATENT), with the window's
`reference_audio` (same span as the latent: history + window) and the audio VAE.
Everything that reads that latent (guide, sampling memory, sampler) reads this
node's output instead.
"""

from __future__ import annotations

import logging

import torch

from .onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('h3_tools_bundle', 'onyx_h3_audio_lock', __package__)
    return ns


LOG = logging.getLogger("onyx.h3_audio_lock")


def _resample(*args, **kwargs):
    return _onyx_core()['_resample'](*args, **kwargs)



class OnyxH3LockSourceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The H3 window latent (video + audio), e.g. from "
                                                 "H3 Relay Apply History."}),
                "audio": ("AUDIO", {"tooltip": "Source audio covering the SAME span as the latent: "
                                               "H3 Relay Window Inputs -> reference_audio."}),
                "audio_vae": ("VAE", {"tooltip": "The MiniMax H3 audio VAE."}),
                "enabled": ("BOOLEAN", {"default": True,
                                        "tooltip": "Off = latent passed through unchanged (the model "
                                                   "generates its own audio)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "lock"
    CATEGORY = "Onyx/H3"
    DESCRIPTION = ("Locks the window's audio to the real source audio (encoded, noise mask 0): the picture "
                   "and the lips are generated against the original soundtrack, not a regenerated one.")

    @staticmethod
    def _ensure_av_masks(*args, **kwargs):
        return _onyx_core()['OnyxH3LockSourceAudio']._ensure_av_masks(*args, **kwargs)


    def lock(self, *args, **kwargs):
        return _onyx_core()['OnyxH3LockSourceAudio'].lock(self, *args, **kwargs)



NODE_CLASS_MAPPINGS = {"OnyxH3LockSourceAudio": OnyxH3LockSourceAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxH3LockSourceAudio": "Onyx H3 Lock Source Audio"}
