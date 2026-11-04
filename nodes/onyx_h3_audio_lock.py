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

LOG = logging.getLogger("onyx.h3_audio_lock")


def _resample(waveform, rate, target):
    if rate == target:
        return waveform
    try:
        import torchaudio
        return torchaudio.functional.resample(waveform, rate, target)
    except Exception:
        import comfy.audio
        return comfy.audio.resample(waveform, rate, target)


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
    def _ensure_av_masks():
        """Per-stream (video / audio) denoise masks for H3: native in recent ComfyUI, otherwise
        enabled by the H3 Relay pack's compatibility layer (same one its masked modes use)."""
        import sys
        for name, mod in list(sys.modules.items()):
            if name.endswith("vendor.context_loop.masked_context") and hasattr(mod, "_require_h3_mask_support"):
                mod._require_h3_mask_support()
                return
        LOG.warning("[Onyx H3 Audio Lock] H3 Relay pack not found: relying on native ComfyUI AV masks.")

    def lock(self, latent, audio, audio_vae, enabled=True):
        if not enabled:
            return (latent,)
        from comfy.nested_tensor import NestedTensor
        self._ensure_av_masks()

        samples = latent["samples"]
        if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
            raise ValueError("[Onyx H3 Audio Lock] expected the native H3 video/audio latent pair.")
        video, target_audio = samples.tensors
        steps = int(target_audio.shape[-1])

        rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        wave = _resample(audio["waveform"], int(audio["sample_rate"]), rate)[:1]
        # Same call as H3 Relay's own audio encode: [B, C, T] -> [B, T, C].
        encoded = audio_vae.encode(wave.movedim(1, -1))
        enc_steps = int(encoded.shape[-1])
        # reference_audio and the latent both END at the window's last frame (the latent may or
        # may not include the history span at its start): align on the end.
        if enc_steps > steps:
            encoded = encoded[..., -steps:]
        elif enc_steps < steps:                  # rounding at the edges: hold the first step
            pad = encoded[..., :1].expand(*encoded.shape[:-1], steps - enc_steps)
            encoded = torch.cat([pad, encoded], -1)
        if enc_steps < steps - 2:
            LOG.warning("[Onyx H3 Audio Lock] source audio gave %d latent steps for %d expected: "
                        "check that `audio` is the window's reference_audio.", enc_steps, steps)
        encoded = encoded.to(device=target_audio.device, dtype=target_audio.dtype)
        if encoded.shape[1:] != target_audio.shape[1:]:
            raise ValueError(f"[Onyx H3 Audio Lock] encoded audio {tuple(encoded.shape)} does not match "
                             f"the H3 audio latent {tuple(target_audio.shape)}.")
        encoded = encoded.expand_as(target_audio).clone()

        old = latent.get("noise_mask")
        if old is not None and getattr(old, "is_nested", False):
            video_mask = old.tensors[0]
        else:
            video_mask = torch.ones_like(video)
        audio_mask = torch.zeros_like(target_audio)
        print(f"🔒 [Onyx H3 Audio Lock] window audio locked to the source ({steps} latent steps)")
        return ({**latent, "samples": NestedTensor([video, encoded]),
                 "noise_mask": NestedTensor([video_mask, audio_mask])},)


NODE_CLASS_MAPPINGS = {"OnyxH3LockSourceAudio": OnyxH3LockSourceAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxH3LockSourceAudio": "Onyx H3 Lock Source Audio"}
