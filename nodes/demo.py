"""Demo node for the Onyx Glass Lab theme preview.

Carries one of every common widget type plus an IMAGE / MASK / STRING / INT
slot mix so the slot colouring is visible, and a `simulate_sync_error` toggle
that raises the same SS-204 error text the real pack raises, so the red
"glass error" state can be previewed without a licence setup.
"""

import torch


class OnyxGlassLabDemo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["generate", "edit", "faceswap"], {"default": "generate"}),
                "prompt": ("STRING", {"multiline": True, "default": "a candid phone photo, soft window light"}),
                "strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 150}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "keep_metadata": ("BOOLEAN", {"default": False}),
                "simulate_sync_error": ("BOOLEAN", {"default": False,
                                                    "tooltip": "Raise the SS-204 sync error to preview the red glass state."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "latent": ("LATENT",),
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("image", "mask", "prompt", "seed")
    FUNCTION = "run"
    CATEGORY = "Onyx/Glass Lab"
    DESCRIPTION = "Preview node for the glass theme. Does nothing useful."

    def run(self, mode, prompt, strength, steps, seed, keep_metadata,
            simulate_sync_error, image=None, mask=None, latent=None, model=None):
        if simulate_sync_error:
            raise RuntimeError(
                "Sync error SS-204: place the Onyx Session node anywhere in this "
                "workflow and run it once, then retry."
            )
        if image is None:
            image = torch.zeros((1, 64, 64, 3))
        if mask is None:
            mask = torch.zeros((1, 64, 64))
        return (image, mask, f"[{mode}] {prompt}", seed)
