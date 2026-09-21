"""Wan Animate 2 — arbitrary-length generation in a single node.
Features auto-segmentation, Replacement Mode, and an optional Low VRAM mode.

The segment chain itself (generate()'s body) lives server-side — see
nodes/onyx_remote_exec.py. This file keeps only the ComfyUI scaffolding
(INPUT_TYPES / RETURN_TYPES).
"""

import comfy.samplers

from .onyx_render_profile import ensure_profile_ready
from .onyx_remote_exec import load_remote


class OnyxAnimate2Infinity:
    """Generate a Wan Animate 2 video of any length with flat VRAM use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Animate 2 UNet."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "segment_length": ("INT", {
                    "default": 81, "min": 5, "max": 321, "step": 4,
                    "tooltip": "Frames generated per segment, snapped to 4k+1."}),
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Hard cap on total output frames. 0 = follow pose_video to end."}),
                "vary_seed_per_segment": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF = same seed every segment. ON = adds segment index to seed."}),
                "decode_tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Decode each segment in tiles to save VRAM."}),
                # --- LOW VRAM OPTIMIZATION SWITCH ---
                "low_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "ON = Purges CLIP / TextEncoder from VRAM before sampling. Saves 3-6 GB VRAM to prevent OOM."}),
                # --- REPLACEMENT MODE SWITCH ---
                "replacement_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True = Keeps original background. Composites the generated character back onto original background."}),
                "mask_dilation": ("INT", {
                    "default": 4, "min": -100, "max": 100, "step": 1,
                    "tooltip": "Expand (+) or contract (-) mask boundary."}),
                "mask_feather": ("INT", {
                    "default": 8, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Blur mask edges for a seamless blend."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "The character to animate."}),
                "pose_video": ("IMAGE", {"tooltip": "Frames carrying the motion."}),
                "pose_video_mask": ("IMAGE", {"tooltip": "Black & white mask of the character."}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "positive_pose": ("CONDITIONING",),
                "clip_vision_output_pose": ("CLIP_VISION_OUTPUT",),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "info")
    FUNCTION = "generate"
    CATEGORY = "Onyx/Video"
    DESCRIPTION = ("Wan Animate 2 at any length. Runs segment chain internally so peak VRAM stays flat.")

    def generate(self, model, positive, negative, vae, width, height, seed, steps, cfg,
                 sampler_name, scheduler, segment_length, max_frames,
                 vary_seed_per_segment, decode_tiled, low_vram, replacement_mode,
                 mask_dilation, mask_feather,
                 reference_image=None, pose_video=None, pose_video_mask=None,
                 clip_vision_output=None, positive_pose=None, clip_vision_output_pose=None,
                 pose_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0,
                 reference_image_strength=1.0):

        ensure_profile_ready()
        ns = load_remote("onyx_animate2_infinity_core")
        return ns["generate_impl"](
            model, positive, negative, vae, width, height, seed, steps, cfg,
            sampler_name, scheduler, segment_length, max_frames,
            vary_seed_per_segment, decode_tiled, low_vram, replacement_mode,
            mask_dilation, mask_feather,
            reference_image, pose_video, pose_video_mask,
            clip_vision_output, positive_pose, clip_vision_output_pose,
            pose_strength, pose_start_percent, pose_end_percent,
            reference_image_strength,
        )


NODE_CLASS_MAPPINGS = {
    "OnyxAnimate2Infinity": OnyxAnimate2Infinity,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxAnimate2Infinity": "Onyx Animate 2 Infinity",
}
