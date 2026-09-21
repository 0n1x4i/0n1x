# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
Onyx Save Image No Metadata
Saves images exactly like ComfyUI's native SaveImage node but without
embedding the workflow JSON or prompt metadata into the file.
Supports PNG and JPEG output.
"""

import os
import logging
import numpy as np
from PIL import Image
import folder_paths


class OnyxSaveImageNoMetadataNode:

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Prefix for saved filenames. Supports ComfyUI date tokens like %date:yyyy-MM-dd%.",
                }),
                "format": (["PNG", "JPEG"], {
                    "default": "PNG",
                    "tooltip": "Output format. PNG is lossless. JPEG is smaller but lossy.",
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "JPEG quality (1-100). Ignored for PNG.",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Onyx/Image"

    def save_images(self, images, filename_prefix="ComfyUI", format="PNG", quality=95):
        # Resolve output path and counter (same logic as native SaveImage)
        ensure_profile_ready()
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                images[0].shape[1],
                images[0].shape[0],
            )

        results = []
        for batch_idx in range(images.shape[0]):
            # Tensor [H, W, 3 ou 4] float32 0-1 -> uint8 PIL. Le nombre de
            # canaux decide le mode plutot que "RGB" code en dur : le VAE de
            # Qwen Image 2.1 est RGBA natif (comfy/sd.py detecte ce VAE via
            # decoder.head.2.weight.shape[0] et sort 4 canaux, "opaque alpha
            # for RGB input"). Forcer mode="RGB" sur un buffer a 4 canaux
            # decale chaque pixel d'un octet, le decalage s'accumule le long
            # de la ligne, et le sujet se retrouve reproduit ~4/3 fois sur la
            # largeur avec des rayures.
            img_t = images[batch_idx]
            if not img_t.is_contiguous():
                img_t = img_t.contiguous()
            np_img = (img_t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mode = "RGBA" if np_img.shape[-1] == 4 else "RGB"
            pil_img = Image.fromarray(np_img, mode=mode)

            if format == "JPEG":
                ext = "jpg"
                file = f"{filename}_{counter:05}_.{ext}"
                if pil_img.mode == "RGBA":
                    # JPEG ne supporte pas l'alpha ; sans cette conversion PIL
                    # leve une exception a la sauvegarde.
                    pil_img = pil_img.convert("RGB")
                pil_img.save(
                    os.path.join(full_output_folder, file),
                    "JPEG",
                    quality=quality,
                    optimize=True,
                )
            else:
                ext = "png"
                file = f"{filename}_{counter:05}_.{ext}"
                # No pnginfo= argument -> zero metadata embedded
                pil_img.save(
                    os.path.join(full_output_folder, file),
                    compress_level=self.compress_level,
                )

            logging.info("[Onyx SaveNoMeta] Saved %s/%s", subfolder or "output", file)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })
            counter += 1

        return {"ui": {"images": results}}
