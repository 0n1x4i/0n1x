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
import random
import numpy as np
from PIL import Image
import folder_paths

from . import metadata_spoof as _spoof


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
            # Ajoutes en "optional" et en dernier : l'ordre des widgets des
            # workflows deja sauvegardes ne bouge pas.
            "optional": {
                "metadata_profile": (_spoof.IMAGE_PROFILES, {
                    "default": _spoof.PROFILE_NONE,
                    "tooltip": "None = no metadata at all (as before).\n"
                               "iOS (iPhone) = saved as JPEG with a full, fresh iPhone EXIF per "
                               "image: device, iOS version, date, optics, exposure and GPS. "
                               "Forces JPEG — a PNG carrying iPhone EXIF would contradict itself.",
                }),
                "iphone_model": (_spoof.IPHONE_CHOICES, {
                    "default": "random",
                    "tooltip": "iOS profile only. 'random' picks a model per image.",
                }),
                "country": (_spoof.COUNTRY_CODES, {
                    "default": "US",
                    "tooltip": "iOS profile only. GPS position and timezone are drawn in this country.",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Onyx/Image"

    def save_images(self, images, filename_prefix="ComfyUI", format="PNG", quality=95,
                    metadata_profile=_spoof.PROFILE_NONE, iphone_model="random", country="US"):
        spoof_ios = metadata_profile == _spoof.PROFILE_IOS
        if spoof_ios:
            format = "JPEG"   # an iPhone EXIF only makes sense in a JPEG
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
                if spoof_ios:
                    desc = _spoof.spoof_jpeg(
                        os.path.join(full_output_folder, file), iphone_model, country,
                        seed=random.getrandbits(64),
                        width=pil_img.width, height=pil_img.height)
                    logging.info("[%s SaveNoMeta] iOS EXIF: %s", "Onyx", desc)
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
