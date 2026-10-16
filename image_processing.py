"""
ComfyUI nodes for image post-processing.
- Remove Image Metadata: strip all metadata and re-save as clean JPEG
- Save as Phone Photo: re-save with realistic phone camera EXIF
"""

import logging
import os
import math
import random
from datetime import datetime, timedelta

import numpy as np
from PIL import Image
from PIL.ExifTags import IFD
from PIL.TiffImagePlugin import IFDRational
import folder_paths


# ---------- Directory helpers ----------

from .nodes.onyx_render_profile import ensure_profile_ready

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'image_processing', __package__)
    return ns

_VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _load_images_from_folder(*args, **kwargs):
    return _onyx_core()['_load_images_from_folder'](*args, **kwargs)



# ---------- Remove Image Metadata ----------

class MetadataRemoveNode:
    """Batch-strip all metadata from images in a folder (EXIF, XMP, PNG chunks, ICC, C2PA)."""

    def __init__(self, *args, **kwargs):
        return _onyx_core()['MetadataRemoveNode'].__init__(self, *args, **kwargs)


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (folder_paths.get_input_subfolders(),),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "output_folder": ("STRING", {"default": "cleaned"}),
                "quality": ("INT", {"default": 93, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "add_noise": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "remove_metadata"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def remove_metadata(self, *args, **kwargs):
        return _onyx_core()['MetadataRemoveNode'].remove_metadata(self, *args, **kwargs)



# ---------- Save as Phone Photo ----------

_PHONE_PROFILES = {
    "iPhone 15 Pro Max": {
        "make": "Apple", "model": "iPhone 15 Pro Max",
        "software": "17.5.1",
        "focal_length": (6765, 1000), "f_number": (178, 100),
        "iso_range": (50, 800), "exposure_range": (1, 4000),
        "lens_model": "iPhone 15 Pro Max back triple camera 6.765mm f/1.78",
        "resolutions": {"12MP": (4032, 3024), "48MP": (8064, 6048)},
    },
    "iPhone 16 Pro Max": {
        "make": "Apple", "model": "iPhone 16 Pro Max",
        "software": "18.3.1",
        "focal_length": (6765, 1000), "f_number": (178, 100),
        "iso_range": (50, 1600), "exposure_range": (1, 4000),
        "lens_model": "iPhone 16 Pro Max back quad camera 6.765mm f/1.78",
        "resolutions": {"12MP": (4032, 3024), "48MP": (8064, 6048)},
    },
    "Samsung Galaxy S24 Ultra": {
        "make": "samsung", "model": "SM-S928B",
        "software": "S928BXXS4AXL1",
        "focal_length": (6300, 1000), "f_number": (170, 100),
        "iso_range": (50, 800), "exposure_range": (1, 3000),
        "lens_model": "SM-S928B",
        "resolutions": {"12MP": (4000, 3000), "50MP": (8160, 6120)},
    },
    "Samsung Galaxy S25 Ultra": {
        "make": "samsung", "model": "SM-S938B",
        "software": "S938BXXU1AXK5",
        "focal_length": (6300, 1000), "f_number": (170, 100),
        "iso_range": (50, 1000), "exposure_range": (1, 4000),
        "lens_model": "SM-S938B",
        "resolutions": {"12MP": (4000, 3000), "50MP": (8160, 6120)},
    },
}

_PHONE_CHOICES = list(_PHONE_PROFILES.keys())

_RESOLUTION_CHOICES = ["12MP", "48MP/50MP", "Original"]

_GPS_LOCATIONS = {
    "None": None,
    "New York": (40.7128, -74.0060, 10),
    "Los Angeles": (34.0522, -118.2437, 71),
    "London": (51.5074, -0.1278, 11),
    "Tokyo": (35.6762, 139.6503, 40),
    "Paris": (48.8566, 2.3522, 35),
    "Sydney": (-33.8688, 151.2093, 58),
    "Dubai": (25.2048, 55.2708, 5),
    "Random US City": "random",
}

_RANDOM_US_CITIES = [
    (40.7128, -74.0060, 10),    # New York
    (34.0522, -118.2437, 71),   # Los Angeles
    (41.8781, -87.6298, 181),   # Chicago
    (29.7604, -95.3698, 15),    # Houston
    (33.4484, -112.0740, 331),  # Phoenix
    (29.4241, -98.4936, 198),   # San Antonio
    (32.7157, -117.1611, 20),   # San Diego
    (30.2672, -97.7431, 149),   # Austin
    (37.7749, -122.4194, 16),   # San Francisco
    (47.6062, -122.3321, 56),   # Seattle
    (39.7392, -104.9903, 1609), # Denver
    (35.2271, -80.8431, 229),   # Charlotte
    (36.1627, -86.7816, 182),   # Nashville
    (25.7617, -80.1918, 2),     # Miami
    (38.9072, -77.0369, 22),    # Washington DC
]

_GPS_CHOICES = list(_GPS_LOCATIONS.keys())


def _decimal_to_dms(*args, **kwargs):
    return _onyx_core()['_decimal_to_dms'](*args, **kwargs)



def _build_phone_exif(*args, **kwargs):
    return _onyx_core()['_build_phone_exif'](*args, **kwargs)



class SaveAsPhonePhotoNode:
    """Batch-save images from a folder as JPEG with realistic phone camera EXIF metadata."""

    def __init__(self, *args, **kwargs):
        return _onyx_core()['SaveAsPhonePhotoNode'].__init__(self, *args, **kwargs)


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (folder_paths.get_input_subfolders(),),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "phone_model": (_PHONE_CHOICES,),
                "resolution": (_RESOLUTION_CHOICES, {"default": "12MP"}),
                "gps_location": (_GPS_CHOICES, {"default": "Random US City"}),
                "output_folder": ("STRING", {"default": "phone_photos"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "add_noise": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_phone_photo"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_phone_photo(self, *args, **kwargs):
        return _onyx_core()['SaveAsPhonePhotoNode'].save_phone_photo(self, *args, **kwargs)
