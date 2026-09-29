# -*- coding: utf-8 -*-
"""Metadata profiles for the Save (no metadata) nodes.

Only the widget choices live here, so ComfyUI can register the nodes at
startup with no network call. The implementation is fetched at run time via
onyx_remote_exec.load_remote() — only when a profile other than
"None (strip only)" is picked.
"""

from .onyx_remote_exec import load_remote

PROFILE_NONE = "None (strip only)"
PROFILE_IOS = "iOS (iPhone)"
PROFILE_CAPCUT_IOS = "CapCut iOS"
PROFILE_CAPCUT_ANDROID = "CapCut Android"
PROFILE_CAPCUT_PC = "CapCut PC"

IMAGE_PROFILES = [PROFILE_NONE, PROFILE_IOS]
VIDEO_PROFILES = [PROFILE_NONE, PROFILE_IOS, PROFILE_CAPCUT_IOS,
                  PROFILE_CAPCUT_ANDROID, PROFILE_CAPCUT_PC]

IPHONE_CHOICES = [
    "random", "iPhone 12", "iPhone 12 Pro", "iPhone 13", "iPhone 13 Pro",
    "iPhone 14", "iPhone 14 Pro", "iPhone 14 Pro Max", "iPhone 15",
    "iPhone 15 Pro", "iPhone 15 Pro Max", "iPhone 16", "iPhone 16 Pro",
    "iPhone 16 Pro Max", "iPhone 17", "iPhone 17 Pro", "iPhone 17 Pro Max",
    "iPhone Air",
]

COUNTRY_CODES = [
    "US", "AE", "AR", "AT", "AU", "BE", "BR", "CA", "CH", "CL", "CO", "CZ",
    "DE", "DK", "EG", "ES", "FI", "FR", "GB", "GR", "HK", "HU", "ID", "IE",
    "IL", "IN", "IT", "JP", "KR", "MA", "MX", "MY", "NG", "NL", "NO", "NZ",
    "PE", "PH", "PL", "PT", "RO", "SA", "SE", "SG", "TH", "TR", "TW", "UY",
    "VN", "ZA",
]


def video_extension(profile):
    return ".mov" if profile == PROFILE_CAPCUT_IOS else ".mp4"


def spoof_jpeg(path, iphone_model="random", country="US", seed=None, width=0, height=0):
    ns = load_remote("metadata_spoof_core")
    return ns["spoof_jpeg"](path, iphone_model, country, seed=seed, width=width, height=height)


def spoof_video(src, dst, profile, iphone_model="random", country="US", seed=None):
    ns = load_remote("metadata_spoof_core")
    return ns["spoof_video"](src, dst, profile, iphone_model, country, seed=seed)
