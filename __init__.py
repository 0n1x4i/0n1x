"""Onyx Glass Lab — standalone preview pack for the Onyx glass node theme.

One demo node, one JS theme file. Installable next to the real Onyx pack with
zero overlap: different class names, different extension name, different CSS
ids, different window namespace. Delete the folder when done evaluating.
"""

from .nodes.demo import OnyxGlassLabDemo

NODE_CLASS_MAPPINGS = {"OnyxGlassLabDemo": OnyxGlassLabDemo}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxGlassLabDemo": "Onyx Glass Lab — Demo"}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
