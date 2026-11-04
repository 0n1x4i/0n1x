# -*- coding: utf-8 -*-
"""Onyx Krea2 Story Director — one photo in, a whole carousel out.

A vision LLM reads the ORIGINAL photo (and the optional image 2), plans a set of
2 to 8 shots like a photographer (a day story, a lookbook, UGC product shots)
and this node turns every shot into one short Krea2 Identity Edit instruction.

Every instruction is rendered from the SAME original photo (Krea2EditModelPatch
+ Krea2EditGroundedEncode), so the identity never drifts from shot to shot.

The `edit_prompts` output is a LIST: everything after it runs once per shot.

The LLM only writes the plan (place, pose, framing, light, expression...). The
final instruction is assembled here, with the change written FIRST and kept
short: the edit model weighs the start of the instruction most, and long
"keep everything" paragraphs make it copy the reference pose.

Plans are cached on disk (<pack>/_cache/story_plans): the same photo + settings
+ seed never pays for the same plan twice. Change `seed` for a new plan.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
from PIL import Image

MAX_SHOTS = 8
MODES = ("Lifestyle", "Fashion", "UGC")
CAMERAS = ("auto", "smartphone", "professional", "cinematic", "clean")
FACES = ("auto", "natural", "custom")
IMAGE2_KINDS = ("outfit", "product", "person", "body shape")
TIME_SPANS = ("auto", "one moment (same place)", "one hour", "one day", "a trip (several days)")
_SPAN_RULES = {
    "one moment (same place)": ("TIME SPAN: one single moment of about 10 to 20 minutes, ALL shots in the SAME place "
                                "(same room or same spot), same light and same time of day; only her pose, the "
                                "framing and the camera angle change, like a burst of photos taken one after another."),
    "one hour": ("TIME SPAN: about one hour, at most two or three neighbouring places (e.g. a café then the street "
                 "outside), the same time of day and a consistent light."),
    "one day": "TIME SPAN: one day, from morning to night, a different place for each shot.",
    "a trip (several days)": "TIME SPAN: a trip over several days, different places, days and lights.",
}
STYLE_T2I = "Text-to-image (character LoRA)"
STYLE_EDIT = "Krea2 Edit (reference photo)"
STYLES = (STYLE_T2I, STYLE_EDIT)
CAMERA_T2I = {
    "smartphone": ("Casual smartphone photo shot on an iPhone, handheld, natural phone-camera look, slight "
                   "wide-angle, everything in focus, real everyday light."),
    "professional": ("Professional photo shot on a full-frame camera with an 85mm lens, shallow depth of field, "
                     "softly blurred background, crisp detail on her."),
    "cinematic": "Cinematic film still, moody directional light, rich contrast, deep shadows, subtle film grain.",
    "clean": "Natural realistic photo, true-to-life colors.",
}
BACKENDS = ("Vertex AI", "Gemini API", "Grok API", "Transformers local")

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_PACK_ROOT, "_cache", "story_plans")

CAMERA_LINES = {
    "smartphone": ("Shot on a smartphone: a casual handheld iPhone photo, natural phone-camera look, "
                   "slight wide-angle, everything in focus, real everyday light."),
    "professional": ("Shot on a professional full-frame camera with an 85mm lens: shallow depth of field, "
                     "softly blurred background, crisp detail on her."),
    "cinematic": ("Cinematic film still: moody directional light, rich contrast, deep shadows, "
                  "subtle film grain."),
    "clean": "Same photographic look as the reference image: same camera feel, colors and grain.",
}

_MODE_BRIEF = {
    "Lifestyle": (
        "LIFESTYLE: a believable story of her life told in {n} photos ({story}). Each shot is a DIFFERENT real "
        "place that fits the story and her style (cafe, street, boutique, car, rooftop, home, gym, beach, "
        "restaurant, hotel...). Vary the framing across the set: at least one close-up or selfie, one waist-up, "
        "one full-body. Poses are natural social-media poses, never stiff."
    ),
    "Fashion": (
        "FASHION LOOKBOOK of the outfit in {n} photos. The outfit is the star: every shot shows it clearly. "
        "Mix studio shots (plain seamless backdrop, front, three-quarter, back view, a detail close-up) with "
        "street-style shots in a stylish city, depending on the scene field. Model poses, fashion framing."
    ),
    "UGC": (
        "UGC PRODUCT PHOTOS in {n} photos: she is a real creator showing the product from image 2 "
        "({product}). Classic creator shots: the product held beside her face, the product close to the lens, "
        "the product in use, a close-up with the product under her chin, an unboxing or shelf moment. "
        "Places fit the product (bathroom or vanity for skincare, kitchen for food, desk for tech...). "
        "The product is big and sharp in the frame, label facing the camera."
    ),
}

_CANDID_RULES = """

CANDID SNAP (smartphone look): every shot is a real phone snapshot from a camera roll, casual, unposed or
lightly posed, with specific small camera flaws, never polished. For EVERY shot also fill:
- "framing": one of Close-up · Medium close-up · Waist-up · Medium · Three-quarter · Full-body ·
  Mirror selfie · Selfie, plus the camera height when it is not eye level (e.g. "Low-angle waist-up").
- "action": what she is doing, as an -ing phrase ("sipping an iced latte", "tying her shoe").
- "time": time of day / weather ("late afternoon", "on a rainy evening").
- "hands": BOTH hands placed: which hand, verb, which part of the object, where it sits.
- "background": EXACTLY three concrete visible things, the third partially visible, as
  {"a": "...", "b": "...", "c": "..."}.
- "light": ONE key light from the scene: its source, its direction, and where it lands on her face.
- "flaws": 2 to 4 mild phone flaws, at least one framing flaw, max one per group (texture max two),
  never on her face. FRAMING: slightly tilted framing, a slightly crooked horizon, top of her head partly
  cropped, her elbow cut off at the frame edge, off-center with empty space at the top. EXPOSURE: the
  window slightly overexposed, the [light] slightly clipped, slightly lifted grey shadows, slight HDR
  flattening (daylight only). COLOR: slightly yellow cast, slightly orange cast, slightly cool blue cast,
  uneven auto white balance, mixed warm and cool light, slightly muted colors. TEXTURE: mild digital noise in
  the shadows (never in bright daylight), a little motion blur in [moving part], faint lens flare at the
  frame edge, soft bloom around [bulbs], [background object] softly blurred behind her.
- "unaware": true only when she is caught unaware (not looking at the lens, not posing, not a selfie).
Allowed strength words: mild, slight, slightly, a little, faint, small, soft. Never: grainy, heavy grain,
blurry, out of focus, low-res, distorted, glitch, quality or camera words (8k, cinematic, DSLR, bokeh,
f-numbers, ISO, mm). Never pair: flash + window or sun as key light; night + sunlight; a yellow cast + a
cool blue cast (unless two named sources); overcast + hard shadows.
"""

_SYSTEM = """You are a professional photographer and art director planning an Instagram carousel.
You get the ORIGINAL photo of one woman{img2}. You plan {n} new photos of THIS SAME woman.
Her face, hair, skin and body never change: you never describe them (no hair colour, no eye colour,
no skin tone, no age, no body unless the brief asks for a body shape in words).

Brief:
{brief}

Rules:
- Each shot = one clear, physically possible moment. One person only{person_rule}.
- place: a concrete location with 2-4 visible details (materials, colours, objects). Real places, never fantasy.
- pose: what her body, hands and head do, in plain words (15-30 words). Hands do something real.
  Every pose is clearly different from the original photo and from the other shots.
- framing: one of: close-up selfie / close-up / waist-up / three-quarter body / full-body / back view /
  over-the-shoulder; plus the camera height (eye level, slightly high, low angle).
- light: the real light of that place and time of day (window light, golden hour, neon, overcast...).
- expression: {face_rule}
- creativity {creativity}: 0.6 = calm and safe places and angles, 0.9 = varied, 1.2 = bold.
{outfit_rule}
Return ONLY JSON, no markdown:
{{"story": "one sentence", "bottoms": "what she wears below the waist if the photo does not show it, else empty",
"image_2_name": "precise name of image 2 (brand, colour, shape) or empty",
"outfit": "see OUTFIT DESCRIPTION below; empty only if the user gave the outfit",
"image_2_description": "precise visual description of image 2 in words (shape, colours, materials, label / text) or empty",
"photo_place": "see PLACE OF THE PHOTO below",
"shots": [{{"place": "...", "pose": "...", "framing": "...", "light": "...", "expression": "...",
"action": "what she does with image 2 in this shot, or empty"}}]}}
The shots array has exactly {n} items.

OUTFIT DESCRIPTION (the "outfit" field). The images are generated from text only, so this text is the ONLY
thing that makes the clothes match from shot to shot: it must let a painter reproduce the outfit exactly.
Describe what she wears in the photo (or the garment of image 2 when image 2 is the outfit), garment by
garment, from top to bottom, then shoes, then accessories. For EVERY garment give, when visible:
- the exact garment type (e.g. "strapless bandeau crop top", not "top");
- the exact colour with its shade and finish (e.g. "matte jet black", "pale dusty pink", "off-white ecru"),
  and the colour of any trim, lining, stitching or hardware;
- the fabric and its texture (ribbed knit, satin, washed denim, sheer mesh, cotton jersey, leather...);
- the cut and fit (fitted, oversized, cropped, high-waisted, bodycon, A-line, flared, straight-leg...);
- neckline, straps, sleeves (type and length), hem length (exact: "mid-thigh", "ankle", "just below the bust");
- pattern or print (stripes with width and colours, floral with colours, solid), logos or text (exact words,
  colour, position);
- construction details: buttons, zips, laces, pockets, slits (side and height), ruching, draping, pleats,
  cut-outs, seams, belt;
- how it is worn: tucked in, rolled sleeves, open or buttoned, worn off-shoulder.
Shoes: type, colour, heel height, material. Socks or tights. Accessories: jewellery (metal colour, size,
where), bag, belt, glasses, hat, phone case colour if she holds a phone.
Write it as one flowing description in plain English, 70 to 140 words, no lists. Never guess what is not
visible: if the photo does not show her legs, say nothing about them (the bottoms field covers it).
Never describe her body, skin, hair or face here.

PLACE OF THE PHOTO (the "photo_place" field). Describe the place visible in the ORIGINAL photo so precisely
that a set designer could rebuild it: the kind of room or spot; the walls (colour with its shade, material,
texture, panelling, tiles, wallpaper pattern); the floor (material, colour, rug); every visible piece of
furniture with its colour, material and position (left / right / behind her); decor and objects (frames,
plants, lamps, mirror, curtains, shelves, bedding with its colour and pattern); the windows or doors and
where the light comes from; the overall colour palette and the light's colour temperature. 80 to 150 words,
one flowing description, only what is visible. When the set stays in this place (see TIME SPAN), every
shot's "place" names only WHICH PART of this same place she is in ("by the window", "sitting on the bed",
"in front of the mirror"), never a new room.
"""


def _clean(value, limit=4000):
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value or "")).strip()
    return value[:limit]


def _lines(value):
    out = []
    for raw in _clean(value).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip()
        if line:
            out.append(line)
    return out


def _to_pil(image):
    if image is None:
        return None
    frame = image[0] if getattr(image, "ndim", 0) == 4 else image
    arr = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3]).convert("RGB")


def _side_by_side(a, b):
    """The cloud helpers take ONE image: photo on the left, image 2 on the right."""
    h = 1024
    a = a.resize((max(1, round(a.width * h / a.height)), h), Image.Resampling.LANCZOS)
    b = b.resize((max(1, round(b.width * h / b.height)), h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (a.width + 24 + b.width, h), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + 24, 0))
    return canvas


def _parse_json(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("[Onyx Story Director] The LLM did not return JSON.")
    return json.loads(text[start:end + 1])


def _sentence(text):
    text = _clean(text, 600).strip().rstrip(".")
    return (text[:1].upper() + text[1:] + ".") if text else ""


def _lower_first(text):
    text = _clean(text, 600).strip().rstrip(".")
    return text[:1].lower() + text[1:] if text else ""


class OnyxKrea2StoryDirector:
    """Plans a whole Krea2 Edit carousel (Lifestyle / Fashion / UGC) from one photo."""

    SEARCH_ALIASES = ["krea carousel", "story carousel", "ugc carousel", "lookbook"]
    DESCRIPTION = (
        "One photo in, 2 to 8 consistent shots out. A vision LLM plans the set (a day story, a fashion "
        "lookbook or UGC product shots) and writes one Krea2 Identity Edit instruction per shot. "
        "edit_prompts is a LIST: the rest of the workflow runs once per shot, always from the original photo."
    )
    CATEGORY = "Onyx/Krea2 Carousel"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("edit_prompts", "empty_negative_prompts", "plan", "count")
    OUTPUT_IS_LIST = (True, True, False, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "photo": ("IMAGE", {"tooltip": "The ORIGINAL photo of the person (never a generated image)."}),
                "mode": (list(MODES), {"tooltip": "Lifestyle: her day in real places · Fashion: a lookbook of "
                                                  "an outfit · UGC: her with a product (image 2)."}),
                "prompt_style": (list(STYLES), {"tooltip": "Text-to-image: full prompts for a normal Krea "
                                                           "workflow; identity comes from the character LoRA "
                                                           "(set character_trigger), the outfit is written in "
                                                           "words and repeated in every shot. Krea2 Edit: "
                                                           "edit instructions for Krea2EditModelPatch + "
                                                           "grounded encode, rendered from the photo."}),
                "shots": ("INT", {"default": 4, "min": 1, "max": MAX_SHOTS}),
                "camera": (list(CAMERAS), {"tooltip": "smartphone = Candid Snap phone photos (the Candid skill's prompt "
                                                      "structure, text-to-image). auto = smartphone for Lifestyle, professional for "
                                                      "Fashion and UGC. clean = keeps your photo's own look."}),
                "face": (list(FACES), {"tooltip": "auto: a different subtle expression per shot · natural: "
                                                  "relaxed and real · custom: write them in custom_faces."}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**31 - 1,
                                 "tooltip": "A new plan from the same photo (plans are cached per seed)."}),
                "backend": (list(BACKENDS), {"default": "Vertex AI"}),
                "model_id": ("STRING", {"default": "gemini-3.6-flash",
                                        "tooltip": "Vertex / Gemini model id, or the Grok model id."}),
                "vertex_json_folder": ("STRING", {"default": "C:\\Comfyui\\json",
                                                  "tooltip": "Folder with your Vertex service-account .json "
                                                             "files (Vertex AI only). Gemini / Grok keys come "
                                                             "from GEMINI_API_KEY / XAI_API_KEY."}),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "Optional: an outfit, a product, a second person or a body "
                                                 "shape. Also wire it to image_b / source_image_b of the edit nodes."}),
                "image_2_is": (list(IMAGE2_KINDS),),
                "image_2_name": ("STRING", {"default": "", "tooltip": "e.g. 'Creed Carmina perfume, gold cap'. "
                                                                    "Empty = read from the image."}),
                "scene": ("STRING", {"default": "", "multiline": True,
                                     "tooltip": "Empty = chosen from the photo. One line = a vibe ('a sunny "
                                                "Saturday in Lisbon'). Several lines = one exact place per shot."}),
                "story": ("STRING", {"default": "", "multiline": True,
                                     "tooltip": "Lifestyle: the order of the day (morning to night, a trip, "
                                                "a night out...). Empty = the LLM picks."}),
                "custom_faces": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "face = custom. One line = same for every shot; one "
                                                       "line per shot = in order."}),
                "outfit": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "Empty = she keeps the outfit of the photo in every shot. "
                                                 "Text = this outfit in every shot."}),
                "bottoms": ("STRING", {"default": "", "tooltip": "What she wears below the waist when the "
                                                                "photo does not show it. Empty = picked once."}),
                "body_shape": ("STRING", {"default": "", "tooltip": "Her figure in words, e.g. 'slim athletic "
                                                                   "figure'. Empty = her own."}),
                "creativity": ("FLOAT", {"default": 0.9, "min": 0.3, "max": 1.5, "step": 0.05,
                                         "tooltip": "How bold the places and angles are (0.6 calm, 0.9 varied, 1.2 bold). "
                                                    "Higher = MORE variety, not more continuity: use time_span for that."}),
                "character_trigger": ("STRING", {"default": "", "tooltip": "Trigger word of a character LoRA, "
                                                                          "put at the start of every prompt."}),
                "extra_prompt": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "Appended verbatim to the end of every edit prompt."}),
                "time_span": (list(TIME_SPANS), {"default": "auto",
                    "tooltip": "How much time the set covers. one moment = same place, same light, a burst of "
                               "photos · one hour · one day (morning to night) · a trip. auto = the mode's own "
                               "(Lifestyle: one day). The `story` text still adds details."}),
            },
        }

    # ── plan ────────────────────────────────────────────────────────────────
    def _plan(self, photo, image_2, kind, mode, n, face, custom, scene, story, outfit, body_shape,
              creativity, image_2_name, seed, backend, model_id, json_folder, candid=False, time_span="auto"):
        scene_lines = _lines(scene)
        if len(scene_lines) > 1:
            scene_txt = "Use exactly these places, in order: " + " | ".join(scene_lines[:n]) + "."
        elif scene_lines:
            scene_txt = "Overall setting / vibe for the whole set: " + scene_lines[0] + "."
        else:
            scene_txt = "Choose the places from what fits her photo and style."
        default_story = {"one moment (same place)": "one moment in one place", "one hour": "about one hour",
                         "a trip (several days)": "a trip over several days"}.get(time_span, "a natural flow, e.g. morning to night")
        brief = _MODE_BRIEF[mode].format(n=n, story=_clean(story, 500) or default_story,
                                         product=_clean(image_2_name, 300) or "read it from image 2")
        brief += "\n" + scene_txt
        if time_span in _SPAN_RULES:
            brief += "\n" + _SPAN_RULES[time_span]
            if time_span == "one moment (same place)":
                brief = brief.replace("Each shot is a DIFFERENT real place", "Each shot is a different moment of the SAME place")
                brief += ("\nThe place is the place of HER PHOTO (photo_place), unless the scene field names "
                          "another one: rebuild it, do not invent a new room.")
        if body_shape:
            brief += f"\nHer body shape in words: {body_shape} (never more than that about her body)."
        face_rule = {
            "auto": "a different SUBTLE expression per shot (soft smile, relaxed, playful glance, calm "
                    "confident look...). Never big laughs, never lip bites.",
            "natural": "relaxed and real, small natural smile or neutral, like a candid photo.",
            "custom": "copy the one given by the user for this shot.",
        }[face]
        if outfit:
            outfit_rule = f"- She wears this outfit in every shot: {outfit}."
        elif image_2 is not None and kind == "outfit":
            outfit_rule = "- She wears the outfit shown in image 2 (right side) in every shot."
        else:
            outfit_rule = ("- She keeps the outfit of her photo in every shot (do not invent a new one). Poses and "
                           "framings never hide or change it: no jacket added, nothing removed.")
        img2 = ""
        person_rule = ""
        if image_2 is not None:
            img2 = (" on the LEFT, and IMAGE 2 on the RIGHT, which is "
                    + {"outfit": "an outfit (garment only: take nothing else from it)",
                       "product": "a product she will hold, show and use",
                       "person": "a second person who is in the photos with her",
                       "body shape": "a body-shape reference (only the figure matters)"}[kind])
            if kind == "person":
                person_rule = " (plus the person of image 2, interacting naturally with her)"
        system = _SYSTEM.format(img2=img2, n=n, brief=brief, person_rule=person_rule, face_rule=face_rule,
                                creativity=creativity, outfit_rule=outfit_rule)
        if candid:
            system += _CANDID_RULES
        user = f"Plan the {n} shots now. Seed {seed}."
        if face == "custom" and custom:
            user += "\nExpressions given by the user, in order: " + " | ".join(custom)

        image = _to_pil(photo)
        if image_2 is not None:
            image = _side_by_side(image, _to_pil(image_2))

        from . import pose_generator as pg
        core = pg._onyx_core()
        timeout = 300
        if backend == "Vertex AI":
            text = core["_vertex_generate"](json_folder, model_id or "gemini-3.6-flash", system, user, image, timeout)
        elif backend == "Gemini API":
            text = core["_gemini_generate"](pg.RUNTIME_KEYS.get("Gemini API", ""), model_id, system, user,
                                            image, timeout)
        elif backend == "Grok API":
            text = core["_grok_generate"](pg.RUNTIME_KEYS.get("Grok API", ""), model_id, system, user,
                                          image, timeout)
        else:
            from . import local_vlm
            text = local_vlm.generate_text(system, user, pil_images=[image], image_labels=["PHOTO"],
                                           model_id=None, temperature=0.8, max_new_tokens=3000, seed=seed)
        plan = _parse_json(text)
        shots = [s for s in (plan.get("shots") or []) if isinstance(s, dict)]
        if not shots:
            raise ValueError("[Onyx Story Director] The plan has no shots.")
        while len(shots) < n:                       # pad by reusing places with a new framing
            shots.append(dict(shots[len(shots) % len(shots)]))
        plan["shots"] = shots[:n]
        return plan

    # ── prompt assembly ─────────────────────────────────────────────────────
    @staticmethod
    def _instruction(shot, mode, camera, kind, has_img2, img2_name, outfit, bottoms, body_shape, face,
                     custom, index):
        place = _lower_first(shot.get("place"))
        parts = []
        if mode == "Fashion" and re.search(r"\bstudio|backdrop|seamless\b", place):
            parts.append(f"Change the scene: she is now in a photo studio, {place}.")
        else:
            parts.append(f"Change the scene: she is now {place if place.startswith(('in ', 'at ', 'on ')) else 'in ' + place}.")
        parts.append(f"Change her pose: {_lower_first(shot.get('pose'))}.")
        if shot.get("framing"):
            parts.append(_sentence("Framing: " + _lower_first(shot.get("framing"))))
        if shot.get("light"):
            parts.append(_sentence("Light: " + _lower_first(shot.get("light"))))
        expr = ""
        if face == "custom" and custom:
            expr = custom[0] if len(custom) == 1 else custom[index % len(custom)]
        else:
            expr = shot.get("expression") or ""
        if expr:
            parts.append(_sentence("Her expression: " + _lower_first(expr)))
        if has_img2:
            name = f" ({img2_name})" if img2_name else ""
            action = _lower_first(shot.get("action"))
            if kind == "outfit":
                parts.append(f"She wears the outfit from the second image{name}, replacing her original "
                             "clothes; take only the garments from it.")
            elif kind == "product":
                parts.append(f"She holds the product from the second image{name}"
                             + (f" ({action})" if action else "")
                             + ", exactly as in that image: same shape, colors and label, big and sharp in the frame.")
            elif kind == "person":
                parts.append(f"The person from the second image{name} is with her in the photo"
                             + (f", {action}" if action else "") + "; two people, both with their own faces.")
        if outfit:
            parts.append(_sentence("She wears " + _lower_first(outfit)))
        if bottoms:
            parts.append(_sentence("Below the waist she wears " + _lower_first(bottoms)))
        if body_shape:
            parts.append(_sentence("Her body: " + _lower_first(body_shape)))
        parts.append(CAMERA_LINES[camera])
        keep = "Keep her identity (facial features), body proportions and hair"
        if not outfit and not (has_img2 and kind == "outfit"):
            keep += ", and her outfit from the reference image"
        parts.append(keep + ".")
        parts.append("One photo." if (has_img2 and kind == "person") else "One person, one photo.")
        return " ".join(p for p in parts if p)

    @staticmethod
    def _candid_prompt(shot, kind, has_img2, img2_name, img2_desc, outfit, bottoms, body_shape, face, custom,
                       index, trigger, room=""):
        """Candid Snap order: frame + action, expression, outfit, hands, background (3 things), one light,
        phone flaws, skin. Line 1 = the trigger alone. Identity (face, hair, skin tone) is the LoRA's."""
        framing = _clean(shot.get("framing"), 80).strip().rstrip(".") or "Waist-up"
        framing = framing[:1].upper() + framing[1:]
        if not re.search(r"\b(shot|selfie)\b", framing, re.I):
            framing += " shot"
        place = _lower_first(shot.get("place"))
        if place and not place.startswith(("in ", "at ", "on ", "inside ", "outside ", "by ", "near ")):
            place = "in " + place
        action = _lower_first(shot.get("action") or shot.get("pose"))
        body = _lower_first(body_shape).removeprefix("a ") if body_shape else ""
        who = f"a {body}" if body else "a young woman"
        time_ = _lower_first(shot.get("time"))
        s1 = f"{framing}, A candid phone snapshot of {who}, {action} {place}" + (f", {time_}" if time_ else "")
        parts = [s1.replace(" ,", ",").rstrip(",") + "."]
        expr = (custom[0] if len(custom) == 1 else custom[index % len(custom)]) if (face == "custom" and custom) \
            else (shot.get("expression") or "")
        if expr:
            parts.append(_sentence("Her expression: " + _lower_first(expr)))
        if outfit:
            o = re.sub(r"^(?:she\s+(?:is\s+)?wear(?:s|ing)\s+)", "", outfit.strip().rstrip("."), flags=re.I)
            parts.append(f"She wears exactly this outfit, identical in every detail: {o[:1].lower() + o[1:]}.")
        if room:      # the same set in every shot: written word for word the same
            r = room.strip().rstrip(".")
            parts.append(f"The place, identical in every detail: {r[:1].lower() + r[1:]}.")
        if bottoms and (not outfit or "below" not in outfit):
            parts.append(_sentence("Below the waist she wears " + _lower_first(bottoms)))
        if shot.get("hands"):
            parts.append(_sentence(shot["hands"]))
        if has_img2 and kind == "product":
            what = img2_desc or img2_name or "the product"
            parts.append(_sentence(f"She holds {what}, big and sharp in the frame, label facing the camera"))
        elif has_img2 and kind == "person":
            parts.append(_sentence(f"With her in the photo: {img2_desc or 'a friend'}"))
        bg = shot.get("background") or {}
        if isinstance(bg, dict) and bg.get("a"):
            parts.append(f"The background shows {_lower_first(bg.get('a'))}, {_lower_first(bg.get('b'))} "
                         f"and a partially visible {_lower_first(bg.get('c')).removeprefix('a ').removeprefix('the ')}.")
        if shot.get("light"):
            parts.append(_sentence(shot["light"]))
        flaws = [_lower_first(f) for f in (shot.get("flaws") or []) if str(f).strip()][:4]
        if flaws:
            lead = "Shot quickly on a phone without posing" if shot.get("unaware") else "Shot quickly on a phone"
            parts.append(f"{lead}: " + ", ".join(flaws) + ".")
        parts.append("Realistic skin with natural texture and visible pores.")
        para = " ".join(p for p in parts if p)
        return f"{trigger}\n{para}" if trigger else para

    @staticmethod
    def _t2i_prompt(shot, camera, kind, has_img2, img2_name, img2_desc, outfit, bottoms, body_shape, face,
                    custom, index, trigger, room=""):
        """A full text-to-image prompt: identity from the character LoRA, everything else in words."""
        place = _lower_first(shot.get("place"))
        if place and not place.startswith(("in ", "at ", "on ", "inside ", "outside ")):
            place = "in " + place
        subject = (trigger + ", a young woman") if trigger else "A young woman"
        parts = [f"{subject} {_lower_first(shot.get('pose'))}, {place}.".replace(" ,", ",")]
        # The outfit comes right after the subject (Krea weighs the start of the prompt most) and is
        # written word for word the same in every shot: it is the only thing that makes the clothes match.
        if outfit:
            o = re.sub(r"^(?:she\s+(?:is\s+)?wear(?:s|ing)\s+)", "", outfit.strip().rstrip("."), flags=re.I)
            parts.append(f"She wears exactly this outfit, identical in every detail: {o[:1].lower() + o[1:]}.")
        if room:      # the same set in every shot: written word for word the same
            r = room.strip().rstrip(".")
            parts.append(f"The place, identical in every detail: {r[:1].lower() + r[1:]}.")
        if bottoms and (not outfit or "below" not in outfit):
            parts.append(_sentence("Below the waist she wears " + _lower_first(bottoms)))
        if shot.get("framing"):
            parts.append(_sentence("Framing: " + _lower_first(shot.get("framing"))))
        if shot.get("light"):
            parts.append(_sentence("Light: " + _lower_first(shot.get("light"))))
        expr = (custom[0] if len(custom) == 1 else custom[index % len(custom)]) if (face == "custom" and custom) \
            else (shot.get("expression") or "")
        if expr:
            parts.append(_sentence("Her expression: " + _lower_first(expr)))
        if body_shape:
            parts.append(_sentence("She has a " + _lower_first(body_shape).removeprefix("a ")))
        if has_img2 and kind == "product":
            what = img2_desc or img2_name or "the product"
            action = _lower_first(shot.get("action"))
            parts.append(_sentence(f"She holds {what}" + (f", {action}" if action else "")
                                   + ", big and sharp in the frame, label facing the camera"))
        elif has_img2 and kind == "person":
            what = img2_desc or "a friend"
            action = _lower_first(shot.get("action"))
            parts.append(_sentence(f"With her in the photo: {what}" + (f", {action}" if action else "")))
        parts.append(CAMERA_T2I[camera])
        return " ".join(p for p in parts if p)

    def execute(self, photo, mode, prompt_style, shots, camera, face, seed, backend, model_id, vertex_json_folder,
                image_2=None, image_2_is="outfit", image_2_name="", scene="", story="", custom_faces="",
                outfit="", bottoms="", body_shape="", creativity=0.9, character_trigger="", extra_prompt="", time_span="auto"):
        n = max(1, min(MAX_SHOTS, int(shots)))
        mode = mode if mode in MODES else "Lifestyle"
        cam = camera if camera in CAMERA_LINES else ("smartphone" if mode == "Lifestyle" else "professional")
        kind = image_2_is if image_2_is in IMAGE2_KINDS else "outfit"
        custom = _lines(custom_faces)
        outfit, bottoms, body_shape = _clean(outfit, 1500), _clean(bottoms, 300), _clean(body_shape, 300)
        has_img2 = image_2 is not None and kind != "body shape"
        # Smartphone look in text-to-image = the Candid Snap prompt structure (the Candid skill).
        candid = cam == "smartphone" and prompt_style != STYLE_EDIT

        arr = photo[0].detach().cpu().numpy()
        key_src = {
            "photo": hashlib.sha256(np.ascontiguousarray(arr[::8, ::8]).tobytes()).hexdigest(),
            "img2": (hashlib.sha256(np.ascontiguousarray(image_2[0].detach().cpu().numpy()[::8, ::8])
                                    .tobytes()).hexdigest() if image_2 is not None else ""),
            "mode": mode, "n": n, "face": face, "custom": custom, "scene": scene, "story": story,
            "outfit": outfit, "body": body_shape, "creativity": creativity, "kind": kind,
            "name": image_2_name, "seed": seed, "backend": backend, "model": model_id, "candid": candid, "span": time_span, "v": 5,
        }
        key = hashlib.sha256(json.dumps(key_src, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
        path = os.path.join(_CACHE_DIR, f"{key}.json")
        plan = None
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    plan = json.load(fh)
                print(f"[Onyx Story Director] plan from cache ({key})")
            except Exception:
                plan = None
        if plan is None:
            print(f"[Onyx Story Director] planning {n} {mode} shots with {backend}…")
            plan = self._plan(photo, image_2, kind, mode, n, face, custom, scene, story, outfit, body_shape,
                              creativity, image_2_name, seed, backend, model_id, vertex_json_folder,
                              candid=candid, time_span=time_span)
            os.makedirs(_CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=1)

        name = _clean(image_2_name, 300) or _clean(plan.get("image_2_name"), 300)
        bottoms = bottoms or _clean(plan.get("bottoms"), 300)
        trigger = _clean(character_trigger, 200)
        extra = _clean(extra_prompt, 2000)
        t2i = prompt_style != STYLE_EDIT
        # One moment in the photo's own place: the same, precise room text in every prompt.
        room = (_clean(plan.get("photo_place"), 1500)
                if time_span == "one moment (same place)" and not _lines(scene) else "")
        prompts = []
        for i, shot in enumerate(plan["shots"]):
            if candid:
                p = self._candid_prompt(shot, kind, has_img2, name, _clean(plan.get("image_2_description"), 600),
                                        outfit or _clean(plan.get("outfit"), 1500), bottoms, body_shape, face,
                                        custom, i, trigger, room=room)
            elif t2i:
                p = self._t2i_prompt(shot, cam, kind, has_img2, name, _clean(plan.get("image_2_description"), 600),
                                     outfit or _clean(plan.get("outfit"), 1500), bottoms, body_shape, face,
                                     custom, i, trigger, room=room)
            else:
                p = self._instruction(shot, mode, cam, kind, has_img2, name, outfit, bottoms, body_shape, face,
                                      custom, i)
                if trigger:
                    p = f"{trigger}, {p}"
            if extra:
                p = f"{p} {extra}"
            prompts.append(p)

        readable = [f"STORY: {plan.get('story', '')}",
                    f"mode {mode} · {'text-to-image' if t2i else 'krea2 edit'} · {n} shots · camera {cam}"
                    f"{' (Candid Snap)' if candid else ''} · face {face}",
                    ""]
        for i, p in enumerate(prompts, 1):
            readable.append(f"[{i}] {p}")
        plan_text = "\n".join(readable)
        return {
            "ui": {"text": [plan_text]},
            "result": (prompts, ["" for _ in prompts], plan_text, len(prompts)),
        }


__all__ = ["OnyxKrea2StoryDirector"]
