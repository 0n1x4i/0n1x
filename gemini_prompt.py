"""
ComfyUI node — OFM Prompt Generator.
Uses Gemini LLM to analyze images and generate structured JSON prompts,
or allows custom freeform prompt generation.
"""

import io
import base64
import logging
import re
import os
import sys
import json
import subprocess

import numpy as np
import requests
from PIL import Image



from .nodes.onyx_render_profile import ensure_profile_ready

from .nodes.onyx_remote_exec import load_bundle_module as _onyx_load_bundle_module


def _onyx_core():
    ns = _onyx_load_bundle_module('postfx_bundle', 'gemini_prompt', __package__)
    return ns

_PROVIDERS = ["Gemini", "Grok", "Vertex"]

_GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

_GROK_MODELS = [
    # Vision-capable models (support image input)
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-2-vision-1212",
    # Text-only models (no image input)
    "grok-3",
    "grok-3-fast",
    "grok-3-mini",
    "grok-3-mini-fast",
]

# The `model` dropdown is declared with every id, because ComfyUI validates the
# widget value server-side and would reject a Grok id if the list only held
# Gemini ones. The JS narrows what's *shown* to the active provider; run()
# re-checks the pairing, since a workflow driven through the API never runs it.
_ALL_MODELS = _GEMINI_MODELS + _GROK_MODELS

# ── Thinking levels ─────────────────────────────────────────────────────────
# Source: https://ai.google.dev/gemini-api/docs/thinking ("Controlling thinking").
# Levels are NOT uniform across models — 3.1 Pro and the 2.5 family reject
# "minimal" — so an unsupported pick is dropped rather than sent and 400'd.
_THINKING_LEVELS  = ["default", "minimal", "low", "medium", "high"]
_THINKING_SUPPORT = {
    "gemini-3.6-flash":       {"minimal", "low", "medium", "high"},
    "gemini-3.5-flash":       {"minimal", "low", "medium", "high"},
    "gemini-3.5-flash-lite":  {"minimal", "low", "medium", "high"},
    "gemini-3.1-pro-preview": {"low", "medium", "high"},
    "gemini-2.5-pro":         {"low", "medium", "high"},
    "gemini-2.5-flash":       {"low", "medium", "high"},
}

# ── Deprecated sampling parameters ──────────────────────────────────────────
# "Starting with Gemini 3.6 Flash and Gemini 3.5 Flash-Lite, temperature, top_p
# and top_k are deprecated. The API ignores these parameters and returns an
# error in future model generations."  — docs/latest-model
#
# Listed explicitly rather than derived from the version number: Gemini's
# numbering isn't monotonic (3.5 Flash-Lite shipped *after* 3.5 Flash), so a
# ">= 3.5" test would wrongly strip temperature from 3.5 Flash.
# ⚠️  Add every new Gemini model to this set — the deprecation applies to all
# future releases, and sending temperature to them will eventually be a 400.
_GEMINI_NO_SAMPLING = {
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
}


def _is_grok_model(*args, **kwargs):
    return _onyx_core()['_is_grok_model'](*args, **kwargs)


_PROMPT_MODES = [
    "JSON Image Analysis",
    "Style Transfer Prompt",
    "Seedream Edit (2 face + 1 ref)",
    "Seedream Edit (1 face + 1 ref)",
    "Seedream Edit (2 face and body + 1 ref)",
    "Seedream Edit (2 faces + 2 bodies + 1ref)",
    "Custom Prompt",
]

# ---------- Built-in JSON analysis prompt ----------

_JSON_ANALYSIS_PROMPT = r"""## Context & Goal
You are an expert visual analysis specialist with 15+ years of experience in digital art, photography, graphic design, and AI image generation. You excel at deconstructing visual elements and translating artistic styles into technical specifications.

Your task: Analyze the uploaded image and return a comprehensive JSON profile for recreating the visual style.

## Output Format
Return ONLY valid JSON. No explanations, no commentary, no markdown formatting.

## Core JSON Schema

{
"metadata": {
"confidence_score": "high/medium/low - assessment of analysis accuracy",
"image_type": "photograph/digital art/illustration/graphic design/mixed media",
"primary_purpose": "marketing/editorial/social media/product/portrait/landscape/abstract"
},
"composition": {
"rule_applied": "rule of thirds/golden ratio/center composition/symmetry/asymmetry",
"aspect_ratio": "width:height ratio or format description",
"layout": "grid/single subject/multi-element/layered",
"focal_points": [
"Primary focal point location and element",
"Secondary focal point if present"
],
"visual_hierarchy": "Description of how eye moves through the image",
"balance": "symmetric/asymmetric/radial - with description"
},
"color_profile": {
"dominant_colors": [
{
"color": "Specific color name",
"hex": "#000000",
"percentage": "approximate percentage of image",
"role": "background/accent/primary subject"
}
],
"color_palette": "complementary/analogous/triadic/monochromatic/split-complementary",
"temperature": "warm/cool/neutral - overall feeling",
"saturation": "highly saturated/moderate/desaturated/black and white",
"contrast": "high contrast/medium contrast/low contrast/soft"
},
"lighting": {
"type": "natural window/artificial/mixed/studio/practical lights",
"source_count": "single source/multiple sources - number and placement",
"direction": "front/45-degree side/90-degree side/back/top/bottom/diffused from above",
"directionality": "highly directional/moderately directional/diffused/omni-directional",
"quality": "hard light/soft light/dramatic/even/gradient/sculpted",
"intensity": "bright/moderate/low/moody/high-key/low-key",
"contrast_ratio": "high contrast (dramatic shadows)/medium contrast/low contrast (flat)",
"mood": "cheerful/dramatic/mysterious/calm/energetic/professional/casual",
"shadows": {
"type": "harsh defined edges/soft gradual edges/minimal/dramatic/absent",
"density": "deep black/gray/transparent/faint",
"placement": "under subject/on wall/from objects/cast patterns",
"length": "short/medium/long - shadow projection distance"
},
"highlights": {
"treatment": "blown out/preserved/subtle/dramatic/specular",
"placement": "on face/hair/clothing/background - where light hits strongest"
},
"ambient_fill": "present/absent - secondary fill light reducing shadows",
"light_temperature": "warm (golden)/neutral/cool (blue) - color cast"
},
"technical_specs": {
"medium": "digital photography/3D render/digital painting/vector/photo manipulation/mixed",
"style": "realistic/hyperrealistic/stylized/minimalist/maximalist/abstract/surreal",
"texture": "smooth/grainy/sharp/soft/painterly/glossy/matte",
"sharpness": "tack sharp/slightly soft/deliberately soft/bokeh effect",
"grain": "none/film grain/digital noise/intentional grain - level",
"depth_of_field": "shallow/medium/deep - with subject isolation description",
"perspective": "straight on/low angle/high angle/dutch angle/isometric/one-point/two-point"
},
"artistic_elements": {
"genre": "portrait/landscape/abstract/conceptual/commercial/editorial/street/fine art",
"influences": [
"Identified artistic movement, photographer, or style influence"
],
"mood": "energetic/calm/dramatic/playful/sophisticated/raw/polished",
"atmosphere": "Description of overall feeling and emotional impact",
"visual_style": "clean/cluttered/minimal/busy/organic/geometric/fluid/structured"
},
"typography": {
"present": "true or false",
"fonts": [
{
"type": "sans-serif/serif/script/display/handwritten",
"weight": "thin/light/regular/medium/bold/black",
"characteristics": "modern/vintage/playful/serious/technical"
}
],
"placement": "overlay/integrated/border/corner - with strategic description",
"integration": "subtle/prominent/dominant/background"
},
"subject_analysis": {
"primary_subject": "Main subject description - DO NOT describe face structure, hair, skin tone, or any identity features",
"positioning": "center/left/right/top/bottom/rule of thirds placement",
"scale": "close-up/medium/full/environmental/macro",
"interaction": "isolated/interacting with environment/multiple subjects",
"facial_expression": {
"mouth": "closed smile/open smile/slight smile/neutral/serious/pursed - exact mouth position",
"smile_intensity": "no smile/subtle/moderate/broad/wide - degree of smile",
"eyes": "direct gaze/looking away/squinting/wide/relaxed/intense - eye expression",
"eyebrows": "raised/neutral/furrowed/relaxed - brow position",
"overall_emotion": "happy/content/serious/playful/confident/approachable/guarded/warm/cold",
"authenticity": "genuine/posed/candid/formal/natural"
},
"hands_and_gestures": {
"left_hand": "Exact position and gesture - touching face/holding object/resting on surface/in pocket/behind back/clasped/visible or not visible",
"right_hand": "Exact position and gesture - touching face/holding object/resting on surface/in pocket/behind back/clasped/visible or not visible",
"finger_positions": "Specific details: pointing/peace sign/thumbs up/relaxed/gripping/spread/interlaced/curled",
"finger_interlacing": "if hands clasped: natural loose interlacing/tight formal interlacing/fingers overlapping/thumbs position",
"hand_tension": "relaxed/tense/natural/posed/rigid - muscle tension observable",
"interaction": "What hands are doing: holding phone/touching hair/on hip/crossed/clasped at waist/clasped at chest/gesturing",
"naturalness": "organic casual gesture/deliberately posed/caught mid-motion/static formal pose"
},
"body_positioning": {
"posture": "standing/sitting/leaning/lying - exact position",
"angle": "facing camera/45 degree turn/profile/back to camera",
"weight_distribution": "leaning left/right/centered/shifted",
"shoulders": "level/tilted/rotated/hunched/back"
}
},
"background": {
"setting_type": "indoor/outdoor/studio/natural environment - specific location",
"spatial_depth": "shallow/medium/deep - layers description",
"elements_detailed": [
{
"item": "Specific object name (if plant: species like monstera/pothos/bird of paradise/fern)",
"position": "left/right/center/top/bottom - exact placement with quadrant",
"distance": "foreground/midground/background",
"size": "dominant/medium/small - relative scale and proportion",
"condition": "new/worn/vintage/pristine/wilted/thriving - state description",
"specific_features": "For plants: flower color, leaf pattern, pot type; For objects: brand, wear, details"
}
],
"wall_surface": {
"material": "painted drywall/concrete/brick/wood paneling/tile/wallpaper/plaster - exact base material",
"surface_treatment": "smooth paint/textured paint/raw concrete/polished concrete/exposed brick/finished/unfinished",
"texture": "perfectly smooth/slightly textured/rough/patterned/brushed - tactile quality",
"finish": "matte/satin/glossy/flat - sheen level",
"color": "Specific color with undertones (e.g., warm gray, cool blue-gray, off-white)",
"color_variation": "uniform/gradient/patchy/streaked - color consistency",
"features": "clean/water stains/vertical streaks/horizontal marks/cracks/patches/fixtures/artwork/scuffs - ALL observable surface details",
"wear_indicators": "pristine/aged/weathered/industrial/residential - condition and style"
},
"floor_surface": {
"material": "wood/tile/carpet/concrete/grass - exact type",
"color": "Specific color",
"pattern": "solid/checkered/striped/herringbone - if present"
},
"objects_catalog": "List every visible object with position: furniture pieces, decorative items, functional objects, natural elements",
"background_treatment": "blurred/sharp/minimal/detailed/gradient/textured"
},
"generation_parameters": {
"prompts": [
"Detailed technical prompt for recreating this style",
"Alternative angle or variation prompt"
],
"keywords": [
"keyword1",
"keyword2",
"keyword3",
"keyword4",
"keyword5"
],
"technical_settings": "Recommended camera/render settings description for recreation",
"post_processing": "Color grading, filters, or editing techniques applied"
}
}

## Analysis Rules

### Composition Analysis
- Identify grid systems, alignment, and spatial relationships
- Note use of negative space and breathing room
- Describe visual flow and eye movement path
- Identify focal points using color, contrast, size, or placement
- Assess balance between elements

### Color Analysis
- Extract dominant colors with approximate hex values
- Identify color relationships (complementary, analogous, etc.)
- Assess temperature bias (warm vs cool)
- Note saturation levels and contrast intensity
- Describe how color creates mood and directs attention

### Lighting Assessment
- Determine light source type, direction, and quality
- Analyze shadow characteristics and depth
- Assess highlight preservation or blown-out areas
- Describe overall lighting mood and emotional impact
- Note light's role in creating dimension and form

### Technical Evaluation
- Identify creation medium and technique
- Assess texture, sharpness, and grain characteristics
- Evaluate depth of field and focus points
- Analyze perspective and viewpoint
- Note any technical limitations or intentional choices

### Artistic Context
- Identify genre and artistic influences
- Assess mood, atmosphere, and emotional tone
- Describe visual style (minimal, maximalist, etc.)
- Note any cultural or temporal references
- Evaluate overall aesthetic cohesion

### Typography (if present)
- Identify font styles and weights
- Assess placement and integration strategy
- Evaluate readability and hierarchy
- Describe relationship to other visual elements

### Subject Treatment
- **CRITICAL EXCLUSION — Face & Hair**: Do NOT describe facial features, facial expression, hair color, hair style, hair length, hair texture, skin tone, eye color, or any identity-defining features. These will be provided by separate face reference images. Skip these entirely.
- **Hand & Gesture Analysis (CRITICAL)**:
  - Describe EACH hand separately with exact position
  - Note if hands are visible or hidden (in pockets, behind back, out of frame)
  - Document specific finger positions and shapes
  - For clasped hands: describe interlacing style (natural loose vs formal tight), thumb positions, finger overlap patterns
  - Assess hand tension: relaxed vs tense, organic vs posed
  - Describe what hands are interacting with (phone, face, object, clothing, each other)
  - Note natural vs posed quality of gesture
  - Specify pressure/contact points (lightly touching vs gripping)
  - Evaluate overall naturalness: organic casual vs deliberately posed
- **Background Elements (CRITICAL)**:
  - Catalog EVERY visible object with exact position and quadrant placement
  - For plants: identify species (monstera, pothos, bird of paradise, fern, etc.)
  - Describe spatial relationships between objects and their depth layers
  - **Wall analysis is CRITICAL**:
    - Distinguish between painted drywall vs concrete vs brick vs other materials
    - Note surface treatment: smooth paint vs textured vs raw vs polished
    - Document finish: matte vs glossy vs satin
    - Identify any surface features: water stains, streaks, cracks, patches, wear
    - Assess condition: pristine residential vs industrial weathered vs aged
  - Document floor type, color, pattern
  - Specify distance/depth layer for each element
  - Note condition and state of objects (new/worn/vintage/thriving/wilted)
  - Describe any text, artwork, or decorative elements
  - Include architectural features (windows, doors, molding, fixtures, frames)
- **Lighting Analysis (CRITICAL)**:
  - Distinguish between dramatic directional lighting vs flat even lighting
  - Assess directionality: highly directional (strong shadows) vs diffused (soft minimal shadows)
  - Document shadow characteristics: harsh defined edges vs soft gradual edges vs minimal
  - Note contrast ratio: high contrast (dramatic) vs low contrast (flat)
  - Identify if ambient fill light is present reducing shadow depth
  - Describe how light sculpts the subject vs evenly illuminates
  - Document cast shadows from objects (like plants) on walls
  - Note shadow density: deep black vs gray vs faint
- Analyze primary subject and positioning (but NOT face, hair, or identity features)
- Assess scale and framing choices
- Describe subject-background relationship
- Note any secondary subjects or supporting elements

### Generation Parameters
- Create actionable technical prompts for recreation
- Extract relevant keywords for searchability
- Recommend technical settings for similar results
- Describe post-processing techniques applied

## Output Requirements
- **Format**: Valid JSON only
- **No markdown**: No ```json``` blocks, no backticks
- **No commentary**: No explanatory text before/after JSON
- **No instructions**: No "Here is your analysis" or "Copy this"
- **Clean structure**: Properly formatted, parseable JSON
- **Single object**: Return one complete JSON analysis object
- **Comprehensive**: All sections must be populated with detailed analysis
- **Specific**: Use precise technical terminology, not vague descriptions
- **Actionable**: Generation parameters must be detailed enough for recreation

## Processing Logic
1. Analyze the provided image comprehensively across all categories
2. Extract technical specifications and artistic elements
3. Generate recreation prompts and parameters
4. Output single, complete JSON object

## Quality Standards
- **Confidence score**: Honest assessment of analysis certainty
- **Hex codes**: Approximate but reasonable color values
- **Specific descriptions**: Avoid generic terms like "nice" or "good"
- **Technical accuracy**: Use correct terminology for medium and technique
- **Completeness**: Every JSON field must contain meaningful analysis
- **Actionability**: Prompts and keywords must be specific enough to recreate style

### CRITICAL Accuracy Requirements:

**CRITICAL EXCLUSION — Face Structure & Hair:**
- Do NOT include any hair fields (length, cut, texture, styling, part, volume, details, color)
- Do NOT describe face structure, skin tone, eye color, hair color, or any identity-defining features
- These are controlled by separate face reference images and must be omitted entirely
- EXCEPTION: DO include facial_expression fields (mouth position, smile intensity, eye gaze, eyebrow position, emotion, authenticity) — expressions must be captured from the analyzed image

**Hand & Gesture Description Must Include:**
- Position of BOTH hands (even if one is hidden)
- Exact finger configurations
- For clasped hands: interlacing pattern (loose/tight), thumb position, finger overlap
- Hand tension assessment (relaxed vs tense)
- What hands are interacting with
- Natural vs posed quality with specific evidence
- Contact points and pressure
- Overall gesture naturalness assessment

**Background Description Must Include:**
- Every visible object cataloged with quadrant position
- Plant species identification (not just "plant")
- Wall material distinction (painted drywall vs concrete vs brick - THIS IS CRITICAL)
- Wall surface treatment (smooth paint vs raw concrete vs textured)
- Wall finish (matte vs glossy) and condition (pristine vs weathered vs industrial)
- Any surface features: water stains, vertical streaks, cracks, patches
- Floor materials with exact colors
- Spatial depth layers (foreground/mid/background) for each element
- Object conditions and states
- Architectural features
- Text or decorative elements

**Lighting Description Must Include:**
- Directionality assessment (highly directional vs diffused)
- Contrast ratio (high/medium/low)
- Shadow characteristics: edge quality (harsh vs soft), density (deep vs faint), placement
- Cast shadows from objects onto walls
- Presence or absence of ambient fill light
- Whether lighting is dramatic and sculpting vs flat and even
- This distinction is CRITICAL for recreation accuracy

**Facial Expression (INCLUDE — this is the exception):**
- Exact mouth position (closed smile/slight smile/neutral/serious)
- Smile intensity quantification
- Eye gaze direction and expression
- Eyebrow position
- Overall emotional tone (warm vs neutral vs serious)
- Genuine vs posed quality assessment
- NOTE: Describe the EXPRESSION only, NOT the facial structure (no face shape, no eye color, no skin tone)"""


# ---------- Built-in style transfer prompt ----------

_STYLE_TRANSFER_PROMPT = r"""You are an expert image editing prompt engineer. Analyze this reference image and extract ALL visual characteristics that should be transferred to another image. Then generate a concise, actionable editing instruction.

Extract and describe these elements:

**Subject Appearance:**
- **Clothing**: exact garments, colors, fabrics, fit, style (e.g., "oversized cream knit sweater with a V-neck", "black leather jacket unzipped over a white graphic tee")
- **Pose**: exact body position, angle, weight distribution, hand placement, arm positions, lean direction
- **Facial expression**: exact mouth position (smile type/intensity), eye gaze direction, eyebrow position, overall emotional tone
- **Hair styling**: current style, part, volume, texture state

**Visual Style:**
- **Lighting**: direction, quality (hard/soft), intensity, color temperature, shadow characteristics
- **Color grading**: overall palette, temperature, saturation, contrast, any color tint or cast
- **Mood/atmosphere**: emotional tone, ambiance
- **Texture/grain**: film grain, noise, smoothness, sharpness
- **Depth of field**: bokeh style, focus
- **Post-processing**: filters, vignetting, bloom, haze, matte look
- **Background**: setting type, colors, blur level, key elements

Output ONLY the editing instruction as a single paragraph. No explanations, no bullet points, no headers, no markdown.

Format your output exactly like this example:
"Edit this image so the subject is wearing an oversized cream knit sweater with a V-neck, posed leaning slightly against a wall with their left hand in their pocket and right hand touching their hair, with a relaxed genuine slight smile and direct eye contact, warm golden hour lighting from the left side with soft diffused shadows, desaturated film look with lifted blacks and muted earth tones, subtle film grain, shallow depth of field with a blurred neutral-toned indoor background."

Be extremely specific about clothing, pose, and expression. The output will be fed directly to an AI image editing model."""


# ---------- Built-in Seedream Edit prompt ----------

_SEEDREAM_EDIT_PROMPT = r"""You are Gemini 2.5, an expert prompt engineer specializing in the Seedream 4.5 AI model. You create complete, detailed, and technically precise image generation prompts.
Primary Directive: Your task is to analyze Reference Image 3 (a complete scene) and generate a single, comprehensive prompt for Seedream 4.5. This prompt will instruct the model on how to use a total of three reference images.
Critical Context (Non-negotiable): Seedream will always receive 3 reference images in this specific order:
Images 1 & 2: Provide the subject's complete face structure, facial features, and identity.
Image 3: The complete scene reference (this is the image you will be given to analyze).
Your analysis must focus exclusively on Image 3. Your generated prompt must correctly instruct Seedream on this specific 3-image workflow.
Your Generation Task:
You will be given Image 3.
You will analyze Image 3 ONLY.
You will output ONLY the complete, formatted prompt for Seedream. Do not add any conversational preamble, explanation, or text outside the specified format.
Mandatory Output Format (Strict Template):
Use the first two reference images for the subject's complete face, features, and identity. Use reference image 3 as the complete reference for all other elements: clothing, pose, action, body type, scene composition, background environment, lighting, and overall atmosphere.
Subject details: [Describe the subject's clothing in exhaustive detail: every visible garment (e.g., shirt, jacket, trousers, dress), accessories (e.g., hat, scarf, belt, bag), jewelry (e.g., necklace, earrings, rings, watch), and footwear. Specify colors, patterns, textures (e.g., denim, silk, wool, leather), cuts (e.g., loose-fitting, tailored), and styles (e.g., formal, casual, athletic)]. [Describe the exact pose: sitting, standing, leaning. Detail the position of the torso, arms (e.g., folded, extended, one hand in pocket), legs (e.g., crossed, straight), and head (e.g., tilted, looking forward)]. [Describe the subject's action or gesture (e.g., holding a cup, pointing, walking, reading) and overall body language. Describe the facial expression type (e.g., a wide smile, a serious expression, a thoughtful look, a laugh) but NOT the features.]
The scene: [Describe the location type (e.g., a city street, a living room, a forest, an office)]. The environment features [describe all significant background and foreground elements: architectural details (e.g., buildings, windows, walls), furniture (e.g., chairs, tables, lamps), props (e.g., books, plants, cars), and natural elements (e.g., trees, mountains, water)]. The setting is [describe the spatial layout, e.g., "indoors in a cluttered studio," "outdoors on a crowded beach"].
Lighting: [Describe the lighting in technical detail: identify the primary light source(s) (e.g., sun, studio softbox, window, lamp), its direction (e.g., side-lit, backlit, overhead, three-point lighting), its quality (e.g., hard, soft, diffused), and the resulting shadows (e.g., long and soft, sharp and deep). Note the time of day (e.g., golden hour, midday, night) and the overall color temperature (e.g., warm, cool, neutral).]
Camera: [Describe the camera's properties: the angle (e.g., eye-level, low-angle, high-angle, dutch angle), the shot type (e.g., full-body shot, medium shot, cowboy shot), the depth of field (e.g., shallow with heavy bokeh, deep with everything in focus), and the overall composition (e.g., rule of thirds, centered, leading lines).]
Atmosphere: [Describe the mood or ambiance of the scene (e.g., serene, chaotic, melancholic, energetic, professional, mysterious). If outdoors, note weather conditions (e.g., sunny, overcast, rainy, foggy) or environmental effects (e.g., lens flare, mist).]
Colors and textures: [Describe the dominant color palette of the entire image (e.g., monochrome with a blue tint, vibrant analogous colors, muted complementary colors). Highlight key materials and their surface textures (e.g., smooth glass, rough brick, shiny metal, matte fabric, glossy paint).]
Technical quality: [Describe the image's perceived technical quality: resolution feel (e.g., crisp and high-resolution), noise level (e.g., clean, slight grain), and overall rendering style (e.g., photorealistic, painterly, cinematic).]"""


_SEEDREAM_EDIT_2FACE_BODY_1REF_PROMPT = r"""You are an expert prompt engineer specializing in the Seedream 4.5 AI model. You create complete, detailed, and technically precise image generation prompts.

Primary Directive: Your task is to analyze all 3 provided images and generate a single, comprehensive prompt for Seedream 4.5.

Critical Context (Non-negotiable): Seedream will always receive 3 reference images in this specific order:
Images 1 & 2: Provide the subject's complete face structure, facial features, identity, AND body type, body shape, body proportions, skin tone, and any body-specific details such as tattoos, scars, or markings. The output body MUST match these images.
Image 3: The complete scene reference. Provides ONLY the clothing, pose, action, scene composition, background environment, lighting, and overall atmosphere. The body type in Image 3 should be IGNORED — the body must come from Images 1 & 2.

Your Generation Task:
You will be given all 3 images.
You will analyze Images 1 & 2 for the subject's body proportions and physical build.
You will analyze Image 3 for the scene, clothing, pose, lighting, and atmosphere.
You will output ONLY the complete, formatted prompt for Seedream. Do not add any conversational preamble, explanation, or text outside the specified format.

Mandatory Output Format (Strict Template):
Use the first two reference images for the subject's complete face, features, identity, body type, body shape, proportions, skin tone, and any body-specific details such as tattoos or markings. Use reference image 3 as the reference for clothing, pose, action, scene composition, background environment, lighting, and overall atmosphere ONLY — do NOT use image 3 for body type or physique.
Body proportions (from images 1 & 2): [Describe the subject's body proportions in explicit detail as observed in images 1 & 2. Include: overall build (e.g., slim, athletic, curvy, petite, voluptuous), bust/chest size (e.g., small, medium, large, very large), waist (e.g., narrow, thin, wide), hip and butt size (e.g., small, round, wide, large), leg shape (e.g., long, toned, thick), shoulder width, and any other distinctive body features. Be specific and direct — these proportions MUST be preserved in the output regardless of what Image 3's body looks like.]
Subject details: [Describe the subject's clothing from Image 3 in exhaustive detail: every visible garment (e.g., shirt, jacket, trousers, dress), accessories (e.g., hat, scarf, belt, bag), jewelry (e.g., necklace, earrings, rings, watch), and footwear. Specify colors, patterns, textures (e.g., denim, silk, wool, leather), cuts (e.g., loose-fitting, tailored), and styles (e.g., formal, casual, athletic)]. [Describe the exact pose from Image 3: sitting, standing, leaning. Detail the position of the torso, arms (e.g., folded, extended, one hand in pocket), legs (e.g., crossed, straight), and head (e.g., tilted, looking forward)]. [Describe the subject's action or gesture (e.g., holding a cup, pointing, walking, reading) and overall body language. Describe the facial expression type (e.g., a wide smile, a serious expression, a thoughtful look, a laugh) but NOT the facial features.] [The body wearing these clothes must match images 1 & 2's body proportions exactly. Any tattoos, scars, or markings from images 1 & 2 must be visible on exposed skin and must not appear on top of clothing.]
The scene: [Describe the location type (e.g., a city street, a living room, a forest, an office)]. The environment features [describe all significant background and foreground elements: architectural details (e.g., buildings, windows, walls), furniture (e.g., chairs, tables, lamps), props (e.g., books, plants, cars), and natural elements (e.g., trees, mountains, water)]. The setting is [describe the spatial layout, e.g., "indoors in a cluttered studio," "outdoors on a crowded beach"].
Lighting: [Describe the lighting in technical detail: identify the primary light source(s) (e.g., sun, studio softbox, window, lamp), its direction (e.g., side-lit, backlit, overhead, three-point lighting), its quality (e.g., hard, soft, diffused), and the resulting shadows (e.g., long and soft, sharp and deep). Note the time of day (e.g., golden hour, midday, night) and the overall color temperature (e.g., warm, cool, neutral).]
Camera: [Describe the camera's properties: the angle (e.g., eye-level, low-angle, high-angle, dutch angle), the shot type (e.g., full-body shot, medium shot, cowboy shot), the depth of field (e.g., shallow with heavy bokeh, deep with everything in focus), and the overall composition (e.g., rule of thirds, centered, leading lines).]
Atmosphere: [Describe the mood or ambiance of the scene (e.g., serene, chaotic, melancholic, energetic, professional, mysterious). If outdoors, note weather conditions (e.g., sunny, overcast, rainy, foggy) or environmental effects (e.g., lens flare, mist).]
Colors and textures: [Describe the dominant color palette of the entire image (e.g., monochrome with a blue tint, vibrant analogous colors, muted complementary colors). Highlight key materials and their surface textures (e.g., smooth glass, rough brick, shiny metal, matte fabric, glossy paint).]
Technical quality: [Describe the image's perceived technical quality: resolution feel (e.g., crisp and high-resolution), noise level (e.g., clean, slight grain), and overall rendering style (e.g., photorealistic, painterly, cinematic).]"""


# ---------- Built-in Seedream Edit (1 face + 1 ref) prompt ----------

_SEEDREAM_EDIT_1FACE_PROMPT = r"""You are Gemini 2.5, an expert prompt engineer specializing in the Seedream 4.5 AI model. You create complete, detailed, and technically precise image generation prompts.

Primary Directive: Your task is to analyze Reference Image 2 (a complete scene) and generate a single, comprehensive prompt for Seedream 4.5. This prompt will instruct the model on how to use a total of two reference images.

Critical Context (Non-negotiable): Seedream will always receive 2 reference images in this specific order:
Image 1: Provides the subject's complete face structure, facial features, and identity.
Image 2: The complete scene reference (this is the image you will be given to analyze).

Your analysis must focus exclusively on Image 2. Your generated prompt must correctly instruct Seedream on this specific 2-image workflow.

Your Generation Task:
You will be given Image 2.
You will analyze Image 2 ONLY.
You will output ONLY the complete, formatted prompt for Seedream. Do not add any conversational preamble, explanation, or text outside the specified format.

Mandatory Output Format (Strict Template):
Use reference image 1 for the subject's complete face, features, and identity. Use reference image 2 as the complete reference for all other elements: clothing, pose, action, body type, scene composition, background environment, lighting, and overall atmosphere.
Subject details: [Describe the subject's clothing in exhaustive detail: every visible garment (e.g., shirt, jacket, trousers, dress), accessories (e.g., hat, scarf, belt, bag), jewelry (e.g., necklace, earrings, rings, watch), and footwear. Specify colors, patterns, textures (e.g., denim, silk, wool, leather), cuts (e.g., loose-fitting, tailored), and styles (e.g., formal, casual, athletic)]. [Describe the exact pose: sitting, standing, leaning. Detail the position of the torso, arms (e.g., folded, extended, one hand in pocket), legs (e.g., crossed, straight), and head (e.g., tilted, looking forward)]. [Describe the subject's action or gesture (e.g., holding a cup, pointing, walking, reading) and overall body language. Describe the facial expression type (e.g., a wide smile, a serious expression, a thoughtful look, a laugh) but NOT the features.]
The scene: [Describe the location type (e.g., a city street, a living room, a forest, an office)]. The environment features [describe all significant background and foreground elements: architectural details (e.g., buildings, windows, walls), furniture (e.g., chairs, tables, lamps), props (e.g., books, plants, cars), and natural elements (e.g., trees, mountains, water)]. The setting is [describe the spatial layout, e.g., "indoors in a cluttered studio," "outdoors on a crowded beach"].
Lighting: [Describe the lighting in technical detail: identify the primary light source(s) (e.g., sun, studio softbox, window, lamp), its direction (e.g., side-lit, backlit, overhead, three-point lighting), its quality (e.g., hard, soft, diffused), and the resulting shadows (e.g., long and soft, sharp and deep). Note the time of day (e.g., golden hour, midday, night) and the overall color temperature (e.g., warm, cool, neutral).]
Camera: [Describe the camera's properties: the angle (e.g., eye-level, low-angle, high-angle, dutch angle), the shot type (e.g., full-body shot, medium shot, cowboy shot), the depth of field (e.g., shallow with heavy bokeh, deep with everything in focus), and the overall composition (e.g., rule of thirds, centered, leading lines).]
Atmosphere: [Describe the mood or ambiance of the scene (e.g., serene, chaotic, melancholic, energetic, professional, mysterious). If outdoors, note weather conditions (e.g., sunny, overcast, rainy, foggy) or environmental effects (e.g., lens flare, mist).]
Colors and textures: [Describe the dominant color palette of the entire image (e.g., monochrome with a blue tint, vibrant analogous colors, muted complementary colors). Highlight key materials and their surface textures (e.g., smooth glass, rough brick, shiny metal, matte fabric, glossy paint).]
Technical quality: [Describe the image's perceived technical quality: resolution feel (e.g., crisp and high-resolution), noise level (e.g., clean, slight grain), and overall rendering style (e.g., photorealistic, painterly, cinematic).]"""


_SEEDREAM_EDIT_2FACES_2BODIES_1REF_PROMPT = r"""You are an expert at creating complete image generation prompts for Seedream 4.0 AI model.

IMPORTANT CONTEXT:
- Seedream will receive 5 reference images in this order:
  1. Images 1-2: Face structure references
  2. Images 3-4: Body type and physique references
  3. Image 5: THIS image - complete scene reference
- You are analyzing image 5 ONLY
- Your output must be a COMPLETE prompt for Seedream

YOUR TASK:
Analyze this image and create a complete Seedream prompt that instructs the AI how to use all references and describes everything visible in THIS image.

OUTPUT FORMAT (mandatory structure):
"Use the first two reference images for the face structure. Use reference images 3-4 for the body type and physique. Use reference image 5 as the complete reference for clothing, pose, action, scene composition, background environment, lighting setup, and overall atmosphere.

Subject details: [Describe the person's clothing in complete detail - every garment, accessories, jewelry, shoes, specific details like patterns, textures, colors, cuts, styles]. [Describe the exact pose - standing, sitting, body position, arm placement, leg position]. [Describe what the person is doing - their action, gesture, body language, facial expression like smiling/serious but WITHOUT describing facial features].

The scene: [describe location type and setting]. The environment features [describe architectural elements, furniture, props, and background in detail]. The setting is [indoor/outdoor details with spatial relationships].

Lighting: [describe light source, direction, quality, shadows, time of day, color temperature in technical detail].

Camera: [describe perspective, depth of field, focal distance, composition. For camera angle: default to 'eye-level' unless the angle is clearly extreme. If the camera is only slightly below eye level, describe it as 'approximately eye-level with a very subtle upward perspective.' If slightly above, use 'approximately eye-level, tilted marginally downward.' Always understate the angle by one level to compensate for Seedream's tendency to exaggerate].

Atmosphere: [describe mood, ambiance, weather if applicable, environmental effects].

Colors and textures: [describe dominant colors throughout the scene, materials, surface properties, color palette].

Technical quality: [high-resolution, sharp focus, professional photography, etc.]."

CRITICAL RULES:
- DO describe: clothing (every detail), pose, action, body language, gesture, expression type (smile/serious)
- NEVER describe: hair color, hair style, eye color, facial features, skin tone, ethnic features
- Use "this person", "the subject" when referring to the individual
- Be extremely detailed about clothing and accessories
- Be precise about pose and body position
- Focus on EVERYTHING visible except facial/hair features
- CAMERA ANGLE DAMPENING: Seedream heavily amplifies camera angle descriptions. Always understate angles by one full level. Near eye-level → "eye-level." Slightly low → "approximately eye-level with a very subtle upward perspective." Moderately low → "slightly below eye-level." Only use "low angle" if the camera is clearly at knee height or below. Same rule applies upward.
- BANNED CAMERA TERMS: Never use "low-angle shot," "worm's-eye view," "looking up at," "shot from below," or "high-angle shot" unless the source image shows a genuinely extreme perspective. Prefer neutral terms like "eye-level," "approximately eye-level," or "slightly adjusted."

Output ONLY the formatted prompt, nothing else."""


# ---------- Vertex AI helpers ----------

def _ensure_google_auth(*args, **kwargs):
    return _onyx_core()['_ensure_google_auth'](*args, **kwargs)



def _load_vertex_json_folder(*args, **kwargs):
    return _onyx_core()['_load_vertex_json_folder'](*args, **kwargs)



def _load_vertex_credentials(*args, **kwargs):
    return _onyx_core()['_load_vertex_credentials'](*args, **kwargs)



# ---------- Serve prompts to frontend ----------

try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/ofm_prompt_generator/get_prompts")
    async def _get_prompts(request):
        return web.json_response({
            "JSON Image Analysis": _JSON_ANALYSIS_PROMPT,
            "Style Transfer Prompt": _STYLE_TRANSFER_PROMPT,
            "Seedream Edit (2 face + 1 ref)": _SEEDREAM_EDIT_PROMPT,
            "Seedream Edit (1 face + 1 ref)": _SEEDREAM_EDIT_1FACE_PROMPT,
            "Seedream Edit (2 face and body + 1 ref)": _SEEDREAM_EDIT_2FACE_BODY_1REF_PROMPT,
            "Seedream Edit (2 faces + 2 bodies + 1ref)": _SEEDREAM_EDIT_2FACES_2BODIES_1REF_PROMPT,
            "Custom Prompt": "",
        })
except Exception:
    logging.warning("OFM Prompt Generator: could not register API routes (server not available)")


# ---------- Image encoding helper ----------

def _image_tensor_to_parts(*args, **kwargs):
    return _onyx_core()['_image_tensor_to_parts'](*args, **kwargs)



def _strip_markdown_fences(*args, **kwargs):
    return _onyx_core()['_strip_markdown_fences'](*args, **kwargs)



# ---------- Node ----------

class OnyxGeminiPromptNode:
    """OFM Prompt Generator — analyze images or generate prompts with Gemini."""

    # Cache last known good API keys to survive auto-queue serialization drops
    _cached_gemini_key = ""
    _cached_grok_key = ""
    _cached_vertex_folder = ""

    # Which service account to use next. A folder holds one JSON per GCP project,
    # and Vertex quota is per project — advancing on every call spreads the load
    # instead of hammering the first file until it 429s. Same idea as
    # OnyxNanoBananaAIO._vertex_rotation_offset, applied per call rather than per batch
    # slot, because this node makes exactly one request per run.
    _vertex_rotation_offset = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (_PROMPT_MODES, {"default": "JSON Image Analysis"}),
                "provider": (_PROVIDERS, {"default": "Gemini"}),
                "gemini_api_key": ("STRING", {"default": "", "multiline": False}),
                "grok_api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "vertex_json_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "[VERTEX] Path to the folder containing service account JSON files "
                               "(1 .json file = 1 project). Successive runs rotate through them to "
                               "spread the per-project quota.\n"
                               "Pointing straight at a single .json also works.",
                }),
                "model": (_ALL_MODELS, {
                    "default": "gemini-3.6-flash",
                    "tooltip": "Model. The list shown follows the selected provider:\n"
                               "  Gemini / Vertex -> Gemini text models\n"
                               "  Grok            -> xAI models",
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "System instruction, applied on all three providers "
                               "(Gemini: systemInstruction | Vertex: system_instruction | "
                               "Grok: a leading system message).\n"
                               "Paste a whole .md file here — it sets the persona and the "
                               "rules, and stays separate from the per-run prompt.\n"
                               "On Gemini 3.x this is the main steering lever left, since "
                               "temperature is deprecated there.",
                }),
                "thinking_level": (_THINKING_LEVELS, {
                    "default": "default",
                    "tooltip": "[Gemini / Vertex] Reasoning effort. 'default' sends nothing "
                               "and leaves the model's own default in place.\n"
                               "minimal/low = faster and cheaper (extraction, classification)\n"
                               "medium/high = better on multi-step reasoning\n"
                               "Billed as output tokens. Ignored by Grok, and a level a model "
                               "doesn't support is dropped rather than sent.",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Ignored by Gemini 3.6 Flash and 3.5 Flash-Lite — Google "
                               "deprecated temperature/top_p/top_k there, and the node stops "
                               "sending it rather than pretending it works.\n"
                               "Use system_prompt and thinking_level on those models.",
                }),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 65536, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def generate(self, *args, **kwargs):
        return _onyx_core()['OnyxGeminiPromptNode'].generate(self, *args, **kwargs)


    def _call_gemini(self, *args, **kwargs):
        return _onyx_core()['OnyxGeminiPromptNode']._call_gemini(self, *args, **kwargs)


    def _call_vertex(self, *args, **kwargs):
        return _onyx_core()['OnyxGeminiPromptNode']._call_vertex(self, *args, **kwargs)


    def _call_grok(self, *args, **kwargs):
        return _onyx_core()['OnyxGeminiPromptNode']._call_grok(self, *args, **kwargs)



def _extract_text(*args, **kwargs):
    return _onyx_core()['_extract_text'](*args, **kwargs)



def _response_summary(*args, **kwargs):
    return _onyx_core()['_response_summary'](*args, **kwargs)
