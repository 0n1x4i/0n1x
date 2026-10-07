# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
ComfyUI node - H3 Context-IR (Gemini).

Rebuilds, locally, the preprocessing stage the MiniMax H3 cloud API runs before
H3-Base: read the actual media, reason about it, and hand the video model a
structured, enriched instruction instead of a bare prompt.

Two passes, deliberately separate:

  Pass A - grounding. The real media go to a vision model, which returns plain
  facts as JSON. No H3 formatting at this stage, only "what is in these files".

  Pass B - compilation. That JSON plus the user's intent become an H3 prompt.
  This pass never sees a pixel.

The split is the whole point. Asked to look and to format in one call, the model
skims the media and falls back on generic phrasing, because most of its output
budget goes to satisfying a long format specification. Splitting forces it to
commit to observations first, then write from them.

Pass B's system prompt is the official MiniMax guide, embedded verbatim in this
pack. The rules are never paraphrased here: a paraphrase drifts from the source
the moment the source is updated.

The actual two-pass logic (prompts, media encoding, transcription, provider
calls) lives server-side — see nodes/onyx_remote_exec.py. This file keeps only
what ComfyUI needs to register the node with no network access: INPUT_TYPES,
the widget lists, and guide_folder resolution (public MiniMax docs, not worth
protecting, so kept local rather than duplicated into the remote blob).
"""

import os

from .onyx_h3_guides import GUIDE_BASE_TEXT, GUIDE_REF_TEXT
from .onyx_remote_exec import load_remote


# ─────────────────────────────────────────────────────────────────────────────
# Official guides
# ─────────────────────────────────────────────────────────────────────────────

# Both guides are needed, not one: the reference guide opens by saying its shot,
# camera, speaker and dialogue formats are "shared with" the base guide. Using
# only the reference guide would leave the compiler without the formats it is
# told to reuse.
#
# Embedded verbatim in onyx_h3_guides.py (vendored from MiniMaxAI/MiniMax-H3 at
# revision bfc8ed0353f5a9733be73e6b2c98ec0948195b86) rather than read from a
# separate ComfyUI-MiniMaxH3-Prompt-Writer install on disk: every user of this
# pack gets them automatically, with nothing to clone or point a path at.
_GUIDES_TEXT = GUIDE_BASE_TEXT + "\n\n" + GUIDE_REF_TEXT


def _load_guides(folder: str = "") -> str:
    """Both official guides, verbatim, base first.

    `folder` only matters for the rare case of testing a locally edited guide:
    leave it empty (the default) and the guides built into the pack are used,
    no disk access at all. Point it at a folder holding
    VIDEO_PROMPT_WRITING_GUIDE_base_en.md and VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
    to override them for this run.
    """
    folder = (folder or "").strip()
    if not folder:
        return _GUIDES_TEXT

    if not os.path.isdir(folder):
        raise RuntimeError(
            f"[H3 Context-IR] guide_folder override not found: {folder}\n"
            f"-> Leave 'guide_folder' empty to use the guides built into the pack, "
            f"or point it at a real folder to override them."
        )

    chunks = []
    for name in ("VIDEO_PROMPT_WRITING_GUIDE_base_en.md", "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md"):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"[H3 Context-IR] Missing guide in override folder: {path}\n"
                f"-> Both files are required, or leave 'guide_folder' empty to use "
                f"the guides built into the pack."
            )
        with open(path, "r", encoding="utf-8") as fh:
            chunks.append(fh.read())

    text = "\n\n".join(chunks)
    print(f"📖 [H3 Context-IR] Guides overridden from {folder} ({len(text)} chars)")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Widget lists (needed locally so INPUT_TYPES can build with no network call)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_H3 = "MiniMax H3"
TARGET_SEEDANCE = "Seedance"
TARGET_SEEDANCE_V2 = "Seedance v2"
TARGET_H3_ENDLESS = "MiniMax H3 Endless"
_TARGETS = [TARGET_H3, TARGET_SEEDANCE, TARGET_SEEDANCE_V2, TARGET_H3_ENDLESS]

ROLE_REFERENCE = "reference"
ROLE_FIRST = "first_frame"
ROLE_LAST = "last_frame"
_ROLES = [ROLE_REFERENCE, ROLE_FIRST, ROLE_LAST]

# Verifie sur ai.google.dev/gemini-api/docs/models (page datee du 2026-09-02) :
# 3.8 et 3.7 flash sont tous deux en Stable. Les combos ComfyUI serialisent la
# CHAINE et non l'index, donc reordonner cette liste ne casse aucun workflow.
_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

_PROVIDERS = ["Gemini", "Vertex", "Grok", "Kie"]   # Kie: same Gemini models / Kie Grok models, Kie.ai key

# Meme liste que _GROK_MODELS dans grok_prompt.py, modeles vision en tete.
_GROK_MODELS = [
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-2-vision-1212",
]

# Grok models sold by Kie.ai (provider=Kie only); with Kie, the Gemini models
# above are used as they are.
_KIE_GROK_MODELS = ["grok-4-7", "grok-4-6", "grok-4-5", "grok-4-3"]

# Gemini 3 image token counts, from the media-resolution guide: 1120 at the
# default, 280 at low - a factor of four on every keyframe. The same guide
# recommends low for action recognition and description, which is exactly what
# pass A does with the video frames. Kept as a widget rather than forced to low,
# because the identity and wardrobe grounding does want the detail.
_MEDIA_RES = ["default", "low", "medium", "high"]

_VIDEO_KEYFRAMES = 8


class OnyxH3ContextIR:
    """Two-pass local replacement for the H3 cloud preprocessing stage."""

    # Cle = empreinte des medias + modele. Iterer sur l'intent ne rappelle donc
    # que la passe texte : la passe vision, qui est la lente et la chere, n'est
    # rejouee que si les medias changent reellement. Un dict au niveau classe,
    # pas une variable dans le module distant : le module distant est reexecute
    # a chaque appel (aucun cache en memoire cote serveur, par design), donc ce
    # cache doit vivre ici, en local, et lui etre passe par reference.
    _grounding_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "intent": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Keep this SHORT, and write only what must CHANGE. The official "
                               "Context-IR calls send one sentence — 'Pull focus to the people in "
                               "the background and add more steam to the ramen bowl.' The media are "
                               "read for everything else; describing the scene here competes with "
                               "what pass A actually observed.",
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 8.0, "min": 2.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Target length. The official API takes this as a Context-IR input, "
                               "not just a generation parameter: shot count, pacing and where "
                               "beats land all depend on it. A prompt written blind to duration "
                               "plans a timeline the render cannot hold.",
                }),
                # STRING and not a combo, on purpose: a combo input only accepts a
                # link whose output type is that same combo, so the STRING coming
                # out of Onyx Resolution (MP) could never be wired to it. The
                # value is checked in run() instead, which warns without blocking -
                # any ratio the compiler can read is legitimate here.
                "aspect_ratio": ("STRING", {
                    "default": "adaptive", "multiline": False,
                    "tooltip": "Target framing: adaptive, 16:9, 9:16, 1:1, 4:3, 3:4, 21:9, 3:2, "
                               "2:3, 5:4, 4:5.\n"
                               "'adaptive' is the official default and lets the compiler follow "
                               "the media. Connect the aspect_ratio output of Onyx Resolution "
                               "(MP) to keep it in sync with what you actually render.",
                }),
                "target_model": (_TARGETS, {
                    "default": TARGET_H3,
                    "tooltip": "Which video model the prompt is written for.\n"
                               "MiniMax H3: compiles against the two official MiniMax guides, "
                               "verbatim, with <Picture N> / <Video N> labels.\n"
                               "MiniMax H3 Endless: same guides, written for Onyx Endless Sampler "
                               "(one continuous take, beats every 3-5 s, short dialogue lines "
                               "spoken once).\n"
                               "Seedance: compiles against ByteDance's six-slot grammar with "
                               "@Image 1 / @Video 1 tags.\n"
                               "Seedance v2: same tags, but outputs one structured JSON "
                               "object (people, scene_events with measured dialogue, a single "
                               "continuous shot, negative_prompt, reference_images).\n\n"
                               "The two are not interchangeable. H3's guide asks for an "
                               "exhaustive description; ByteDance's says a long paragraph fights "
                               "itself and that two or three sentences is the target. Pass A does "
                               "not change — it describes media, not a destination.",
                }),
                "provider": (_PROVIDERS, {"default": "Gemini"}),
                # Une seule liste pour les trois providers, comme le fait deja
                # gemini_prompt.py apres la fusion des widgets grok/gemini. Le
                # node verifie la coherence au run et le dit, plutot que de
                # laisser partir un modele Grok vers l'endpoint Google.
                "model": (_MODELS + _GROK_MODELS + _KIE_GROK_MODELS, {"default": "gemini-3.6-flash"}),
                "guide_folder": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Leave empty (default): the two official MiniMax guides are "
                               "built into the pack, nothing to link or install.\n"
                               "Only fill this in to override them with a locally edited "
                               "copy — point it at a folder holding "
                               "VIDEO_PROMPT_WRITING_GUIDE_base_en.md and "
                               "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md.",
                }),
                "max_tokens": ("INT", {"default": 8192, "min": 512, "max": 65536, "step": 512}),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Reference image. Slots are labelled in order: the "
                                                 "first connected slot becomes <Picture 1>, the next "
                                                 "<Picture 2>, and so on. Skipping a slot never leaves "
                                                 "a gap in the numbering."}),
                "image_1_role": (_ROLES, {
                    "default": ROLE_REFERENCE,
                    "tooltip": "What image_1 is FOR, mirroring the official Context-IR payload.\n"
                               "reference: a picture of the subject.\n"
                               "first_frame: this image IS the opening frame of the render.\n"
                               "last_frame: this image IS the closing frame.\n\n"
                               "This is not a mode selector — you describe the asset, the compiler "
                               "still decides the mode on its own.",
                }),
                "image_2": ("IMAGE",),
                "image_2_role": (_ROLES, {
                    "default": ROLE_REFERENCE,
                    "tooltip": "Same as image_1_role. Set image_1 to first_frame and image_2 to "
                               "last_frame for a first-and-last-frame render.",
                }),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "images_batch": ("IMAGE", {"tooltip": "Optional extra images as a batch, appended "
                                                      "AFTER the numbered slots. Useful with the "
                                                      "Onyx Image Batch Loader."}),
                "video": ("IMAGE", {"tooltip": "Source video frames, labelled <Video 1>. "
                                               "Sampled to 8 timecoded keyframes."}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                                        "tooltip": "Used to write the timecode on each keyframe. "
                                                   "Wrong here means wrong shot boundaries."}),
                "media_resolution": (_MEDIA_RES, {
                    "default": "low",
                    "tooltip": "Token budget per image sent to pass A. Gemini 3: 1120 at default, "
                               "280 at low — a factor of four on every keyframe.\n"
                               "Google recommends low for action recognition and description, "
                               "which is what the video frames are for. Raise it only if the "
                               "grounding starts missing wardrobe or face detail, and consider "
                               "spending the saving on more keyframes instead.",
                }),
                "video_keyframes": ("INT", {
                    "default": 8, "min": 2, "max": 32, "step": 1,
                    "tooltip": "How many frames of the video pass A actually looks at.\n"
                               "The compiler writes the action sequence from these and nothing "
                               "else, so this sets the resolution of the timeline it can describe. "
                               "8 over a 15 s clip is one frame every 1.9 s — enough to say what "
                               "happens, not enough to say exactly how.\n"
                               "Roughly one per second is a good target. Each frame costs vision "
                               "tokens, so raise it for long or busy clips, not for everything."}),
                "video_audio": ("AUDIO", {
                    "tooltip": "The soundtrack that belongs to the video above. It is described as "
                               "part of <Video 1>, not as a separate asset.\n"
                               "This is what fills overall_soundscape with dated events instead of "
                               "a generic line."}),
                "ref_audio_1": ("AUDIO", {
                    "tooltip": "A standalone audio reference, labelled <Audio 1> — a voice timbre "
                               "to follow, a music bed, a sound to reproduce. Separate from the "
                               "video's own soundtrack, exactly as reference_audio is separate from "
                               "reference_video in the official payload."}),
                "ref_audio_2": ("AUDIO", {"tooltip": "Second standalone audio reference, <Audio 2>."}),
                "gemini_api_key": ("STRING", {"default": "", "multiline": False}),
                "grok_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "xAI key (console.x.ai), for provider=Grok.\n"
                               "Audio assets are not sent to Grok: its image input is documented "
                               "and already used elsewhere in this pack, its audio input is not "
                               "verified. Use Gemini or Vertex when the soundtrack matters."}),
                "vertex_json_folder": ("STRING", {"default": "", "multiline": False}),
                "grounding_override": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Paste a corrected grounding JSON to skip pass A entirely and "
                               "recompile without another vision call.",
                }),
                "window_start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.001,
                    "tooltip": "Where this render starts inside the supplied source, in seconds.\n"
                               "Leave at 0 for a normal one-shot render.\n"
                               "For a chained render, give every segment the SAME full video and "
                               "only change this: the grounding is then computed once and served "
                               "from cache, and each segment gets its own window with timestamps "
                               "re-based to zero.\n"
                               "Added LAST on purpose: ComfyUI serialises widget values by "
                               "position, so inserting it higher would shift every widget in "
                               "already-saved workflows."}),
                "pinned_lead_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.001,
                    "tooltip": "How many seconds at the START of this render are already fixed "
                               "by a guide, and must therefore NOT be described.\n"
                               "For a chained render: connect the Segment node's `guide_seconds`. "
                               "Those frames are pinned by AddGuide — image AND audio — so any "
                               "line of dialogue the compiler places inside them is rendered a "
                               "second time and then thrown away by the Join. That is what makes "
                               "a character repeat a sentence across a seam.\n"
                               "The timeline origin does not move: 00:00.000 stays the first "
                               "frame of the render, because H3 encodes time absolutely and the "
                               "frames really are being generated. Only the DESCRIPTION starts "
                               "later."}),
                "transcribe_speech": ("BOOLEAN", {
                    "default": False,
                    "label_on": "measure speech timings",
                    "label_off": "off",
                    "tooltip": "Runs gemini-3.5-transcribe on the soundtrack first and gives both "
                               "passes the MEASURED start and end time of every spoken word.\n"
                               "Pass A is a vision pass: asked when a line is spoken it reads "
                               "keyframes and estimates, and it estimates badly. This replaces the "
                               "guess with a measurement.\n"
                               "REQUIRES a gemini_api_key: it uses the Interactions API, which is "
                               "documented on the Gemini API and which I could not confirm on "
                               "Vertex. Without a key the pass is skipped and the node behaves "
                               "exactly as before.\n"
                               "Costs one call per soundtrack for the whole render — it is cached "
                               "on disk beside the grounding, so a four-segment chain transcribes "
                               "once."}),
                "extra_prompt_rules": ("BOOLEAN", {
                    "default": False,
                    "label_on": "add the rules written after the reference run",
                    "label_off": "prompt as of the last known-good render",
                    "tooltip": "OFF restores the Pass B prompt to the shape it had for the render "
                               "you judged near-perfect, before a week of accumulated rules.\n\n"
                               "What it removes:\n"
                               "• 'Never write stillness' — 1464 characters that were always on\n"
                               "• the 'Render window' block on segment 1 (segments 2+ keep theirs, "
                               "they need the re-basing)\n\n"
                               "Each rule fixed something real. What was never measured is their "
                               "combined effect: they stack in the same context as the official "
                               "guide, and the compiler averages them. Turn them back on one at a "
                               "time if a defect they addressed comes back.",
                }),
                "keyframes_match_model": ("BOOLEAN", {
                    "default": False,
                    "label_on": "same frames as the model (2 fps)",
                    "label_off": "evenly spaced (video_keyframes)",
                    "tooltip": "Samples the reference video EXACTLY as H3 does — every 12th frame "
                               "at 24 fps, labelled 0.0, 0.5, 1.0 …\n\n"
                               "H3 hands its text encoder the video at 2 fps and nothing else "
                               "(nodes_minimax_h3.py: range(0, n, FPS // 2)). Evenly spaced "
                               "keyframes land on OTHER frames with OTHER labels, so a beat "
                               "written at 00:07.333 points at an instant the model does not "
                               "have — its strip only holds 7.0 and 7.5.\n\n"
                               "With this on, pass A describes the very images the model will "
                               "see, and pass B is told to put action beats on the same 0.5 s "
                               "grid.\n"
                               "video_keyframes is ignored: the stride belongs to the model. A "
                               "294-frame segment gives 25 frames either way, so it costs "
                               "nothing.",
                }),
                "motion_from_video_only": ("BOOLEAN", {
                    "default": False,
                    "label_on": "the video owns the timing",
                    "label_off": "the prompt may time actions",
                    "tooltip": "Stops the prompt from putting timestamps on physical actions.\n\n"
                               "In a reference-video render the footage already fixes WHEN every "
                               "movement happens, frame by frame. A timestamp in the prompt is a "
                               "SECOND answer to the same question — and it is an estimate: pass A "
                               "reads keyframes and misjudges by around a second. The model then "
                               "has two sources that disagree and splits the difference, which is "
                               "what 'the same scene, played differently' looks like.\n\n"
                               "Turn this on for motion control. Dialogue keeps its timestamps: "
                               "H3 generates the audio, so nothing else says when a line is "
                               "spoken.",
                }),
                # Added LAST on purpose: ComfyUI stores widget values by position,
                # anything inserted higher would shift every saved workflow.
                "kie_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Kie.ai key, for provider=Kie: the Gemini models of the list, or "
                               "grok-4-7 / 4-6 / 4-5 / 4-3, billed on your Kie.ai credits."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "grounding_json")
    FUNCTION = "run"
    CATEGORY = "Onyx/Prompt"
    DESCRIPTION = (
        "Local rebuild of the H3 cloud preprocessing stage. Pass A reads the real media "
        "and reports facts as JSON; pass B compiles that JSON plus your intent into an H3 "
        "prompt, using the official MiniMax guides verbatim as its system prompt."
    )

    def run(self, intent, duration_seconds, aspect_ratio, target_model,
            provider, model, guide_folder="", max_tokens=8192,
            image_1=None, image_1_role=ROLE_REFERENCE,
            image_2=None, image_2_role=ROLE_REFERENCE,
            image_3=None, image_4=None,
            image_5=None, image_6=None, images_batch=None,
            video=None, video_fps=24.0, media_resolution="low",
            video_keyframes=_VIDEO_KEYFRAMES,
            video_audio=None, ref_audio_1=None, ref_audio_2=None,
            gemini_api_key="", grok_api_key="", vertex_json_folder="",
            grounding_override="", window_start_seconds=0.0,
            pinned_lead_seconds=0.0, transcribe_speech=False,
            motion_from_video_only=False, keyframes_match_model=False,
            extra_prompt_rules=False, kie_api_key=""):
        ensure_profile_ready()

        guides_text = _load_guides(guide_folder)

        ns = load_remote("h3_context_ir_core", extra_globals={"__file__": __file__})
        # only sent when used: a server-side core older than Kie still runs the rest
        kie = {"kie_api_key": kie_api_key} if (kie_api_key or "").strip() else {}
        return ns["run_impl"](
            OnyxH3ContextIR._grounding_cache, guides_text,
            intent, duration_seconds, aspect_ratio, target_model,
            provider, model, max_tokens,
            image_1, image_1_role, image_2, image_2_role,
            image_3, image_4, image_5, image_6, images_batch,
            video, video_fps, media_resolution, video_keyframes,
            video_audio, ref_audio_1, ref_audio_2,
            gemini_api_key, grok_api_key, vertex_json_folder,
            grounding_override, window_start_seconds,
            pinned_lead_seconds, transcribe_speech,
            motion_from_video_only, keyframes_match_model,
            extra_prompt_rules, **kie,
        )


NODE_CLASS_MAPPINGS = {"OnyxH3ContextIR": OnyxH3ContextIR}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxH3ContextIR": "H3 Context-IR (Gemini)"}
