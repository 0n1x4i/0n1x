"""ComfyUI custom nodes for Nano Banana Pro and Nano Banana 2 APIs (Monthly)."""

# ── OpenCV : verification des installations en double ──────────────────────
# Plusieurs builds OpenCV (opencv-python, -headless, -contrib...) partagent le
# meme dossier cv2 et se cassent mutuellement. Avant, le pack lancait pip tout
# seul a chaque demarrage. Desormais : simple detection + message, et la
# reparation ne tourne que si l'utilisateur l'a demandee via le bouton du
# dialogue (js/onyx_opencv_check.js). Elle s'execute alors ICI, au prochain
# demarrage, avant le tout premier import de cv2 (sur Windows un .pyd charge
# est verrouille). Voir utils/onyx_opencv.py.
try:
    from .utils import onyx_opencv as _onyx_opencv
    _onyx_opencv.startup_check()
except Exception as _e:
    _onyx_opencv = None
    print(f"[Onyx] OpenCV check skipped: {_e}")

from .Node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .gemini_prompt import OnyxGeminiPromptNode
from .prompt_selector import PromptSelectorNode
from .directory_loader import OnyxDirectoryImageLoaderNode
from .api_keys import OnyxLoadAPIKeysNode
from .video_frame_extractor import OnyxVideoFrameExtractorNode
from .image_processing import MetadataRemoveNode, SaveAsPhonePhotoNode
from .lora_caption import OnyxLoraCaptionGeneratorNode
from .nodes.nano_banana_aio import OnyxNanoBananaAIO
from .nodes.imageUtils import OnyxPreviewImageWithoutMetadata
from .nodes.onyx_video_loader import OnyxVideoLoader
from .nodes.instagram_faceswap import OnyxInstagramFaceSwapNode
from .nodes.dataset_creator          import OnyxDatasetCreatorNode
from .nodes.onyx_image_batch_loader import OnyxImageBatchLoader
from .nodes.repose_carousel import OnyxReposeCarouselNode
from .nodes.metadata_bypass import OnyxMetadataBypassNode
from .nodes.image_enhancement import OnyxImageEnhancementNode
from .nodes.image_black_check import OnyxImageBlackCheckNode
from .nodes.grok_prompt import OnyxGrokPromptNode
from .nodes.onyx_resolution_mp import OnyxResolutionMP
from .nodes.onyx_scale_long_side import OnyxScaleLongSide
from .nodes.onyx_h3_frame_snap import OnyxH3FrameSnap
from .nodes.onyx_audio_switch import OnyxAudioSwitch
from .nodes.onyx_lora_trigger import OnyxLoraTrigger
from .nodes.onyx_lora_trigger_stack import OnyxLoraTriggerStack
from .nodes.onyx_face_dataset_prompts import OnyxFaceDatasetPrompts
from .nodes.onyx_face_mask import OnyxFaceMask
from .nodes.h3_context_ir import OnyxH3ContextIR
from .nodes.onyx_group_toggle import OnyxGroupToggle
from .nodes.onyx_image_blur_batched import OnyxImageBlurBatched, OnyxImageCompositeMaskedBatched
from .nodes.onyx_images_to_video import OnyxImagesToVideo
from .nodes.onyx_animate2_infinity import OnyxAnimate2Infinity
from .nodes.onyx_save_video_no_metadata import OnyxSaveVideoNoMetadata
from .nodes.onyx_segment_cache import OnyxSegmentCache
from .nodes.onyx_rife_batched import OnyxRifeVfiBatched
from .nodes.onyx_prompt_library import OnyxPromptSaver, OnyxPromptGallery
from .nodes.onyx_video_chain import (OnyxVideoChainSegment, OnyxVideoChainJoin,
                                         OnyxVideoChainPrepare, OnyxVideoChainCommit)
from .nodes.save_image_no_metadata import OnyxSaveImageNoMetadataNode
from .nodes.onyx_speed_hd_sampler import OnyxSpeedHDSampler
from .nodes.onyx_eye_detailer import OnyxEyeBBoxDetectorProvider, OnyxDetailer
from .nodes.onyx_temporal_mask_smooth import OnyxTemporalMaskSmooth
from .nodes.onyx_depth_colormap import OnyxDepthColormap
from .nodes.onyx_mask_to_box import OnyxMaskToBox
from .nodes.onyx_session import OnyxSessionNode
from .nodes.onyx_h3_vfx import OnyxH3VfxLora
from .nodes.onyx_conditioning_cache import OnyxConditioningCache
from .nodes.onyx_studio import OnyxStudio
from .nodes.onyx_camera_shake import OnyxCameraShake
from .nodes.onyx_post_style import OnyxPostStyle
from .nodes.onyx_post_studio import OnyxPostStudio
from .nodes.onyx_phone_fx import NODE_CLASS_MAPPINGS as _PHONE_FX, NODE_DISPLAY_NAME_MAPPINGS as _PHONE_FX_NAMES
from .nodes.onyx_lens_character import OnyxLensCharacter
from .nodes import onyx_post_studio as _onyx_post_studio_mod
from .nodes.onyx_famegrid import OnyxFameGridColorFinish
from .nodes.onyx_more_jpeg import OnyxMoreJPEG
from .nodes.onyx_realism_injector import OnyxRealismInjector
try:   # needs spandrel (shipped with ComfyUI) for its optional upscale stage
    from .nodes.onyx_post_suite import OnyxPostProcessSuite
except Exception as _e:
    OnyxPostProcessSuite = None
    print(f"[Onyx] ❌ Post-Process Suite failed: {_e}")
from .nodes import onyx_studio as _onyx_studio_mod

# Post-processing nodes (dossier avec espace + noms unicode -> importlib)
import importlib.util as _ilu, os as _os

_pack_root = _os.path.dirname(_os.path.abspath(__file__))

def _load_pp(filename, modname):
    _path = _os.path.join(_pack_root, "post processing", filename)
    _spec = _ilu.spec_from_file_location(modname, _path)
    _mod  = _ilu.module_from_spec(_spec)
    _mod.__package__ = __package__  # lets the module reach the pack's protected-code loader
    _spec.loader.exec_module(_mod)
    return _mod

try:
    _renoise_mod = _load_pp("Onyx_Renoïse.py", "Onyx_Renoise_mod")
    Onyx_Renoise = _renoise_mod.Onyx_Renoise
    print("[Onyx] ✅ Renoïse chargé")
except Exception as _e:
    Onyx_Renoise = None
    print(f"[Onyx] ❌ Renoïse failed: {_e}")

try:
    _camera_look_mod = _load_pp("Onyx_camera_look.py", "Onyx_Camera_Look_mod")
    Onyx_Camera_Look = _camera_look_mod.Onyx_Camera_Look
    print("[Onyx] ✅ Camera Look chargé")
except Exception as _e:
    Onyx_Camera_Look = None
    print(f"[Onyx] ❌ Camera Look failed: {_e}")

try:
    _apply_lut_mod = _load_pp("Onyx_Apply_LUT.py", "Onyx_Apply_LUT_mod")
    Onyx_Apply_LUT = _apply_lut_mod.Onyx_Apply_LUT
    print("[Onyx] ✅ Apply LUT chargé")
except Exception as _e:
    Onyx_Apply_LUT = None
    print(f"[Onyx] ❌ Apply LUT failed: {_e}")

try:
    _cd_mod = _load_pp("Onyx_Clarity_Dehaze.py", "Onyx_Clarity_Dehaze_mod")
    Onyx_Clarity      = _cd_mod.Onyx_Clarity
    Onyx_Dehaze       = _cd_mod.Onyx_Dehaze
    Onyx_Match_Levels = _cd_mod.Onyx_Match_Levels
    print("[Onyx] ✅ Clarity / Dehaze / Match Levels chargés")
except Exception as _e:
    Onyx_Clarity = Onyx_Dehaze = Onyx_Match_Levels = None
    print(f"[Onyx] ❌ Clarity/Dehaze failed: {_e}")

try:
    _st_mod = _load_pp("Onyx_Skin_Tone.py", "Onyx_Skin_Tone_mod")
    Onyx_Skin_Tone = _st_mod.Onyx_Skin_Tone
    print("[Onyx] ✅ Skin Tone Line chargé")
except Exception as _e:
    Onyx_Skin_Tone = None
    print(f"[Onyx] ❌ Skin Tone failed: {_e}")

# Krea2 Carousel (porte V1 du pack NodoForgeLab2 Beta v2.3) — 5 nodes + routes /onyx/nf/*.
# Charge dans un try : si transformers/huggingface_hub manquent, le reste d'Onyx demarre quand meme.
try:
    from .nodes.krea2_carousel import (
        NODE_CLASS_MAPPINGS as _K2C_NODES,
        NODE_DISPLAY_NAME_MAPPINGS as _K2C_NAMES,
    )
    print(f"[Onyx] ✅ Krea2 Carousel chargé ({len(_K2C_NODES)} nodes)")
except Exception as _e:
    _K2C_NODES, _K2C_NAMES = {}, {}
    print(f"[Onyx] ❌ Krea2 Carousel failed: {_e}")

WEB_DIRECTORY = "./js"

# Routes partagees : /onyx/registry (listes de modeles + prix), /onyx/opencv/*,
# /onyx/costs/* — et rafraichissement du registre depuis le Worker en arriere-plan.
try:
    from .utils import onyx_registry as _onyx_registry
    _onyx_registry.register_routes()
    _onyx_registry.start_background_refresh()
    from .utils import onyx_cost as _onyx_cost
    _onyx_cost.register_routes()
    from .utils import onyx_setup_check as _onyx_setup_check
    _onyx_setup_check.register_routes()
    _onyx_studio_mod.register_routes()
    _onyx_post_studio_mod.register_routes()
except Exception as _e:
    print(f"[Onyx] Registry / cost routes failed: {_e}")
if _onyx_opencv is not None:
    try:
        _onyx_opencv.register_routes()
    except Exception as _e:
        print(f"[Onyx] OpenCV routes failed: {_e}")

NODE_CLASS_MAPPINGS["OnyxNanoBananaAIO"] = OnyxNanoBananaAIO
NODE_CLASS_MAPPINGS["OnyxGeminiPromptNode"] = OnyxGeminiPromptNode
NODE_CLASS_MAPPINGS["OnyxPromptSelectorNodeMonthly"] = PromptSelectorNode
NODE_CLASS_MAPPINGS["OnyxDirectoryImageLoaderNode"] = OnyxDirectoryImageLoaderNode
NODE_CLASS_MAPPINGS["OnyxLoadAPIKeysNode"] = OnyxLoadAPIKeysNode
NODE_CLASS_MAPPINGS["OnyxVideoFrameExtractorNode"] = OnyxVideoFrameExtractorNode
NODE_CLASS_MAPPINGS["OnyxMetadataRemoveNodeMonthly"] = MetadataRemoveNode
NODE_CLASS_MAPPINGS["OnyxSaveAsPhonePhotoNodeMonthly"] = SaveAsPhonePhotoNode
NODE_CLASS_MAPPINGS["OnyxLoraCaptionGeneratorNode"] = OnyxLoraCaptionGeneratorNode
NODE_CLASS_MAPPINGS["OnyxPreviewImageWithoutMetadata"] = OnyxPreviewImageWithoutMetadata
NODE_CLASS_MAPPINGS["OnyxVideoLoader"] = OnyxVideoLoader
NODE_CLASS_MAPPINGS["OnyxInstagramFaceSwapNode"] = OnyxInstagramFaceSwapNode
NODE_CLASS_MAPPINGS["OnyxDatasetCreatorNode"]        = OnyxDatasetCreatorNode
NODE_CLASS_MAPPINGS["OnyxImageBatchLoader"] = OnyxImageBatchLoader
NODE_CLASS_MAPPINGS["OnyxReposeCarouselNode"] = OnyxReposeCarouselNode
NODE_CLASS_MAPPINGS["OnyxMetadataBypassNode"] = OnyxMetadataBypassNode
NODE_CLASS_MAPPINGS["OnyxImageEnhancementNode"] = OnyxImageEnhancementNode
NODE_CLASS_MAPPINGS["OnyxImageBlackCheckNode"] = OnyxImageBlackCheckNode
NODE_CLASS_MAPPINGS["OnyxGrokPromptNode"] = OnyxGrokPromptNode
NODE_CLASS_MAPPINGS["OnyxResolutionMP"] = OnyxResolutionMP
NODE_CLASS_MAPPINGS["OnyxScaleLongSide"] = OnyxScaleLongSide
NODE_CLASS_MAPPINGS["OnyxH3FrameSnap"] = OnyxH3FrameSnap
NODE_CLASS_MAPPINGS["OnyxImageBlurBatched"] = OnyxImageBlurBatched
NODE_CLASS_MAPPINGS["OnyxImageCompositeMaskedBatched"] = OnyxImageCompositeMaskedBatched
NODE_CLASS_MAPPINGS["OnyxImagesToVideo"] = OnyxImagesToVideo
NODE_CLASS_MAPPINGS["OnyxAnimate2Infinity"] = OnyxAnimate2Infinity
NODE_CLASS_MAPPINGS["OnyxSaveVideoNoMetadata"] = OnyxSaveVideoNoMetadata
NODE_CLASS_MAPPINGS["OnyxSegmentCache"] = OnyxSegmentCache
NODE_CLASS_MAPPINGS["OnyxRifeVfiBatched"] = OnyxRifeVfiBatched
NODE_CLASS_MAPPINGS["OnyxPromptSaver"] = OnyxPromptSaver
NODE_CLASS_MAPPINGS["OnyxPromptGallery"] = OnyxPromptGallery
NODE_CLASS_MAPPINGS["OnyxVideoChainSegment"] = OnyxVideoChainSegment
NODE_CLASS_MAPPINGS["OnyxVideoChainJoin"] = OnyxVideoChainJoin
NODE_CLASS_MAPPINGS["OnyxVideoChainPrepare"] = OnyxVideoChainPrepare
NODE_CLASS_MAPPINGS["OnyxVideoChainCommit"] = OnyxVideoChainCommit
NODE_CLASS_MAPPINGS["OnyxAudioSwitch"] = OnyxAudioSwitch
NODE_CLASS_MAPPINGS["OnyxLoraTrigger"] = OnyxLoraTrigger
NODE_CLASS_MAPPINGS["OnyxLoraTriggerStack"] = OnyxLoraTriggerStack
NODE_CLASS_MAPPINGS["OnyxFaceDatasetPrompts"] = OnyxFaceDatasetPrompts
NODE_CLASS_MAPPINGS["OnyxFaceMask"] = OnyxFaceMask
NODE_CLASS_MAPPINGS["OnyxH3ContextIR"] = OnyxH3ContextIR
NODE_CLASS_MAPPINGS["OnyxGroupToggle"] = OnyxGroupToggle
NODE_CLASS_MAPPINGS["OnyxSaveImageNoMetadataNode"] = OnyxSaveImageNoMetadataNode
NODE_CLASS_MAPPINGS["OnyxSpeedHDSampler"] = OnyxSpeedHDSampler
NODE_CLASS_MAPPINGS["OnyxEyeBBoxDetectorProvider"] = OnyxEyeBBoxDetectorProvider
NODE_CLASS_MAPPINGS["OnyxDetailer"] = OnyxDetailer
NODE_CLASS_MAPPINGS["OnyxTemporalMaskSmooth"] = OnyxTemporalMaskSmooth
NODE_CLASS_MAPPINGS["OnyxDepthColormap"] = OnyxDepthColormap
NODE_CLASS_MAPPINGS["OnyxMaskToBox"] = OnyxMaskToBox
NODE_CLASS_MAPPINGS["OnyxSessionNode"] = OnyxSessionNode
NODE_CLASS_MAPPINGS["OnyxH3VfxLora"] = OnyxH3VfxLora
NODE_CLASS_MAPPINGS["OnyxConditioningCache"] = OnyxConditioningCache
NODE_CLASS_MAPPINGS["OnyxStudio"] = OnyxStudio
NODE_CLASS_MAPPINGS["OnyxCameraShake"] = OnyxCameraShake
NODE_CLASS_MAPPINGS["OnyxPostStyle"] = OnyxPostStyle
NODE_CLASS_MAPPINGS["OnyxPostStudio"] = OnyxPostStudio
NODE_CLASS_MAPPINGS.update(_PHONE_FX)
NODE_CLASS_MAPPINGS["OnyxLensCharacter"] = OnyxLensCharacter
NODE_CLASS_MAPPINGS["OnyxFameGridColorFinish"] = OnyxFameGridColorFinish
NODE_CLASS_MAPPINGS["OnyxMoreJPEG"] = OnyxMoreJPEG
NODE_CLASS_MAPPINGS["OnyxRealismInjector"] = OnyxRealismInjector
if OnyxPostProcessSuite:
    NODE_CLASS_MAPPINGS["OnyxPostProcessSuite"] = OnyxPostProcessSuite
# Onyx Post looks: every post-processing node gets the optional `onyx_post` input
# (Onyx Studio / Onyx Post Style), applied around its own code.
from .utils import onyx_post as _onyx_post_mod
for _pp_cls in (Onyx_Renoise, Onyx_Camera_Look, Onyx_Apply_LUT, Onyx_Clarity, Onyx_Dehaze,
                Onyx_Match_Levels, Onyx_Skin_Tone, OnyxLensCharacter):
    if _pp_cls is not None:
        _onyx_post_mod.wire(_pp_cls)

if Onyx_Renoise:     NODE_CLASS_MAPPINGS["Onyx_Renoise"]     = Onyx_Renoise
if Onyx_Camera_Look: NODE_CLASS_MAPPINGS["Onyx_Camera_Look"] = Onyx_Camera_Look
if Onyx_Apply_LUT:   NODE_CLASS_MAPPINGS["Onyx_Apply_LUT"]   = Onyx_Apply_LUT
if Onyx_Clarity:      NODE_CLASS_MAPPINGS["Onyx_Clarity"]      = Onyx_Clarity
if Onyx_Dehaze:       NODE_CLASS_MAPPINGS["Onyx_Dehaze"]       = Onyx_Dehaze
if Onyx_Match_Levels: NODE_CLASS_MAPPINGS["Onyx_Match_Levels"] = Onyx_Match_Levels
if Onyx_Skin_Tone:    NODE_CLASS_MAPPINGS["Onyx_Skin_Tone"]    = Onyx_Skin_Tone

NODE_DISPLAY_NAME_MAPPINGS["OnyxNanoBananaAIO"] = "Onyx Image and Video Edit AIO"
NODE_DISPLAY_NAME_MAPPINGS["OnyxGeminiPromptNode"] = "Onyx Prompt Generator"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPromptSelectorNodeMonthly"] = "Onyx Prompt Selector"
NODE_DISPLAY_NAME_MAPPINGS["OnyxDirectoryImageLoaderNode"] = "Onyx Directory Image Loader"
NODE_DISPLAY_NAME_MAPPINGS["OnyxLoadAPIKeysNode"] = "Onyx Load API Keys"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoFrameExtractorNode"] = "Onyx Video Frame Extractor"
NODE_DISPLAY_NAME_MAPPINGS["OnyxMetadataRemoveNodeMonthly"] = "Onyx Remove Image Metadata"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSaveAsPhonePhotoNodeMonthly"] = "Onyx Save as Phone Photo"
NODE_DISPLAY_NAME_MAPPINGS["OnyxLoraCaptionGeneratorNode"] = "Onyx Lora Caption Generator"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPreviewImageWithoutMetadata"] = "Onyx Preview No Metadata"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoLoader"]      = "Onyx Video Loader"
NODE_DISPLAY_NAME_MAPPINGS["OnyxInstagramFaceSwapNode"]   = "Onyx Content Remaker"
NODE_DISPLAY_NAME_MAPPINGS["OnyxDatasetCreatorNode"]        = "Onyx Dataset Creator"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageBatchLoader"] = "Onyx Image and Video Batch Loader"
NODE_DISPLAY_NAME_MAPPINGS["OnyxReposeCarouselNode"] = "Onyx Re-pose for Carousel"
NODE_DISPLAY_NAME_MAPPINGS["OnyxMetadataBypassNode"] = "Onyx Metadata Bypass"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageEnhancementNode"] = "Onyx Image Enhancement"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageBlackCheckNode"] = "Onyx Image Black Check"
NODE_DISPLAY_NAME_MAPPINGS["OnyxGrokPromptNode"] = "Onyx Grok Prompt Generator"
NODE_DISPLAY_NAME_MAPPINGS["OnyxResolutionMP"] = "Onyx Resolution (MP)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxScaleLongSide"] = "Onyx Scale Long Side (MP grid)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxH3FrameSnap"] = "Onyx H3 Frame Snap"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageBlurBatched"] = "Onyx Image Blur (batched)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageCompositeMaskedBatched"] = "Onyx Image Composite Masked (batched)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImagesToVideo"] = "Onyx Images to Video (AB_VIDEO)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxAnimate2Infinity"] = "Onyx Animate 2 Infinity"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSaveVideoNoMetadata"] = "Onyx Save Video (no metadata)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSegmentCache"] = "Onyx Segment Cache"
NODE_DISPLAY_NAME_MAPPINGS["OnyxRifeVfiBatched"] = "Onyx RIFE VFI (batched)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPromptSaver"] = "Onyx Prompt Saver"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPromptGallery"] = "Onyx Prompt Gallery"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoChainSegment"] = "Onyx Video Chain — Segment"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoChainJoin"] = "Onyx Video Chain — Join"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoChainPrepare"] = "Onyx Video Chain — Prepare"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoChainCommit"] = "Onyx Video Chain — Commit"
NODE_DISPLAY_NAME_MAPPINGS["OnyxAudioSwitch"] = "Onyx Audio Switch"
NODE_DISPLAY_NAME_MAPPINGS["OnyxLoraTrigger"] = "Onyx LoRA Trigger"
NODE_DISPLAY_NAME_MAPPINGS["OnyxLoraTriggerStack"] = "Onyx LoRA Trigger Stack"
NODE_DISPLAY_NAME_MAPPINGS["OnyxFaceDatasetPrompts"] = "Onyx Face Dataset Prompts"
NODE_DISPLAY_NAME_MAPPINGS["OnyxFaceMask"] = "Onyx Face Mask"
NODE_DISPLAY_NAME_MAPPINGS["OnyxH3ContextIR"] = "H3 Context-IR (Gemini)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxGroupToggle"] = "Onyx Group Toggle"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSaveImageNoMetadataNode"] = "Onyx Save Image No Metadata"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSpeedHDSampler"] = "Onyx Speed HD Sampler"
NODE_DISPLAY_NAME_MAPPINGS["OnyxEyeBBoxDetectorProvider"] = "Onyx HD Ultralytic BBox Loader"
NODE_DISPLAY_NAME_MAPPINGS["OnyxDetailer"] = "Onyx Detailer"
NODE_DISPLAY_NAME_MAPPINGS["OnyxTemporalMaskSmooth"] = "Onyx Temporal Mask Smooth"
NODE_DISPLAY_NAME_MAPPINGS["OnyxDepthColormap"] = "Onyx Depth Colormap"
NODE_DISPLAY_NAME_MAPPINGS["OnyxMaskToBox"] = "Onyx Mask To Box"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSessionNode"] = "Onyx Session"
NODE_DISPLAY_NAME_MAPPINGS["OnyxH3VfxLora"] = "Onyx H3 VFX LoRA"
NODE_DISPLAY_NAME_MAPPINGS["OnyxConditioningCache"] = "Onyx Conditioning Cache (text encode)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxStudio"] = "Onyx Studio ✦"
NODE_DISPLAY_NAME_MAPPINGS["OnyxCameraShake"] = "Onyx Camera Shake"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPostStyle"] = "Onyx Post Style"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPostStudio"] = "Onyx Post Studio ✦"
NODE_DISPLAY_NAME_MAPPINGS.update(_PHONE_FX_NAMES)
NODE_DISPLAY_NAME_MAPPINGS["OnyxLensCharacter"] = "Onyx Lens Character"
NODE_DISPLAY_NAME_MAPPINGS["OnyxFameGridColorFinish"] = "Onyx FameGrid Color Finish"
NODE_DISPLAY_NAME_MAPPINGS["OnyxMoreJPEG"] = "Onyx More JPEG"
NODE_DISPLAY_NAME_MAPPINGS["OnyxRealismInjector"] = "Onyx Realism Injector"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPostProcessSuite"] = "Onyx Post-Process Suite"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Renoise"]     = "Onyx Renoise"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Camera_Look"] = "📷 Onyx Camera Look"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Apply_LUT"]   = "🎨 Onyx Apply LUT"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Clarity"]      = "✨ Onyx Clarity"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Dehaze"]       = "🌫️ Onyx Dehaze"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Match_Levels"] = "🎚️ Onyx Match Levels (vs pre-refine)"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Skin_Tone"]    = "🧑 Onyx Skin Tone Line"

NODE_CLASS_MAPPINGS.update(_K2C_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(_K2C_NAMES)

# Onyx Endless Sampler: removed (long videos are now rendered in one pass).

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


# ── Onyx licence notice (console, once per start) ────────────────────────────
try:
    print("\n" + "=" * 64 +
          "\n  Onyx — proprietary software, licensed and not sold."
          "\n  Copyright (c) 2026 ONYX-AI. All rights reserved."
          "\n  Use is governed by LICENSE.txt. Contact: onyxskool@gmail.com Copying, resale, extraction of the"
          "\n  protected code or bypassing the licence verification is prohibited."
          "\n" + "=" * 64 + "\n")
except Exception:
    pass
