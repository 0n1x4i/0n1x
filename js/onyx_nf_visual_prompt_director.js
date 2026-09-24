import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { chainCallback } from "./onyx_nf_utility.js";

const NODE_ID = "OnyxVisualPromptDirector";

const MODES = [
  "Photography", "Photo Enhance", "Architecture", "Character", "Product",
  "Image Edit", "Style Transfer", "Dataset Caption", "Video", "Custom",
];
const TARGETS = ["Generic", "Krea 2", "FLUX.2 Klein", "Z-Image", "Qwen Image", "MiniMax", "LTX 2.5"];
const CREATIVITY = ["Strict", "Balanced", "Creative", "Dice"];
const DIRECTORS = [
  "General Director", "Prompt Enhancer", "Reverse Engineer", "Surgical Edit",
  "Face / Identity Analyst", "Subject Appearance Analyst", "Reference Composer",
  "Photography Director", "Smartphone Realism", "Arm's-Length Selfie", "Mirror Selfie",
  "First-Person POV", "Fashion Editorial", "Vintage / Analog", "Intimate Portrait",
  "Krea 2 High Detail", "Krea 2 Smartphone Realism", "Krea 2 Pose Lock",
  "Video Director", "MiniMax H3 Director", "Architecture Director", "Character Director",
  "Product Director", "Style Transfer Director", "Dataset Caption Director",
  "Maximum Detail Director", "Custom",
];
const MODEL_PRESETS = [
  "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
  "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
  "Qwen/Qwen3-VL-2B-Instruct",
  "Qwen/Qwen3-VL-4B-Instruct",
  "Qwen/Qwen3-VL-8B-Instruct",
  "Qwen/Qwen2.5-VL-3B-Instruct",
  "Qwen/Qwen2.5-VL-7B-Instruct",
  "google/gemma-3-4b-it",
  "google/gemma-3-12b-it",
];
const CUSTOM_MODEL = "__onyx_nf_custom__";
let cachedLocalModels = null;
let localModelsRequest = null;

async function loadDownloadedLocalModels(force = false) {
  if (!force && Array.isArray(cachedLocalModels)) return cachedLocalModels;
  if (localModelsRequest) return localModelsRequest;
  localModelsRequest = (async () => {
    const response = await api.fetchApi("/onyx/nf/local/models", {
      cache: "no-store",
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
    cachedLocalModels = data.models || [];
    return cachedLocalModels;
  })();
  try {
    return await localModelsRequest;
  } finally {
    localModelsRequest = null;
  }
}
const MAP_ROWS = [
  ["subject", "Sujeto / apariencia"],
  ["face_identity", "Rostro / identidad"],
  ["body_proportions", "Cuerpo / proporciones"],
  ["outfit", "Ropa"],
  ["pose", "Pose"],
  ["composition", "Composición"],
  ["camera", "Cámara"],
  ["scene_environment", "Escena / entorno"],
  ["lighting", "Iluminación"],
  ["colors", "Colores"],
  ["mood_style", "Atmósfera / estilo"],
  ["materials", "Materiales"],
];
const MAP_OPTIONS = ["Auto", "Image 1", "Image 2", "Image 3", "Blend"];
const MAP_PRESETS = ["Manual", "Identidad 1 / Escena 2", "Ropa en 3", "Pose lock (Krea)", "Solo estilo de 2"];
const PRESERVE_ROWS = [
  ["subject", "Sujeto"], ["face_identity", "Rostro"], ["body_proportions", "Cuerpo"],
  ["outfit", "Ropa"], ["pose", "Pose"], ["composition", "Composición"],
  ["camera", "Cámara"], ["scene_environment", "Escena"], ["lighting", "Iluminación"],
  ["colors", "Colores"], ["mood_style", "Estilo"], ["materials", "Materiales"],
];

const MAP_PRESET_VALUES = {
  "Identidad 1 / Escena 2": {
    subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
    outfit: "Image 1", pose: "Image 1", composition: "Image 2", camera: "Image 2",
    scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2",
    mood_style: "Image 2", materials: "Image 1",
  },
  "Ropa en 3": {
    subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
    outfit: "Image 3", pose: "Image 2", composition: "Image 2", camera: "Image 2",
    scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2",
    mood_style: "Image 2", materials: "Image 3",
  },
  "Pose lock (Krea)": {
    subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
    outfit: "Image 1", pose: "Image 2", composition: "Image 2", camera: "Image 2",
    scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2",
    mood_style: "Image 2", materials: "Image 1",
  },
  "Solo estilo de 2": {
    subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
    outfit: "Image 1", pose: "Image 1", composition: "Image 1", camera: "Image 1",
    scene_environment: "Image 1", lighting: "Image 2", colors: "Image 2",
    mood_style: "Image 2", materials: "Image 2",
  },
};


const CONTENT_MODES = ["SFW", "NSFW"];
const INTENTION_PRESET_NAMES = [
  "Manual",
  "Identidad + escena",
  "Identidad + escena + ropa",
  "Solo cambiar ropa",
  "Pose lock (Krea)",
  "Mirror / gym selfie",
  "Close-up cara",
  "Lingerie / implied",
  "Explicit",
];

const ANTI_IMG3_LEAK =
  "From IMAGE 3 describe ONLY garments, fabrics and accessories. " +
  "Never copy pose, hands, camera angle, framing or background from IMAGE 3. " +
  "Do not render readable brand text or logos on clothing; say color and garment type only.";

const SFW_RULE =
  "Keep the image non-sexual and fully clothed or casually clothed as implied by the outfit reference. " +
  "No nudity, no sexual acts, no explicit anatomy.";

const NSFW_SOFT_RULE =
  "Adult content allowed: lingerie, implied nudity, suggestive posing. " +
  "No illegal content. Prefer tasteful adult framing unless the user request is more explicit.";

const NSFW_EXPLICIT_RULE =
  "Adult explicit content allowed as requested by the intention preset and user request. " +
  "No illegal content, no minors. Describe anatomy only when needed for the shot.";

/** Intention presets: auto-fill request, map, preserve, director, rules. */
const INTENTION_PRESETS = {
  "Manual": null,
  "Identidad + escena": {
    map_preset: "Identidad 1 / Escena 2",
    mode: "Photography",
    target: "Krea 2",
    creativity: "Strict",
    director: "Reference Composer",
    prompt_length: "Maximum Detail",
    preserve: { subject: true, body_proportions: true, materials: false, composition: false, lighting: false, camera: false, colors: false },
    reference_map: {
      subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
      outfit: "Image 1", pose: "Image 2", composition: "Image 2", camera: "Image 2",
      scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2", mood_style: "Image 2", materials: "Image 1",
    },
    denoise_hint: "0.55–0.65",
    nsfw_only: false,
    request_sfw: "Using IMAGE 1 for identity (face, hair, skin, body proportions) and IMAGE 2 for the scene (pose, composition, camera, background, lighting, style), write one prompt for a non-explicit photograph.",
    request_nsfw: "Using IMAGE 1 for identity and IMAGE 2 for scene/pose, write one prompt. Adult content may follow IMAGE 2 wardrobe if present.",
    rules_sfw: SFW_RULE + " Prefer photorealistic coherence between identity and scene.",
    rules_nsfw: NSFW_SOFT_RULE,
  },
  "Identidad + escena + ropa": {
    map_preset: "Ropa en 3",
    mode: "Photography",
    target: "Krea 2",
    creativity: "Strict",
    director: "Reference Composer",
    prompt_length: "Maximum Detail",
    preserve: { subject: true, body_proportions: true, materials: false, composition: false, lighting: false, camera: false, colors: false },
    reference_map: {
      subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
      outfit: "Image 3", pose: "Image 2", composition: "Image 2", camera: "Image 2",
      scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2", mood_style: "Image 2", materials: "Image 3",
    },
    denoise_hint: "0.72–0.80 (ropa); baja a 0.65 si se pierde la pose",
    nsfw_only: false,
    request_sfw: "IMAGE 1 = identity only. IMAGE 2 = scene and pose to recreate. IMAGE 3 = clothing only. Write one prompt putting the IMAGE 1 person into the IMAGE 2 scene wearing the IMAGE 3 outfit.",
    request_nsfw: "IMAGE 1 = identity. IMAGE 2 = scene/pose. IMAGE 3 = outfit (may be revealing). Write one prompt combining them.",
    rules_sfw: SFW_RULE + " " + ANTI_IMG3_LEAK,
    rules_nsfw: NSFW_SOFT_RULE + " " + ANTI_IMG3_LEAK,
  },
  "Solo cambiar ropa": {
    map_preset: "Ropa en 3",
    mode: "Image Edit",
    target: "Krea 2",
    creativity: "Strict",
    director: "Surgical Edit",
    prompt_length: "Detailed",
    preserve: { subject: true, body_proportions: true, materials: false, composition: true, lighting: true, camera: true, colors: false },
    reference_map: {
      subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
      outfit: "Image 3", pose: "Image 2", composition: "Image 2", camera: "Image 2",
      scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2", mood_style: "Image 2", materials: "Image 3",
    },
    denoise_hint: "0.70–0.78",
    nsfw_only: false,
    request_sfw: "Keep identity from IMAGE 1 and the exact pose/scene from IMAGE 2. Replace only the clothing with the outfit from IMAGE 3. Do not change pose or background.",
    request_nsfw: "Keep identity (IMAGE 1) and pose/scene (IMAGE 2). Replace only clothing with IMAGE 3 outfit (adult ok).",
    rules_sfw: SFW_RULE + " " + ANTI_IMG3_LEAK + " Minimal other changes.",
    rules_nsfw: NSFW_SOFT_RULE + " " + ANTI_IMG3_LEAK + " Change garments only.",
  },
  "Pose lock (Krea)": {
    map_preset: "Pose lock (Krea)",
    mode: "Photography",
    target: "Krea 2",
    creativity: "Strict",
    director: "Krea 2 Pose Lock",
    prompt_length: "Maximum Detail",
    preserve: { subject: true, body_proportions: true, materials: false, composition: true, lighting: false, camera: true, colors: false },
    reference_map: {
      subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
      outfit: "Image 1", pose: "Image 2", composition: "Image 2", camera: "Image 2",
      scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2", mood_style: "Image 2", materials: "Image 1",
    },
    denoise_hint: "0.55–0.65",
    nsfw_only: false,
    request_sfw: "Lock pose, joint angles, crop and camera from IMAGE 2. Identity from IMAGE 1. Photoreal, non-explicit.",
    request_nsfw: "Lock pose/camera from IMAGE 2; identity from IMAGE 1. Adult wardrobe allowed if present.",
    rules_sfw: SFW_RULE + " Prioritize spatial pose accuracy for Krea 2.",
    rules_nsfw: NSFW_SOFT_RULE + " Prioritize spatial pose accuracy for Krea 2.",
  },
  "Mirror / gym selfie": {
    map_preset: "Identidad 1 / Escena 2",
    mode: "Photography",
    target: "Krea 2",
    creativity: "Strict",
    director: "Mirror Selfie",
    prompt_length: "Maximum Detail",
    preserve: { subject: true, body_proportions: true, materials: false, composition: false, lighting: false, camera: false, colors: false },
    reference_map: {
      subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
      outfit: "Image 2", pose: "Image 2", composition: "Image 2", camera: "Image 2",
      scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2", mood_style: "Image 2", materials: "Image 2",
    },
    denoise_hint: "0.60–0.70",
    nsfw_only: false,
    request_sfw: "Recreate IMAGE 2 as a believable mirror/gym selfie with identity from IMAGE 1: phone in hand, mirror geometry, casual pose, photoreal. If IMAGE 3 is connected, use it only for outfit.",
    request_nsfw: "Mirror/gym selfie: identity IMAGE 1, scene/pose IMAGE 2, optional outfit IMAGE 3. Adult ok.",
    rules_sfw: SFW_RULE + " Account for mirror reflection and phone selfie perspective. " + ANTI_IMG3_LEAK,
    rules_nsfw: NSFW_SOFT_RULE + " Mirror selfie realism. " + ANTI_IMG3_LEAK,
  },
  "Close-up cara": {
    map_preset: "Identidad 1 / Escena 2",
    mode: "Photography",
    target: "Krea 2",
    creativity: "Strict",
    director: "Face / Identity Analyst",
    prompt_length: "Detailed",
    preserve: { subject: true, body_proportions: true, materials: false, composition: false, lighting: false, camera: false, colors: false },
    reference_map: {
      subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
      outfit: "Image 1", pose: "Image 1", composition: "Image 2", camera: "Image 2",
      scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2", mood_style: "Image 2", materials: "Image 1",
    },
    denoise_hint: "0.50–0.60",
    nsfw_only: false,
    request_sfw: "Close-up portrait emphasizing face identity from IMAGE 1. Lighting/atmosphere may follow IMAGE 2. Non-explicit.",
    request_nsfw: "Close-up portrait, identity from IMAGE 1; mood from IMAGE 2. Adult expression ok, not explicit anatomy focus unless requested.",
    rules_sfw: SFW_RULE + " Stable facial anatomy; no celebrity naming.",
    rules_nsfw: NSFW_SOFT_RULE + " Stable facial anatomy; no celebrity naming.",
  },
  "Lingerie / implied": {
    map_preset: "Ropa en 3",
    mode: "Photography",
    target: "Krea 2",
    creativity: "Balanced",
    director: "Intimate Portrait",
    prompt_length: "Maximum Detail",
    preserve: { subject: true, body_proportions: true, materials: false, composition: false, lighting: false, camera: false, colors: false },
    reference_map: {
      subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
      outfit: "Image 3", pose: "Image 2", composition: "Image 2", camera: "Image 2",
      scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2", mood_style: "Image 2", materials: "Image 3",
    },
    denoise_hint: "0.70–0.80",
    nsfw_only: true,
    request_sfw: "",
    request_nsfw: "Identity IMAGE 1, scene/pose IMAGE 2, lingerie or implied-nude wardrobe from IMAGE 3. Tasteful adult photograph.",
    rules_sfw: SFW_RULE,
    rules_nsfw: NSFW_SOFT_RULE + " " + ANTI_IMG3_LEAK,
  },
  "Explicit": {
    map_preset: "Ropa en 3",
    mode: "Photography",
    target: "Krea 2",
    creativity: "Balanced",
    director: "Intimate Portrait",
    prompt_length: "Maximum Detail",
    preserve: { subject: true, body_proportions: true, materials: false, composition: false, lighting: false, camera: false, colors: false },
    reference_map: {
      subject: "Image 1", face_identity: "Image 1", body_proportions: "Image 1",
      outfit: "Image 3", pose: "Image 2", composition: "Image 2", camera: "Image 2",
      scene_environment: "Image 2", lighting: "Image 2", colors: "Image 2", mood_style: "Image 2", materials: "Image 3",
    },
    denoise_hint: "0.72–0.85",
    nsfw_only: true,
    request_sfw: "",
    request_nsfw: "Identity IMAGE 1, pose/scene IMAGE 2, wardrobe or nude state from IMAGE 3 as appropriate. Explicit adult content allowed.",
    rules_sfw: SFW_RULE,
    rules_nsfw: NSFW_EXPLICIT_RULE + " " + ANTI_IMG3_LEAK,
  },
};

function applyIntentionPreset(state, name, contentMode) {
  const preset = INTENTION_PRESETS[name];
  if (!preset) {
    state.intention_preset = "Manual";
    return state;
  }
  if (preset.nsfw_only && contentMode !== "NSFW") {
    return state; // caller should block
  }
  state.intention_preset = name;
  state.content_mode = contentMode;
  state.map_preset = preset.map_preset;
  state.mode = preset.mode;
  state.target = preset.target;
  state.creativity = preset.creativity;
  state.director = preset.director;
  state.prompt_length = preset.prompt_length;
  state.preserve = { ...FALLBACK_STATE.preserve, ...preset.preserve };
  state.reference_map = { ...FALLBACK_STATE.reference_map, ...preset.reference_map };
  state.denoise_hint = preset.denoise_hint || "";
  const nsfw = contentMode === "NSFW";
  state.request = nsfw ? (preset.request_nsfw || preset.request_sfw) : (preset.request_sfw || preset.request_nsfw);
  state.workflow_rules = nsfw ? (preset.rules_nsfw || preset.rules_sfw) : (preset.rules_sfw || preset.rules_nsfw);
  state.custom_director_behavior =
    "Return only the final positive prompt and respect the configured trigger prefix. " +
    "One coherent paragraph. No reference labels, no alternatives, no generic defect lists.";
  state.generated_prompt = "";
  state.lock_generated_prompt = false;
  return state;
}


async function blobToDataUrl(blob) {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/** Resolve only the node directly connected to the Director.
 * Walking through transforms would analyze the wrong, pre-transform image.
 */
function findUpstreamImageNode(graph, targetNode, inputName) {
  const input = (targetNode.inputs || []).find((i) => i.name === inputName);
  if (!input || input.link == null) return null;
  const link = graph.links?.[input.link];
  if (!link) return null;
  const originId = Array.isArray(link) ? link[1] : link.origin_id;
  return graph.getNodeById?.(originId) || graph._nodes_by_id?.[originId] || null;
}

async function fetchLoadImageAsDataUrl(imageNode) {
  if (!imageNode) return null;
  try {
    if (imageNode.imgs?.[0]) {
      const canvas = document.createElement("canvas");
      const img = imageNode.imgs[0];
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      if (canvas.width && canvas.height) {
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        return canvas.toDataURL("image/png");
      }
    }
  } catch (_) {}

  if (!/LoadImage|LoadImageOutput|Image From/i.test(String(imageNode.type || ""))) {
    return null;
  }

  const widgets = imageNode.widgets || [];
  const nameWidget = widgets.find((w) => w.name === "image") || widgets[0];
  const filename = (nameWidget?.value || "").toString();
  if (!filename) return null;

  const norm = filename.replace(/\\/g, "/");
  const parts = norm.split("/");
  const base = parts.pop();
  const subfolder = parts.join("/");
  const params = new URLSearchParams({ filename: base, type: "input" });
  if (subfolder) params.set("subfolder", subfolder);

  let response = await api.fetchApi(`/view?${params.toString()}`);
  if (!response.ok) {
    response = await api.fetchApi(`/view?filename=${encodeURIComponent(filename)}&type=input`);
  }
  if (!response.ok) throw new Error(`No pude leer ${filename} (${response.status})`);
  const blob = await response.blob();
  return await blobToDataUrl(blob);
}

async function collectDirectorImages(graph, directorNode) {
  const out = { image_1_b64: null, image_2_b64: null, image_3_b64: null };
  const slots = ["image_1", "image_2", "image_3"];
  const keys = ["image_1_b64", "image_2_b64", "image_3_b64"];
  for (let i = 0; i < slots.length; i++) {
    const upstream = findUpstreamImageNode(graph, directorNode, slots[i]);
    if (!upstream) continue;
    try {
      out[keys[i]] = await fetchLoadImageAsDataUrl(upstream);
    } catch (err) {
      console.warn("[Onyx Visual Prompt Director] image fetch", slots[i], err);
    }
  }
  return out;
}

const FALLBACK_STATE = {
  version: 4,
  request: "",
  mode: "Photography",
  target: "Krea 2",
  creativity: "Balanced",
  runtime: "Transformers local",
  model_id: "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
  endpoint: "http://127.0.0.1:11434",
  director: "Photography Director",
  preserve: {
    subject: true, face_identity: true, body_proportions: true, outfit: false, pose: false,
    materials: false, composition: false, scene_environment: false,
    lighting: false, camera: false, colors: false, mood_style: false,
  },
  reference_map: Object.fromEntries(MAP_ROWS.map(([key]) => [key, "Auto"])),
  map_preset: "Manual",
  intention_preset: "Identidad + escena + ropa",
  content_mode: "SFW",
  prompt_language: "English",
  trigger_phrase: "vixmodel",
  run_mode: "Solo prompt",
  denoise_hint: "0.72–0.80 (ropa); baja a 0.65 si se pierde la pose",
  seed: 0,
  prompt_length: "Detailed",
  workflow_rules: "",
  custom_director_behavior: "",
  lock_generated_prompt: false,
  generated_prompt: "",
  four_bit: false,
  attention: "auto",
  unload_after: true,
  max_image_edge: 1536,
  timeout: 600,
};

function injectStyle() {
  if (document.getElementById("onyx-nf-director-style")) return;
  const style = document.createElement("style");
  style.id = "onyx-nf-director-style";
  style.textContent = `
    .onx2d-wrap { height:100%; box-sizing:border-box; display:flex; flex-direction:column; gap:8px;
      overflow-y:auto; padding:8px; color:#e8e8ea; background:linear-gradient(160deg,rgba(8,42,51,.34),rgba(7,26,33,.22) 60%,rgba(4,16,22,.34)); border:1px solid #1ebbd8aa;
      border-radius:6px; font:11px ui-sans-serif,system-ui,sans-serif; scrollbar-width:thin; }
    .onx2d-section { display:flex; flex-direction:column; gap:6px; padding:8px; border:1px solid #0f3036;
      border-radius:6px; background:rgba(13,5,6,0.28); }
    .onx2d-title { color:#60e6ff; font-size:10px; font-weight:700; letter-spacing:.055em; text-transform:uppercase; }
    .onx2d-row { display:flex; align-items:center; gap:7px; }
    .onx2d-grid3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:7px; }
    .onx2d-field { display:flex; min-width:0; flex-direction:column; gap:3px; }
    .onx2d-label { color:#9d8c90; font-size:9px; font-weight:650; letter-spacing:.045em; text-transform:uppercase; }
    .onx2d-input,.onx2d-select,.onx2d-area { width:100%; box-sizing:border-box; border:1px solid #1f4147;
      border-radius:5px; color:#eef3f4; background:#090506; padding:5px 7px; font:11px ui-sans-serif,system-ui,sans-serif; }
    .onx2d-input:focus,.onx2d-select:focus,.onx2d-area:focus { outline:none; border-color:#17dbff; }
    .onx2d-area { resize:vertical; min-height:48px; font-family:ui-monospace,Consolas,monospace; line-height:1.35; }
    .onx2d-area.output { min-height:92px; color:#ffffff; background:#050203; }
    .onx2d-action { border:1px solid #17dbff; border-radius:5px; background:#115c6a; color:#ffffff;
      padding:5px 9px; cursor:pointer; font:700 10px ui-sans-serif,system-ui,sans-serif; white-space:nowrap; }
    .onx2d-action:hover { background:#17dbff; }
    .onx2d-action:disabled { cursor:default; opacity:.45; }
    .onx2d-banner { padding:7px 9px; text-align:center; border:1px solid #17dbff; border-radius:5px;
      background:linear-gradient(90deg,#0e4d59,#1394ac,#0e4d59); color:#ffffff; font-weight:750; letter-spacing:.045em; }
    .onx2d-status { min-height:14px; color:#a89599; font-size:10px; }
    .onx2d-status.err { color:#ff7b91; }
    .onx2d-checks { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:5px 8px; }
    .onx2d-check { display:flex; align-items:center; gap:5px; min-width:0; color:#cbd5d7; cursor:pointer; }
    .onx2d-check input { accent-color:#17dbff; }
    .onx2d-map { display:grid; grid-template-columns:minmax(110px,1.3fr) repeat(5,minmax(48px,1fr)); gap:4px; align-items:center; }
    .onx2d-preflight { padding:6px 8px; border-radius:5px; border:1px solid #1f373c; background:#090506; color:#c1cdcf; font-size:10px; }
    .onx2d-preflight.ok { border-color:#1f6a3a; color:#9ddeb5; }
    .onx2d-preflight.bad { border-color:#ff1744; color:#ff7b91; }
    .onx2d-preflight.warn { border-color:#a67c00; color:#e6c35c; }
    .onx2d-map-head { color:#8f7f83; text-align:center; font-size:8px; font-weight:700; text-transform:uppercase; }
    .onx2d-map-label { color:#c1cdcf; font-size:9px; }
    .onx2d-map-btn { min-height:24px; border:1px solid #1f373c; border-radius:4px; color:#a9999d;
      background:#090506; cursor:pointer; font:9px ui-sans-serif,system-ui,sans-serif; }
    .onx2d-map-btn:hover { border-color:#1892a8; color:#ffffff; }
    .onx2d-map-btn.active { border-color:#17dbff; background:#129fb9; color:#ffffff; font-weight:700; }
    .onx2d-details { border:1px solid #0f3036; border-radius:6px; background:rgba(13,5,6,0.28); }
    .onx2d-details > summary { cursor:pointer; color:#60e6ff; padding:8px; font-size:10px; font-weight:700;
      letter-spacing:.055em; text-transform:uppercase; user-select:none; }
    .onx2d-details-body { display:flex; flex-direction:column; gap:8px; padding:0 8px 8px; }
    .onx2d-spacer { flex:1; }
    .onx2d-guide { color:#c1cdcf; font-size:10px; line-height:1.4; padding:4px 0; }
    .onx2d-hint { color:#59d3e9; font-size:10px; min-height:14px; }
    .onx2d-toggle { display:flex; gap:6px; }
    .onx2d-toggle button { flex:1; min-height:28px; border:1px solid #1f373c; border-radius:5px;
      background:#090506; color:#a9999d; cursor:pointer; font:700 10px ui-sans-serif,system-ui,sans-serif; }
    .onx2d-toggle button.active { border-color:#17dbff; background:#129fb9; color:#ffffff; }
    .onx2d-toggle button.active.nsfw { border-color:#ff8a00; background:#8a3b00; }

  `;
  document.head.appendChild(style);
}

function parseState(value) {
  let incoming = {};
  try { incoming = JSON.parse(value || "{}"); } catch (_) {}
  const state = { ...FALLBACK_STATE, ...incoming };
  state.preserve = { ...FALLBACK_STATE.preserve, ...(incoming.preserve || {}) };
  state.reference_map = { ...FALLBACK_STATE.reference_map, ...(incoming.reference_map || {}) };
  if (!state.content_mode) state.content_mode = "SFW";
  if (!state.intention_preset) state.intention_preset = "Identidad + escena + ropa";
  if (!state.prompt_language) state.prompt_language = "English";
  if (state.trigger_phrase == null) state.trigger_phrase = "vixmodel";
  return state;
}

function optionSelect(values, current) {
  const select = document.createElement("select");
  select.className = "onx2d-select";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
  select.value = values.includes(current) ? current : values[0];
  return select;
}

function field(label, ...controls) {
  const root = document.createElement("label");
  root.className = "onx2d-field";
  const caption = document.createElement("span");
  caption.className = "onx2d-label";
  caption.textContent = label;
  root.append(caption, ...controls);
  return root;
}

function checkControl(label, value, onChange) {
  const root = document.createElement("label");
  root.className = "onx2d-check";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = !!value;
  input.addEventListener("change", () => onChange(input.checked));
  const text = document.createElement("span");
  text.textContent = label;
  root.append(input, text);
  return { root, input };
}

function isLocalEndpoint(value) {
  try {
    const raw = (value || "").trim();
    const parsed = new URL(raw.includes("://") ? raw : `http://${raw}`);
    return ["localhost", "127.0.0.1", "::1", "[::1]"].includes(parsed.hostname.toLowerCase());
  } catch (_) {
    return false;
  }
}

function formatModelSize(bytes) {
  const value = Number(bytes) || 0;
  if (!value) return "";
  if (value >= 1024 ** 3) return `${(value / (1024 ** 3)).toFixed(1)} GiB`;
  return `${Math.round(value / (1024 ** 2))} MiB`;
}

app.registerExtension({
  name: "Onyx.Krea2Carousel.VisualPromptDirector",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    injectStyle();

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const stateWidget = node.widgets?.find((widget) => widget.name === "director_state");
      if (!stateWidget) return;
      stateWidget.hidden = true;
      stateWidget.computeSize = () => [0, -4];

      let state = parseState(stateWidget.value);
      const controls = {};
      const wrap = document.createElement("div");
      wrap.className = "onx2d-wrap";
      for (const event of ["mousedown", "pointerdown", "wheel", "keydown"]) {
        wrap.addEventListener(event, (e) => e.stopPropagation());
      }

      function persist() {
        stateWidget.value = JSON.stringify(state);
        stateWidget.callback?.(stateWidget.value);
        node.graph?.setDirtyCanvas(true, true);
      }

      function bindText(control, key) {
        control.value = state[key] || "";
        control.addEventListener("input", () => { state[key] = control.value; persist(); });
        controls[key] = control;
      }

      function bindSelect(control, key) {
        control.value = state[key];
        control.addEventListener("change", () => { state[key] = control.value; persist(); });
        controls[key] = control;
      }

      function setStatus(message, error = false) {
        controls.status.textContent = message || "";
        controls.status.classList.toggle("err", !!error);
      }

      const requestSection = document.createElement("div");
      requestSection.className = "onx2d-section";
      const requestTitle = document.createElement("div");
      requestTitle.className = "onx2d-title";
      requestTitle.textContent = "¿Qué quieres crear o cambiar?";
      const request = document.createElement("textarea");
      request.className = "onx2d-area";
      request.placeholder = "Describe la imagen o el cambio. Las referencias se asignan abajo.";
      bindText(request, "request");
      requestSection.append(requestTitle, request);

      const directionSection = document.createElement("div");
      directionSection.className = "onx2d-section";
      const directionGrid = document.createElement("div");
      directionGrid.className = "onx2d-grid3";
      const mode = optionSelect(MODES, state.mode);
      const target = optionSelect(TARGETS, state.target);
      const creativity = optionSelect(CREATIVITY, state.creativity);
      bindSelect(mode, "mode"); bindSelect(target, "target"); bindSelect(creativity, "creativity");
      directionGrid.append(field("Modo", mode), field("Destino", target), field("Creatividad", creativity));
      const director = optionSelect(DIRECTORS, state.director);
      bindSelect(director, "director");
      directionSection.append(directionGrid, field("Director", director));

      const engineSection = document.createElement("div");
      engineSection.className = "onx2d-section";
      const engineTitle = document.createElement("div");
      engineTitle.className = "onx2d-title";
      engineTitle.textContent = "Modelo local que redacta el prompt";
      const runtime = optionSelect(["Transformers local", "Ollama", "llama.cpp"], state.runtime);
      controls.runtime = runtime;
      const modelSelect = document.createElement("select");
      modelSelect.className = "onx2d-select";
      modelSelect.title = "Selecciona un VLM descargado o una sugerencia de la lista.";
      const model = document.createElement("input");
      model.className = "onx2d-input";
      model.placeholder = "ID personalizado de Hugging Face";
      model.title = "Cualquier ID compatible; se descarga al usarlo si no está en la caché.";
      let downloadedModels = [];
      let ollamaModels = [];

      function addModelOption(id, detail) {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        if (detail) option.title = detail;
        return option;
      }

      function addModelGroup(label, entries) {
        if (!entries.length) return;
        const group = document.createElement("optgroup");
        group.label = label;
        for (const [id, detail] of entries) group.appendChild(addModelOption(id, detail));
        modelSelect.appendChild(group);
      }

      function renderModelOptions() {
        modelSelect.replaceChildren();
        if (runtime.value === "Transformers local") {
          const downloadedById = new Map(downloadedModels.map((item) => [item.id, item]));
          const describe = (item) =>
            [item.model_type, formatModelSize(item.size_bytes)].filter(Boolean).join(" · ");
          addModelGroup("Descargados", [...downloadedById].map(([id, item]) => [id, describe(item)]));
          addModelGroup(
            "Sugeridos · se descargan al usarlo",
            MODEL_PRESETS.filter((id) => !downloadedById.has(id)).map((id) => [id, ""]),
          );
        } else if (runtime.value === "Ollama") {
          addModelGroup("Ollama", ollamaModels.map((item) => [item.id, ""]));
        }
        const current = (state.model_id || "").trim();
        const custom = document.createElement("option");
        custom.value = CUSTOM_MODEL;
        custom.textContent = "Personalizado…";
        custom.title = "Habilita el campo de abajo para escribir cualquier ID.";
        modelSelect.appendChild(custom);

        const known = [...modelSelect.options].some(
          (option) => option.value === current && option.value !== CUSTOM_MODEL,
        );
        modelSelect.value = known ? current : CUSTOM_MODEL;
        model.style.display = known ? "none" : "";
      }
      renderModelOptions();
      model.value = state.model_id || "";
      modelSelect.addEventListener("change", () => {
        if (modelSelect.value === CUSTOM_MODEL) {
          model.value = state.model_id || "";
          model.style.display = "";
          model.focus();
          return;
        }
        state.model_id = modelSelect.value;
        model.value = state.model_id;
        model.style.display = "none";
        persist();
      });
      model.addEventListener("input", () => {
        state.model_id = model.value;
        persist();
      });
      const endpoint = document.createElement("input");
      endpoint.className = "onx2d-input";
      bindText(endpoint, "endpoint");

      const engineGrid = document.createElement("div");
      engineGrid.className = "onx2d-grid3";
      engineGrid.append(field("Runtime", runtime), field("Modelo VLM", modelSelect, model), field("Endpoint local", endpoint));
      const engineActions = document.createElement("div");
      engineActions.className = "onx2d-row";
      const fetchModels = document.createElement("button");
      fetchModels.type = "button"; fetchModels.className = "onx2d-action"; fetchModels.textContent = "BUSCAR OLLAMA";
      const unload = document.createElement("button");
      unload.type = "button"; unload.className = "onx2d-action"; unload.textContent = "LIBERAR MODELO";
      const actionSpacer = document.createElement("span"); actionSpacer.className = "onx2d-spacer";
      controls.status = document.createElement("span"); controls.status.className = "onx2d-status";
      engineActions.append(fetchModels, unload, actionSpacer, controls.status);
      engineSection.append(engineTitle, engineGrid, engineActions);

      function updateRuntime(resetKnownDefault = false, save = true) {
        const current = runtime.value;
        state.runtime = current;
        endpoint.disabled = current === "Transformers local";
        fetchModels.style.display = current === "llama.cpp" ? "none" : "";
        fetchModels.textContent = current === "Transformers local" ? "ACTUALIZAR LOCALES" : "BUSCAR OLLAMA";
        fetchModels.title = current === "Transformers local"
          ? "Vuelve a examinar la caché local de Hugging Face."
          : "Consulta los modelos instalados en el servidor Ollama local.";
        unload.style.display = current === "Transformers local" ? "" : "none";
        if (resetKnownDefault) {
          const configured = (state.endpoint || "").trim();
          const ollamaDefaults = ["http://127.0.0.1:11434", "http://localhost:11434"];
          const llamaDefaults = ["http://127.0.0.1:8080", "http://localhost:8080"];
          if (current === "Ollama" && (!configured || llamaDefaults.includes(configured))) {
            state.endpoint = "http://127.0.0.1:11434";
            endpoint.value = state.endpoint;
          } else if (current === "llama.cpp" && (!configured || ollamaDefaults.includes(configured))) {
            state.endpoint = "http://127.0.0.1:8080";
            endpoint.value = state.endpoint;
          }
        }
        renderModelOptions();
        if (save) persist();
      }
      runtime.addEventListener("change", () => updateRuntime(true, true));

      async function refreshLocalModels(quiet = false) {
        if (!quiet) {
          fetchModels.disabled = true;
          setStatus("Buscando modelos descargados…");
        }
        try {
          downloadedModels = await loadDownloadedLocalModels(!quiet);
          if (runtime.value === "Transformers local") renderModelOptions();
          if (!quiet) setStatus(`${downloadedModels.length} modelo(s) VLM descargado(s)`);
        } catch (error) {
          if (!quiet) setStatus(`Error: ${error.message}`, true);
        } finally {
          if (!quiet) fetchModels.disabled = false;
        }
      }

      fetchModels.addEventListener("click", async () => {
        if (runtime.value === "Transformers local") {
          await refreshLocalModels(false);
          return;
        }
        fetchModels.disabled = true;
        setStatus("Consultando modelos…");
        try {
          if (!isLocalEndpoint(state.endpoint || "http://127.0.0.1:11434")) {
            throw new Error("El endpoint de Ollama debe ser localhost");
          }
          const response = await api.fetchApi("/onyx/nf/ollama/models", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ host: state.endpoint || "http://127.0.0.1:11434" }),
          });
          const data = await response.json();
          if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
          ollamaModels = data.models || [];
          if (ollamaModels.length && !ollamaModels.some((item) => item.id === model.value)) {
            state.model_id = ollamaModels[0].id;
            model.value = state.model_id;
            persist();
          }
          renderModelOptions();
          setStatus(`${ollamaModels.length} modelo(s) encontrado(s)`);
        } catch (error) {
          setStatus(`Error: ${error.message}`, true);
        } finally {
          fetchModels.disabled = false;
        }
      });

      unload.addEventListener("click", async () => {
        unload.disabled = true;
        setStatus("Liberando memoria…");
        try {
          const response = await api.fetchApi(
            "/onyx/nf/local/unload", { method: "POST" }
          );
          const data = await response.json();
          if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
          setStatus("Modelo local liberado");
        } catch (error) {
          setStatus(`Error: ${error.message}`, true);
        } finally {
          unload.disabled = false;
        }
      });

      const toolsSection = document.createElement("div");
      toolsSection.className = "onx2d-section";
      const toolsTitle = document.createElement("div");
      toolsTitle.className = "onx2d-title";
      toolsTitle.textContent = "Presets de intención (sin escribir prompt)";
      const guide = document.createElement("div");
      guide.className = "onx2d-guide";
      guide.textContent = "1) Elige SFW/NSFW → 2) Elige preset → 3) Conecta fotos → 4) Generar prompt ya (o Queue).";
      const contentToggle = document.createElement("div");
      contentToggle.className = "onx2d-toggle";
      const btnSfw = document.createElement("button"); btnSfw.type = "button"; btnSfw.textContent = "SFW";
      const btnNsfw = document.createElement("button"); btnNsfw.type = "button"; btnNsfw.textContent = "NSFW";
      contentToggle.append(btnSfw, btnNsfw);
      function syncContentToggle() {
        const mode = state.content_mode === "NSFW" ? "NSFW" : "SFW";
        state.content_mode = mode;
        btnSfw.className = mode === "SFW" ? "active" : "";
        btnNsfw.className = mode === "NSFW" ? "active nsfw" : "";
      }
      function refreshIntentionOptions() {
        const mode = state.content_mode === "NSFW" ? "NSFW" : "SFW";
        const allowed = INTENTION_PRESET_NAMES.filter((name) => {
          const p = INTENTION_PRESETS[name];
          if (!p) return true;
          if (p.nsfw_only && mode !== "NSFW") return false;
          return true;
        });
        intention.replaceChildren();
        for (const name of allowed) {
          const opt = document.createElement("option");
          opt.value = name; opt.textContent = name;
          intention.appendChild(opt);
        }
        if (!allowed.includes(state.intention_preset)) {
          state.intention_preset = allowed.includes("Identidad + escena + ropa")
            ? "Identidad + escena + ropa" : allowed[0];
        }
        intention.value = state.intention_preset;
      }
      let generated = null;
      function runIntentionApply(showStatus = true) {
        const name = intention.value || "Manual";
        const mode = state.content_mode === "NSFW" ? "NSFW" : "SFW";
        const preset = INTENTION_PRESETS[name];
        if (preset?.nsfw_only && mode !== "NSFW") {
          setStatus("Ese preset requiere modo NSFW", true);
          return;
        }
        if (name === "Manual") {
          state.intention_preset = "Manual";
          persist();
          if (showStatus) setStatus("Preset manual — edita a mano");
          updatePreflight();
          return;
        }
        applyIntentionPreset(state, name, mode);
        // Reflect into visible controls
        request.value = state.request || "";
        if (controls.mode) controls.mode.value = state.mode;
        if (controls.target) controls.target.value = state.target;
        if (controls.creativity) controls.creativity.value = state.creativity;
        if (controls.director) controls.director.value = state.director;
        if (presetSelect) presetSelect.value = state.map_preset || "Manual";
        if (controls.prompt_length) controls.prompt_length.value = state.prompt_length;
        if (controls.workflow_rules) controls.workflow_rules.value = state.workflow_rules || "";
        if (controls.custom_director_behavior) controls.custom_director_behavior.value = state.custom_director_behavior || "";
        if (generated) generated.value = state.generated_prompt || "";
        if (controls.preserve) {
          for (const [key] of PRESERVE_ROWS) {
            if (controls.preserve[key]) controls.preserve[key].checked = !!state.preserve[key];
          }
        }
        renderMap?.();
        denoiseHint.textContent = state.denoise_hint
          ? `Denoise sugerido: ${state.denoise_hint}` : "";
        persist();
        if (showStatus) setStatus(`Preset aplicado: ${name} (${mode})`);
        updatePreflight();
      }
      btnSfw.addEventListener("click", () => {
        state.content_mode = "SFW";
        syncContentToggle();
        refreshIntentionOptions();
        if (INTENTION_PRESETS[state.intention_preset]?.nsfw_only) {
          state.intention_preset = "Identidad + escena + ropa";
          intention.value = state.intention_preset;
        }
        runIntentionApply(true);
      });
      btnNsfw.addEventListener("click", () => {
        state.content_mode = "NSFW";
        syncContentToggle();
        refreshIntentionOptions();
        runIntentionApply(true);
      });
      const intention = document.createElement("select");
      intention.className = "onx2d-select";
      controls.intention_preset = intention;
      intention.addEventListener("change", () => runIntentionApply(true));
      const presetSelect = optionSelect(MAP_PRESETS, state.map_preset || "Manual");
      bindSelect(presetSelect, "map_preset");
      controls.map_preset = presetSelect;
      presetSelect.addEventListener("change", () => {
        state.intention_preset = "Manual";
        intention.value = "Manual";
        const presetMap = MAP_PRESET_VALUES[state.map_preset];
        if (presetMap) {
          state.reference_map = { ...state.reference_map, ...presetMap };
          renderMap?.();
        }
        persist();
        setStatus(state.map_preset === "Manual" ? "Mapa manual" : `Mapa: ${state.map_preset}`);
        updatePreflight();
      });
      const seed = document.createElement("input");
      seed.className = "onx2d-input";
      seed.type = "number"; seed.min = "0"; seed.step = "1";
      seed.value = state.seed ?? 0;
      seed.addEventListener("change", () => {
        state.seed = Math.max(0, Number(seed.value) || 0);
        seed.value = state.seed; persist(); updatePreflight();
      });
      controls.seed = seed;
      const denoiseHint = document.createElement("div");
      denoiseHint.className = "onx2d-hint";
      controls.denoise_hint = denoiseHint;
      denoiseHint.textContent = state.denoise_hint ? `Denoise sugerido: ${state.denoise_hint}` : "";
      const toolsGrid = document.createElement("div");
      toolsGrid.className = "onx2d-grid3";
      toolsGrid.append(
        field("Preset de intención", intention),
        field("Preset de mapa", presetSelect),
        field("Semilla", seed),
      );
      const runModeToggle = document.createElement("div");
      runModeToggle.className = "onx2d-toggle";
      const btnPromptOnly = document.createElement("button");
      btnPromptOnly.type = "button"; btnPromptOnly.textContent = "Solo prompt";
      const btnFullQueue = document.createElement("button");
      btnFullQueue.type = "button"; btnFullQueue.textContent = "Cola completa";
      runModeToggle.append(btnPromptOnly, btnFullQueue);
      function syncRunMode() {
        const mode = state.run_mode === "Cola completa" ? "Cola completa" : "Solo prompt";
        state.run_mode = mode;
        btnPromptOnly.className = mode === "Solo prompt" ? "active" : "";
        btnFullQueue.className = mode === "Cola completa" ? "active" : "";
        generateNow.textContent = mode === "Solo prompt" ? "GENERAR PROMPT" : "GENERAR PROMPT + IMAGEN";
        generateNow.title = mode === "Solo prompt"
          ? "Analiza las 3 fotos con el VLM, escribe el prompt y lo deja editable. No lanza la cola."
          : "Analiza las 3 fotos, fija el prompt (Bloquear) y lanza Queue para generar la imagen.";
      }
      btnPromptOnly.addEventListener("click", () => { state.run_mode = "Solo prompt"; syncRunMode(); persist(); });
      btnFullQueue.addEventListener("click", () => { state.run_mode = "Cola completa"; syncRunMode(); persist(); });

      const genRow = document.createElement("div"); genRow.className = "onx2d-row";
      const generateNow = document.createElement("button");
      generateNow.type = "button"; generateNow.className = "onx2d-action";
      const preflight = document.createElement("div");
      preflight.className = "onx2d-preflight";
      controls.preflight = preflight;
      genRow.append(generateNow);
      syncContentToggle();
      syncRunMode();
      refreshIntentionOptions();
      if (!(state.request || "").trim() && state.intention_preset && state.intention_preset !== "Manual") {
        runIntentionApply(false);
      }
      toolsSection.append(
        toolsTitle, guide,
        field("Contenido", contentToggle),
        field("Después de generar", runModeToggle),
        toolsGrid, denoiseHint, genRow, preflight,
      );

      function connectedFlags() {
        const inp = (slot) => !!(node.inputs || []).find((i) => i.name === slot && i.link != null);
        return { image_1: inp("image_1"), image_2: inp("image_2"), image_3: inp("image_3") };
      }

      function updatePreflight() {
        const flags = connectedFlags();
        const problems = [];
        if (!(state.request || "").trim() && !flags.image_1 && !flags.image_2 && !flags.image_3) {
          problems.push("Escribe un pedido o conecta una imagen");
        }
        if ((state.runtime === "Ollama" || state.runtime === "llama.cpp") && state.runtime === "Ollama" && !(state.model_id || "").trim()) {
          problems.push("Falta modelo Ollama");
        }
        const outfit = (state.reference_map || {}).outfit || "Auto";
        for (const [key, route] of Object.entries(state.reference_map || {})) {
          if (route === "Image 1" && !flags.image_1) problems.push(`${key} pide Img 1 sin conectar`);
          if (route === "Image 2" && !flags.image_2) problems.push(`${key} pide Img 2 sin conectar`);
          if (route === "Image 3" && !flags.image_3) problems.push(`${key} pide Img 3 sin conectar`);
          if (route === "Blend" && (!flags.image_1 || !flags.image_2)) problems.push(`${key}: Mezclar necesita Img 1 y 2`);
        }
        if (state.lock_generated_prompt && !(state.generated_prompt || "").trim()) {
          problems.push("Bloquear activo sin prompt");
        }
        let tip = "Listo para generar";
        let cls = "ok";
        if (problems.length) { tip = problems[0]; cls = "bad"; }
        else if (flags.image_3 && outfit === "Auto") { tip = "IMAGEN 3 conectada → Auto usará ropa de la 3"; cls = "warn"; }
        else if (state.target === "Krea 2") { tip = "Krea 2: sin prompt negativo"; cls = "ok"; }
        preflight.textContent = tip;
        preflight.className = "onx2d-preflight " + cls;
      }

      generateNow.addEventListener("click", async () => {
        updatePreflight();
        if (preflight.classList.contains("bad")) {
          setStatus(preflight.textContent, true);
          return;
        }
        generateNow.disabled = true;
        const fullQueue = state.run_mode === "Cola completa";
        setStatus(fullQueue
          ? "Leyendo fotos y generando prompt… (luego Queue)"
          : "Leyendo fotos y generando prompt…");
        try {
          const images = await collectDirectorImages(app.graph, node);
          const flags = connectedFlags();
          const unreadable = ["image_1", "image_2", "image_3"].filter(
            (name) => flags[name] && !images[`${name}_b64`],
          );
          if (unreadable.length) {
            if (fullQueue) {
              state.lock_generated_prompt = false;
              if (controls.lock) controls.lock.checked = false;
              persist();
              setStatus(`Las ${unreadable.join(", ")} son dinámicas; generando prompt dentro de la cola…`);
              await app.queuePrompt(0);
              setStatus("Cola lanzada. El Director analizará los tensores reales durante la ejecución.");
              return;
            }
            throw new Error(
              `No pude previsualizar ${unreadable.join(", ")}. Usa Cola completa para analizar tensores dinámicos.`,
            );
          }
          const attached = [images.image_1_b64, images.image_2_b64, images.image_3_b64].filter(Boolean).length;
          if (attached === 0) {
            setStatus("Sin fotos leídas: revisa Load Image 1/2/3. Sigo solo con el texto…", true);
          } else {
            setStatus(`Fotos OK (${attached}/3). Llamando al VLM…`);
          }
          const response = await api.fetchApi("/onyx/nf/director/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              director_state: JSON.stringify(state),
              image_1_b64: images.image_1_b64,
              image_2_b64: images.image_2_b64,
              image_3_b64: images.image_3_b64,
            }),
          });
          const data = await response.json();
          if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
          if (data.prompt) {
            state.generated_prompt = data.prompt;
            generated.value = data.prompt;
            if (fullQueue) {
              state.lock_generated_prompt = true;
              if (controls.lock) controls.lock.checked = true;
            } else {
              state.lock_generated_prompt = false;
              if (controls.lock) controls.lock.checked = false;
            }
            persist();
          }
          if (fullQueue) {
            setStatus((data.status || "Prompt listo") + " · lanzando cola…");
            try {
              await app.queuePrompt(0);
              setStatus("Cola lanzada. El prompt está bloqueado; edítalo o desbloquea si quieres regenerar.");
            } catch (qerr) {
              setStatus(`Prompt listo, pero Queue falló: ${qerr.message || qerr}`, true);
            }
          } else {
            setStatus((data.status || "Prompt generado") + " · editable. Revisa y dale Queue cuando quieras.");
          }
        } catch (error) {
          setStatus(`Error: ${error.message}`, true);
        } finally {
          generateNow.disabled = false;
          updatePreflight();
        }
      });

      const banner = document.createElement("div");
      banner.className = "onx2d-banner";
      banner.textContent = "Solo prompt = editas tú · Cola completa = prompt + imagen";

      const outputSection = document.createElement("div");
      outputSection.className = "onx2d-section";
      const outputRow = document.createElement("div"); outputRow.className = "onx2d-row";
      const outputTitle = document.createElement("div"); outputTitle.className = "onx2d-title";
      outputTitle.textContent = "Prompt generado";
      const outputSpacer = document.createElement("span"); outputSpacer.className = "onx2d-spacer";
      const lock = checkControl("Bloquear", state.lock_generated_prompt, (value) => {
        state.lock_generated_prompt = value; persist();
      });
      controls.lock = lock.input;
      outputRow.append(outputTitle, outputSpacer, lock.root);
      generated = document.createElement("textarea");
      generated.className = "onx2d-area output";
      generated.placeholder = "Aquí aparecerá el prompt después de ejecutar…";
      bindText(generated, "generated_prompt");
      controls.generated = generated;
      outputSection.append(outputRow, generated);

      const advanced = document.createElement("details");
      advanced.className = "onx2d-details";
      advanced.open = true;
      const summary = document.createElement("summary"); summary.textContent = "Control avanzado de referencias";
      const advancedBody = document.createElement("div"); advancedBody.className = "onx2d-details-body";

      const preserveSection = document.createElement("div"); preserveSection.className = "onx2d-section";
      const preserveTitle = document.createElement("div"); preserveTitle.className = "onx2d-title";
      preserveTitle.textContent = "Preservar";
      const preserveGrid = document.createElement("div"); preserveGrid.className = "onx2d-checks";
      controls.preserve = {};
      for (const [key, label] of PRESERVE_ROWS) {
        const item = checkControl(label, state.preserve[key], (value) => {
          state.preserve[key] = value; persist();
        });
        controls.preserve[key] = item.input;
        preserveGrid.appendChild(item.root);
      }
      preserveSection.append(preserveTitle, preserveGrid);

      const mapSection = document.createElement("div"); mapSection.className = "onx2d-section";
      const mapTitle = document.createElement("div"); mapTitle.className = "onx2d-title";
      mapTitle.textContent = "Mapa de referencias";
      const map = document.createElement("div"); map.className = "onx2d-map";
      map.appendChild(document.createElement("span"));
      for (const label of ["Auto", "Img 1", "Img 2", "Ropa", "Mezclar"]) {
        const head = document.createElement("span"); head.className = "onx2d-map-head"; head.textContent = label;
        map.appendChild(head);
      }
      controls.mapButtons = {};
      for (const [key, label] of MAP_ROWS) {
        const caption = document.createElement("span"); caption.className = "onx2d-map-label"; caption.textContent = label;
        map.appendChild(caption);
        controls.mapButtons[key] = [];
        MAP_OPTIONS.forEach((value, index) => {
          const button = document.createElement("button");
          button.type = "button"; button.className = "onx2d-map-btn";
          button.textContent = ["A", "1", "2", "3", "1+2"][index];
          button.title = value;
          button.addEventListener("click", () => {
            state.map_preset = "Manual";
            state.intention_preset = "Manual";
            presetSelect.value = "Manual";
            intention.value = "Manual";
            state.reference_map[key] = value;
            renderMap(); persist();
          });
          controls.mapButtons[key].push(button);
          map.appendChild(button);
        });
      }
      function renderMap() {
        if (!controls.mapButtons) return;
        for (const [key] of MAP_ROWS) {
          controls.mapButtons[key].forEach((button, index) => {
            button.classList.toggle("active", state.reference_map[key] === MAP_OPTIONS[index]);
          });
        }
      }
      renderMap();
      mapSection.append(mapTitle, map);

      const rulesSection = document.createElement("div"); rulesSection.className = "onx2d-section";
      const promptLength = optionSelect(["Short", "Medium", "Detailed", "Maximum Detail"], state.prompt_length);
      bindSelect(promptLength, "prompt_length");
      const promptLanguage = optionSelect(["English", "Español", "Auto"], state.prompt_language);
      bindSelect(promptLanguage, "prompt_language");
      const triggerPhrase = document.createElement("input"); triggerPhrase.className = "onx2d-input";
      triggerPhrase.placeholder = "Vacío = sin trigger";
      bindText(triggerPhrase, "trigger_phrase");
      const workflowRules = document.createElement("textarea"); workflowRules.className = "onx2d-area";
      workflowRules.placeholder = "Reglas adicionales solo para este workflow (opcional)";
      bindText(workflowRules, "workflow_rules");
      const customBehavior = document.createElement("textarea"); customBehavior.className = "onx2d-area";
      customBehavior.placeholder = "Comportamiento personalizado del director (opcional)";
      bindText(customBehavior, "custom_director_behavior");
      rulesSection.append(
        field("Longitud del prompt", promptLength),
        field("Idioma de salida", promptLanguage),
        field("Trigger / prefijo", triggerPhrase),
        field("Reglas del workflow", workflowRules),
        field("Comportamiento del director", customBehavior),
      );

      const backendSection = document.createElement("div"); backendSection.className = "onx2d-section";
      const backendTitle = document.createElement("div"); backendTitle.className = "onx2d-title";
      backendTitle.textContent = "Memoria y entrada visual";
      const backendChecks = document.createElement("div"); backendChecks.className = "onx2d-checks";
      const fourBit = checkControl("Cargar en 4-bit", state.four_bit, (value) => {
        state.four_bit = value; persist();
      });
      const unloadAfter = checkControl("Liberar al terminar", state.unload_after, (value) => {
        state.unload_after = value; persist();
      });
      controls.fourBit = fourBit.input; controls.unloadAfter = unloadAfter.input;
      backendChecks.append(fourBit.root, unloadAfter.root);
      const attention = optionSelect(["auto", "sdpa", "eager", "flash_attention_2"], state.attention);
      bindSelect(attention, "attention");
      const maxEdge = document.createElement("input"); maxEdge.className = "onx2d-input";
      maxEdge.type = "number"; maxEdge.min = "512"; maxEdge.max = "3072"; maxEdge.step = "64";
      maxEdge.value = state.max_image_edge;
      maxEdge.addEventListener("change", () => {
        state.max_image_edge = Math.max(512, Math.min(3072, Number(maxEdge.value) || 1536));
        maxEdge.value = state.max_image_edge; persist();
      });
      controls.maxEdge = maxEdge;
      const timeout = document.createElement("input"); timeout.className = "onx2d-input";
      timeout.type = "number"; timeout.min = "30"; timeout.max = "3600"; timeout.step = "30";
      timeout.value = state.timeout;
      timeout.addEventListener("change", () => {
        state.timeout = Math.max(30, Math.min(3600, Number(timeout.value) || 600));
        timeout.value = state.timeout; persist();
      });
      controls.timeout = timeout;
      const backendGrid = document.createElement("div"); backendGrid.className = "onx2d-grid3";
      backendGrid.append(field("Atención", attention), field("Arista máx. imagen", maxEdge), field("Timeout (s)", timeout));
      backendSection.append(backendTitle, backendChecks, backendGrid);

      advancedBody.append(preserveSection, mapSection, rulesSection, backendSection);
      advanced.append(summary, advancedBody);
      wrap.append(requestSection, directionSection, engineSection, toolsSection, banner, outputSection, advanced);

      node.addDOMWidget("onyx_nf_visual_prompt_director", "div", wrap, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 900,
      });

      function applyState() {
        request.value = state.request || "";
        mode.value = state.mode; target.value = state.target; creativity.value = state.creativity;
        runtime.value = state.runtime; model.value = state.model_id || ""; endpoint.value = state.endpoint || "";
        director.value = state.director; generated.value = state.generated_prompt || "";
        lock.input.checked = !!state.lock_generated_prompt;
        if (controls.seed) controls.seed.value = state.seed ?? 0;
        if (controls.map_preset) controls.map_preset.value = state.map_preset || "Manual";
        else if (typeof presetSelect !== "undefined" && presetSelect) presetSelect.value = state.map_preset || "Manual";
        if (controls.intention_preset) {
          syncContentToggle?.();
          refreshIntentionOptions?.();
          controls.intention_preset.value = state.intention_preset || "Manual";
        }
        if (controls.denoise_hint) {
          controls.denoise_hint.textContent = state.denoise_hint
            ? `Denoise sugerido: ${state.denoise_hint}` : "";
        }
        promptLength.value = state.prompt_length;
        promptLanguage.value = state.prompt_language || "English";
        triggerPhrase.value = state.trigger_phrase ?? "vixmodel";
        workflowRules.value = state.workflow_rules || "";
        customBehavior.value = state.custom_director_behavior || "";
        attention.value = state.attention;
        maxEdge.value = state.max_image_edge; timeout.value = state.timeout;
        fourBit.input.checked = !!state.four_bit; unloadAfter.input.checked = !!state.unload_after;
        for (const [key] of PRESERVE_ROWS) controls.preserve[key].checked = !!state.preserve[key];
        renderMap(); updateRuntime(false, false);
      }

      chainCallback(node, "onConfigure", function () {
        state = parseState(stateWidget.value);
        applyState();
      });

      chainCallback(node, "onExecuted", function (message) {
        const prompt = message?.generated_prompt?.[0];
        if (typeof prompt === "string" && prompt.trim()) {
          state.generated_prompt = prompt;
          generated.value = prompt;
          persist();
        }
        const status = message?.status?.[0];
        if (status) setStatus(status);
      });

      updateRuntime(false, false);
      refreshLocalModels(true);
      updatePreflight();
      // Keep preflight fresh when links change.
      const _onConnectionsChange = node.onConnectionsChange;
      node.onConnectionsChange = function () {
        const r = _onConnectionsChange?.apply(this, arguments);
        try { updatePreflight(); } catch (_) {}
        return r;
      };
      setTimeout(() => {
        node.setSize([Math.max(node.size[0], 650), Math.max(node.size[1], 1010)]);
        node.graph?.setDirtyCanvas(true, true);
      }, 0);
    });
  },
});
