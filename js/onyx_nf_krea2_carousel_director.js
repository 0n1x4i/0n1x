import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { chainCallback } from "./onyx_nf_utility.js";

const NODE_ID = "OnyxKrea2CarouselDirector";
const PRESETS = {
  "Vibe Influencer": "Confident lifestyle influencer photo carousel: natural candid energy, varied hand gestures, one strong selfie, one over-the-shoulder turn, and believable social-media posing.",
  "Casual Phone": "Casual phone-camera carousel with spontaneous poses, relaxed posture, imperfect candid timing, natural hand placement, and one close selfie.",
  "Fashion Editorial": "Fashion editorial carousel with clean silhouettes, deliberate contrapposto, elegant hand placement, profile and over-the-shoulder variety, and controlled confident expressions.",
  "Luxury Lifestyle": "Luxury lifestyle influencer carousel: poised but natural gestures, composed posture, subtle confidence, premium campaign energy, and varied medium and close framing.",
  "Playful Social": "Playful social-media carousel with friendly expressions, lively asymmetrical gestures, candid mid-motion posing, a high-angle selfie, and one teasing look-back pose.",
  "Fitness Confidence": "Athletic confident carousel with grounded stance, strong posture, natural body tension, hands-on-hips and arms-crossed variations, plus a dynamic three-quarter turn.",
};

const MODEL_DEFAULTS = {
  "Transformers local": "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
  Ollama: "",
  "llama.cpp": "",
  "Gemini API": "gemini-3.5-flash",
  "Grok API": "",
};

const DEFAULT_ENGINE = {
  backend: "Transformers local",
  model_id: MODEL_DEFAULTS["Transformers local"],
  model_by_backend: { ...MODEL_DEFAULTS },
  endpoint: "http://127.0.0.1:11434",
  vibe_preset: "Vibe Influencer",
  vibe: PRESETS["Vibe Influencer"],
  include_selfie: true,
  seed: 0,
  timeout: 300,
  four_bit: false,
  unload_after: true,
};

function injectStyle() {
  if (document.getElementById("onx2-krea2-carousel-style")) return;
  const style = document.createElement("style");
  style.id = "onx2-krea2-carousel-style";
  style.textContent = `
    .onx2k-wrap{box-sizing:border-box;width:100%;min-height:970px;padding:12px;color:#e8f3f5;background:linear-gradient(160deg,rgba(8,42,51,.34),rgba(7,26,33,.22) 60%,rgba(4,16,22,.34));font:12px/1.38 Inter,Segoe UI,sans-serif;border:1px solid #1ebbd8aa;border-radius:11px;box-shadow:inset 0 1px #ffffff10}
    .onx2k-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;padding:4px 2px 11px;border-bottom:1px solid #ffffff13}.onx2k-brand{font-size:17px;font-weight:850;letter-spacing:.02em;color:#ffffff}.onx2k-sub{margin-top:2px;color:#9abac0;font-size:10px}.onx2k-badges{display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end}.onx2k-badge{padding:4px 7px;border-radius:999px;font-size:9px;font-weight:800;letter-spacing:.04em;background:#145460;color:#b8f4ff;border:1px solid #2e8ea0}.onx2k-badge.blue{background:#114048;color:#9ff0ff;border-color:#267786}
    .onx2k-section{margin-top:9px;padding:10px;border:1px solid #22444a;border-radius:8px;background:rgba(13,18,19,0.28)}.onx2k-section.blue{border-color:#225058;background:rgba(12,27,29,0.28)}.onx2k-title{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;color:#bff5ff;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}.onx2k-title small{color:#8da8ad;font-size:9px;font-weight:600;letter-spacing:0;text-transform:none}
    .onx2k-grid2{display:grid;grid-template-columns:1fr 1fr;gap:7px}.onx2k-grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.onx2k-row{display:flex;gap:7px;align-items:center}.onx2k-row>*{flex:1}.onx2k-label{display:block;margin-bottom:3px;color:#90a6aa;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.055em}
    .onx2k-input,.onx2k-select,.onx2k-area{box-sizing:border-box;width:100%;border:1px solid #335056;border-radius:6px;background:#080d0e;color:#ecf5f7;padding:7px;font:12px/1.38 inherit;outline:none}.onx2k-input:focus,.onx2k-select:focus,.onx2k-area:focus{border-color:#3ec9e3;box-shadow:0 0 0 2px #3ec9e320}.onx2k-area{min-height:78px;resize:vertical}.onx2k-poses{min-height:150px;font:11px/1.42 ui-monospace,Consolas,monospace}
    .onx2k-btn{border:1px solid #2b92a5;border-radius:6px;background:linear-gradient(#1c6d7c,#104b56);color:#ebfcff;padding:8px 9px;cursor:pointer;font-weight:750}.onx2k-btn:hover{filter:brightness(1.16)}.onx2k-btn:disabled{opacity:.45;cursor:wait}.onx2k-btn.secondary{border-color:#2f6772;background:linear-gradient(#1c5059,#113940);color:#dffaff}.onx2k-btn.ghost{background:#17191c;border-color:#3e464d;color:#cbdadc}
    .onx2k-presets{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.onx2k-chip{border:1px solid #2e454a;border-radius:999px;background:#101617;color:#a9c6cc;padding:5px 4px;cursor:pointer;font-size:9px}.onx2k-chip.active{border-color:#3bc3dd;background:#124c57;color:#cff7ff}
    .onx2k-status{margin-top:9px;padding:8px;border:1px solid #25513c;border-radius:6px;background:#0d241a;color:#9be4b9;white-space:pre-wrap}.onx2k-status.warn{border-color:#74551c;background:#2a210e;color:#ffd77d}.onx2k-status.err{border-color:#7b3037;background:#2c1115;color:#ffadb5}
    .onx2k-check{display:flex;align-items:center;gap:6px;color:#b9c7c9}.onx2k-check input{width:auto;accent-color:#31c2dd}.onx2k-key{font-family:ui-monospace,Consolas,monospace}.onx2k-hidden{display:none!important}.onx2k-preview{max-height:135px;overflow:auto;padding:7px;border:1px solid #23444a;border-radius:6px;background:rgba(7,15,16,0.28);color:#bcdae0;white-space:pre-wrap;font-size:10px}details.onx2k-section>summary{cursor:pointer;color:#b3d1d7;font-weight:750}
    .onx2k-source{padding:6px;border-radius:6px;background:#101d1f;color:#8fc7d2;border:1px solid #1f4e57}.onx2k-source.ok{background:#10251a;color:#9de4ba;border-color:#27583d}
  `;
  document.head.appendChild(style);
}

function make(tag, cls = "") { const el = document.createElement(tag); if (cls) el.className = cls; return el; }
function input(type = "text", cls = "onx2k-input") { const el = make("input", cls); el.type = type; return el; }
function select(options = []) { const el = make("select", "onx2k-select"); for (const value of options) { const o = make("option"); o.value = value; o.textContent = value; el.append(o); } return el; }
function button(label, cls = "") { const el = make("button", `onx2k-btn ${cls}`.trim()); el.type = "button"; el.textContent = label; return el; }
function field(label, el) { const root = make("label"); const cap = make("span", "onx2k-label"); cap.textContent = label; root.append(cap, el); return root; }
function checkbox(label, checked) { const root = make("label", "onx2k-check"); const el = input("checkbox"); el.checked = checked; root.append(el, document.createTextNode(label)); return { root, el }; }

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
  if (widget.element) widget.element.classList.add("onx2k-hidden");
}

function graphLink(graph, node, inputName) {
  const slot = node.inputs?.find((item) => item.name === inputName);
  if (!slot || slot.link == null) return null;
  const link = graph.links?.[slot.link];
  if (!link) return null;
  const originId = Array.isArray(link) ? link[1] : link.origin_id;
  return graph.getNodeById?.(originId) || graph._nodes_by_id?.[originId] || null;
}

async function blobDataUrl(blob) {
  return await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(blob); });
}

async function upstreamImageDataUrl(imageNode) {
  if (!imageNode) return null;
  try {
    if (imageNode.imgs?.[0]) {
      const img = imageNode.imgs[0]; const canvas = make("canvas");
      canvas.width = img.naturalWidth || img.width; canvas.height = img.naturalHeight || img.height;
      if (canvas.width && canvas.height) { canvas.getContext("2d").drawImage(img, 0, 0); return canvas.toDataURL("image/jpeg", .9); }
    }
  } catch (_) {}
  if (String(imageNode.type || "").includes("OnyxImageBatchLoader")) {
    const widget = imageNode.widgets?.find((w) => w.name === "batch_data") || imageNode.widgets?.[0];
    try {
      const data = JSON.parse(widget?.value || "{}"); const id = data.order?.[0]; const meta = data.images?.find((item) => item.id === id);
      if (meta?.filename) { const response = await api.fetchApi(`/onyx/view/${encodeURIComponent(meta.filename)}`); if (response.ok) return await blobDataUrl(await response.blob()); }
    } catch (_) {}
    return null;
  }
  const widget = imageNode.widgets?.find((w) => w.name === "image") || imageNode.widgets?.[0];
  const filename = String(widget?.value || ""); if (!filename) return null;
  const norm = filename.replaceAll("\\", "/"); const parts = norm.split("/"); const base = parts.pop();
  const query = new URLSearchParams({ filename: base, type: "input" }); if (parts.length) query.set("subfolder", parts.join("/"));
  const response = await api.fetchApi(`/view?${query.toString()}`); if (!response.ok) return null;
  return await blobDataUrl(await response.blob());
}

app.registerExtension({
  name: "Onyx.Krea2Carousel.Krea2EditCarouselDirector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    injectStyle();
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const widgets = Object.fromEntries((node.widgets || []).map((w) => [w.name, w]));
      ["pose_list", "slide_count", "output_mode", "slide_selector", "camera_policy", "continuity_strength", "session_direction", "character_trigger"].forEach((key) => hideWidget(widgets[key]));
      node.properties ||= {};
      const storedEngine = node.properties.krea2_pose_engine || {};
      const engine = {
        ...DEFAULT_ENGINE,
        ...storedEngine,
        model_by_backend: { ...MODEL_DEFAULTS, ...(storedEngine.model_by_backend || {}) },
      };
      if (storedEngine.model_id) engine.model_by_backend[engine.backend] = storedEngine.model_id;
      engine.model_id = engine.model_by_backend[engine.backend] ?? MODEL_DEFAULTS[engine.backend] ?? "";
      const ui = {};
      const wrap = make("div", "onx2k-wrap");
      for (const event of ["mousedown", "pointerdown", "wheel", "keydown", "keyup"]) wrap.addEventListener(event, (e) => e.stopPropagation());

      const saveEngine = () => { node.properties.krea2_pose_engine = { ...engine }; node.graph?.setDirtyCanvas(true, true); };
      const setWidget = (name, value) => { const w = widgets[name]; if (!w) return; w.value = value; w.callback?.(value); node.graph?.setDirtyCanvas(true, true); };
      const setStatus = (message, level = "") => { ui.status.textContent = message; ui.status.className = `onx2k-status ${level}`.trim(); };

      const head = make("div", "onx2k-head");
      const brandWrap = make("div"); const brand = make("div", "onx2k-brand"); brand.textContent = "Onyx · Krea2 Carousel Studio";
      const sub = make("div", "onx2k-sub"); sub.textContent = "Identity Edit · pose direction · batch-ready"; brandWrap.append(brand, sub);
      const badges = make("div", "onx2k-badges"); for (const [text, cls] of [["KREA2 EDIT", ""], ["1–8 SLIDES", "blue"], ["VLM/API", "blue"]]) { const b = make("span", `onx2k-badge ${cls}`); b.textContent = text; badges.append(b); } head.append(brandWrap, badges);

      const source = make("div", "onx2k-source"); source.textContent = "MASTER IMAGE · waiting for connection";

      const vibeSection = make("div", "onx2k-section"); const vibeTitle = make("div", "onx2k-title"); vibeTitle.innerHTML = "Pose Vibe Generator <small>one instruction per slide</small>";
      const presetGrid = make("div", "onx2k-presets");
      const chips = {};
      for (const [name, text] of Object.entries(PRESETS)) { const chip = button(name, "onx2k-chip"); chips[name] = chip; chip.addEventListener("click", () => { engine.vibe_preset = name; engine.vibe = text; ui.vibe.value = text; Object.values(chips).forEach((x) => x.classList.remove("active")); chip.classList.add("active"); saveEngine(); }); presetGrid.append(chip); }
      chips[engine.vibe_preset]?.classList.add("active");
      ui.vibe = make("textarea", "onx2k-area"); ui.vibe.value = engine.vibe; ui.vibe.addEventListener("input", () => { engine.vibe = ui.vibe.value; saveEngine(); });
      vibeSection.append(vibeTitle, presetGrid, field("Creative direction / influencer vibe", ui.vibe));

      const engineSection = make("div", "onx2k-section blue"); const engineTitle = make("div", "onx2k-title"); engineTitle.innerHTML = "Pose Intelligence <small>keys never enter the workflow</small>";
      let grokModelIds = [];
      ui.backend = select(["Transformers local", "Ollama", "llama.cpp", "Gemini API", "Grok API"]); ui.backend.value = engine.backend;
      ui.model = input(); ui.model.value = engine.model_id;
      ui.modelList = select([]); ui.modelList.classList.add("onx2k-hidden");
      const modelControl = make("div"); modelControl.append(ui.model, ui.modelList);
      ui.endpoint = input(); ui.endpoint.value = engine.endpoint;
      ui.key = input("password", "onx2k-input onx2k-key"); ui.key.placeholder = "API key stored only for this browser tab";
      ui.refresh = button("↻ Models", "ghost"); ui.generate = button("✦ Generate pose list", "secondary");
      const engineGrid = make("div", "onx2k-grid2"); engineGrid.append(field("Backend", ui.backend), field("Model", modelControl), field("Local endpoint", ui.endpoint), field("Private API key", ui.key));
      const toggles = make("div", "onx2k-row"); const selfie = checkbox("Include one selfie", engine.include_selfie); const unload = checkbox("Unload local VLM after generation", engine.unload_after); toggles.append(selfie.root, unload.root);
      const engineButtons = make("div", "onx2k-row"); engineButtons.append(ui.refresh, ui.generate);
      engineSection.append(engineTitle, engineGrid, toggles, engineButtons);

      function keyStorageName() { return engine.backend === "Gemini API" ? "onx2.krea.gemini" : engine.backend === "Grok API" ? "onx2.krea.grok" : ""; }
      function setGrokModelOptions(ids = grokModelIds) {
        grokModelIds = ids;
        const values = [...new Set([engine.model_id, ...ids].filter(Boolean))];
        const prompt = make("option"); prompt.value = ""; prompt.textContent = "Press Models, then choose a Grok model";
        ui.modelList.replaceChildren(prompt, ...values.map((value) => { const option = make("option"); option.value = value; option.textContent = value; return option; }));
        if (engine.model_id) ui.modelList.value = engine.model_id;
      }
      function refreshEngineFields() {
        const cloud = engine.backend === "Gemini API" || engine.backend === "Grok API";
        const grok = engine.backend === "Grok API";
        ui.key.disabled = !cloud; ui.endpoint.disabled = cloud || engine.backend === "Transformers local";
        ui.refresh.disabled = !(engine.backend === "Transformers local" || engine.backend === "Ollama" || grok);
        engine.model_id = engine.model_by_backend[engine.backend] ?? MODEL_DEFAULTS[engine.backend] ?? "";
        if (engine.backend === "Ollama" && (!engine.endpoint || engine.endpoint.includes("8080"))) engine.endpoint = "http://127.0.0.1:11434";
        if (engine.backend === "llama.cpp" && (!engine.endpoint || engine.endpoint.includes("11434"))) engine.endpoint = "http://127.0.0.1:8080";
        ui.model.classList.toggle("onx2k-hidden", grok); ui.modelList.classList.toggle("onx2k-hidden", !grok);
        if (grok) setGrokModelOptions(); else ui.model.value = engine.model_id;
        ui.endpoint.value = engine.endpoint; ui.key.value = keyStorageName() ? (sessionStorage.getItem(keyStorageName()) || "") : ""; saveEngine();
      }
      ui.backend.addEventListener("change", () => { engine.model_by_backend[engine.backend] = engine.backend === "Grok API" ? ui.modelList.value : ui.model.value.trim(); engine.backend = ui.backend.value; refreshEngineFields(); });
      ui.model.addEventListener("input", () => { engine.model_id = ui.model.value; engine.model_by_backend[engine.backend] = ui.model.value; saveEngine(); });
      ui.modelList.addEventListener("change", () => { engine.model_id = ui.modelList.value; engine.model_by_backend[engine.backend] = ui.modelList.value; saveEngine(); });
      ui.endpoint.addEventListener("input", () => { engine.endpoint = ui.endpoint.value; saveEngine(); });
      ui.key.addEventListener("input", () => {
        const key = keyStorageName(); if (key) sessionStorage.setItem(key, ui.key.value);
        if (engine.backend === "Grok API") {
          grokModelIds = []; engine.model_id = ""; engine.model_by_backend["Grok API"] = "";
          setGrokModelOptions(); saveEngine();
        }
      });
      selfie.el.addEventListener("change", () => { engine.include_selfie = selfie.el.checked; saveEngine(); });
      unload.el.addEventListener("change", () => { engine.unload_after = unload.el.checked; saveEngine(); });

      ui.refresh.addEventListener("click", async () => {
        ui.refresh.disabled = true; setStatus("Reading available models…", "warn");
        try {
          let response;
          if (engine.backend === "Ollama") response = await api.fetchApi("/onyx/nf/ollama/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ host: engine.endpoint }) });
          else if (engine.backend === "Grok API") response = await api.fetchApi("/onyx/nf/grok/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: sessionStorage.getItem("onx2.krea.grok") || "" }) });
          else response = await api.fetchApi("/onyx/nf/local/models", { cache: "no-store" });
          const data = await response.json(); if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
          const ids = (data.models || []).map((item) => item.id).filter(Boolean);
          if (ids.length) {
            if (engine.backend === "Grok API" && !ids.includes(engine.model_id)) engine.model_id = "";
            else if (engine.backend !== "Grok API" && !ids.includes(engine.model_id)) engine.model_id = ids[0];
            engine.model_by_backend[engine.backend] = engine.model_id;
            if (engine.backend === "Grok API") setGrokModelOptions(ids); else ui.model.value = engine.model_id;
            saveEngine();
          }
          setStatus(ids.length ? `Found ${ids.length} model(s). Selected: ${engine.model_id}` : "No compatible models were found.", ids.length ? "" : "warn");
        } catch (error) { setStatus(error.message, "err"); } finally { refreshEngineFields(); }
      });

      const controlSection = make("div", "onx2k-section"); const controlTitle = make("div", "onx2k-title"); controlTitle.innerHTML = "Carousel Control <small>identity / outfit / scene remain locked</small>";
      ui.count = select(["1", "2", "3", "4", "5", "6", "7", "8"]); ui.count.value = String(widgets.slide_count?.value || 4);
      ui.output = select(["All active slides", "Selected slide only"]); ui.output.value = widgets.output_mode?.value || "All active slides";
      ui.selector = select(["1", "2", "3", "4", "5", "6", "7", "8"]); ui.selector.value = String(widgets.slide_selector?.value || 1);
      ui.camera = select(["AUTO", "LOCK", "FREE"]); ui.camera.value = widgets.camera_policy?.value || "AUTO";
      ui.continuity = select(["Balanced", "Strong", "Maximum"]); ui.continuity.value = widgets.continuity_strength?.value || "Strong";
      ui.trigger = input(); ui.trigger.value = widgets.character_trigger?.value || ""; ui.trigger.placeholder = "Optional character LoRA trigger";
      ui.direction = make("textarea", "onx2k-area"); ui.direction.value = widgets.session_direction?.value || ""; ui.direction.placeholder = "Optional direction subordinate to continuity locks";
      const grid3 = make("div", "onx2k-grid3"); grid3.append(field("Slides", ui.count), field("Output", ui.output), field("Selected", ui.selector));
      const grid2 = make("div", "onx2k-grid2"); grid2.append(field("Camera", ui.camera), field("Continuity", ui.continuity));
      controlSection.append(controlTitle, grid3, grid2, field("Character LoRA trigger", ui.trigger), field("Session direction", ui.direction));
      ui.count.addEventListener("change", () => { setWidget("slide_count", Number(ui.count.value)); engine.count = Number(ui.count.value); saveEngine(); });
      ui.output.addEventListener("change", () => setWidget("output_mode", ui.output.value)); ui.selector.addEventListener("change", () => setWidget("slide_selector", Number(ui.selector.value)));
      ui.camera.addEventListener("change", () => setWidget("camera_policy", ui.camera.value)); ui.continuity.addEventListener("change", () => setWidget("continuity_strength", ui.continuity.value));
      ui.trigger.addEventListener("input", () => setWidget("character_trigger", ui.trigger.value)); ui.direction.addEventListener("input", () => setWidget("session_direction", ui.direction.value));

      const posesSection = make("div", "onx2k-section"); const posesTitle = make("div", "onx2k-title"); posesTitle.innerHTML = "Pose List <small>editable · one frame per line</small>";
      ui.poses = make("textarea", "onx2k-area onx2k-poses"); ui.poses.value = widgets.pose_list?.value || ""; ui.poses.addEventListener("input", () => setWidget("pose_list", ui.poses.value));
      const applyPreset = button("Use preset without AI", "ghost"); applyPreset.addEventListener("click", () => { const defaults = ["standing with relaxed confidence, one hand touching her hair and the other resting naturally at her waist", "crossing both arms naturally over her torso with a calm direct gaze", "turning her body away and looking back over one shoulder toward the camera", "taking a close-up selfie from a slightly high camera angle with one arm extended toward the camera", "shifting her weight onto one leg with one hand on her hip and the other hanging naturally", "leaning slightly toward the camera with both hands resting naturally near her thighs", "standing in a candid mid-turn with her hair and arms responding naturally to the movement", "posing in a relaxed three-quarter profile with one hand lightly touching her neck"]; ui.poses.value = defaults.slice(0, Number(ui.count.value)).join("\n"); setWidget("pose_list", ui.poses.value); setStatus("Local influencer preset applied."); });
      posesSection.append(posesTitle, ui.poses, applyPreset);

      ui.generate.addEventListener("click", async () => {
        ui.generate.disabled = true; setStatus(`Generating ${ui.count.value} poses with ${engine.backend}…`, "warn");
        try {
          const upstream = graphLink(node.graph, node, "master_image"); const imageData = await upstreamImageDataUrl(upstream);
          const response = await api.fetchApi("/onyx/nf/krea2/poses/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ backend: engine.backend, model_id: engine.model_id, endpoint: engine.endpoint, api_key: keyStorageName() ? (sessionStorage.getItem(keyStorageName()) || "") : "", vibe: ui.vibe.value, count: Number(ui.count.value), include_selfie: selfie.el.checked, seed: engine.seed || 0, timeout: engine.timeout || 300, four_bit: engine.four_bit, unload_after: unload.el.checked, attention: "auto", image_b64: imageData }) });
          const data = await response.json(); if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
          ui.poses.value = data.pose_list || (data.poses || []).join("\n"); setWidget("pose_list", ui.poses.value); setWidget("slide_count", (data.poses || []).length || Number(ui.count.value));
          setStatus(data.status || "Pose list generated.");
        } catch (error) { setStatus(error.message || String(error), "err"); } finally { ui.generate.disabled = false; }
      });

      const preview = make("details", "onx2k-section"); const summary = make("summary"); summary.textContent = "Last compiled Krea2 edit instruction"; ui.preview = make("div", "onx2k-preview"); ui.preview.textContent = "Queue the workflow to inspect the selected final instruction."; preview.append(summary, ui.preview);
      ui.status = make("div", "onx2k-status");
      wrap.append(head, source, vibeSection, engineSection, controlSection, posesSection, preview, ui.status);
      node.addDOMWidget("onyx_nf_krea2_studio", "div", wrap, { serialize: false, hideOnZoom: false, getMinHeight: () => node.inputs?.some((item) => item.name === "pose_list" && item.link != null) ? 450 : 970 });

      const updateSource = () => { const up = graphLink(node.graph, node, "master_image"); const externalPoses = !!node.inputs?.find((item) => item.name === "pose_list" && item.link != null); source.classList.toggle("ok", !!up); source.textContent = up ? `MASTER SOURCE · ${up.title || up.type} · connected${externalPoses ? " · POSES FROM PREP" : ""}` : "MASTER IMAGE · waiting for connection"; vibeSection.classList.toggle("onx2k-hidden", externalPoses); engineSection.classList.toggle("onx2k-hidden", externalPoses); posesSection.classList.toggle("onx2k-hidden", externalPoses); wrap.style.minHeight = externalPoses ? "450px" : "970px"; node.setSize([Math.max(node.size[0], 760), externalPoses ? 550 : Math.max(node.size[1], 1030)]); };
      chainCallback(node, "onConfigure", () => { ui.poses.value = widgets.pose_list?.value || ui.poses.value; ui.count.value = String(widgets.slide_count?.value || 4); ui.output.value = widgets.output_mode?.value || "All active slides"; ui.camera.value = widgets.camera_policy?.value || "AUTO"; ui.continuity.value = widgets.continuity_strength?.value || "Strong"; updateSource(); refreshEngineFields(); });
      chainCallback(node, "onExecuted", (message) => { const used = message?.pose_list_used?.[0]; if (used) ui.poses.value = used; const prompt = message?.selected_prompt?.[0]; if (prompt) ui.preview.textContent = prompt; const status = message?.status?.[0]; const report = message?.validation_report?.[0]; if (report) try { const parsed = JSON.parse(report); setStatus(`${status || "Ready"}${parsed.warnings?.length ? `\n${parsed.warnings.join("\n")}` : ""}`, parsed.warnings?.length ? "warn" : ""); } catch (_) { if (status) setStatus(status); } else if (status) setStatus(status); });
      const oldConnections = node.onConnectionsChange; node.onConnectionsChange = function () { const result = oldConnections?.apply(this, arguments); updateSource(); return result; };
      refreshEngineFields(); updateSource(); setStatus("Ready · choose a vibe, generate poses, then queue the carousel.");
      setTimeout(() => { updateSource(); node.graph?.setDirtyCanvas(true, true); }, 0);
    });
  },
});
