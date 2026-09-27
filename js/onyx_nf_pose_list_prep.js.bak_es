import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { chainCallback } from "./onyx_nf_utility.js";
import { connectedNode, masterSource, imageViewQuery, poseLines, preparationSignature } from "./onyx_nf_pose_prep_sources.js";

const TYPE = "OnyxKrea2PoseListPrep";
const DEFAULT_MODEL = "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated";
const DEFAULT_VIBE = "Natural Instagram carousel with varied body orientation and hand gestures. Every slide visibly changes the master's pose, especially hand placement. When a selfie is enabled, make it a first-person chest-up photo taken by one extended camera arm, with the free hand relaxed and the phone outside the frame. Output pose and framing only, without scenery or appearance descriptions.";
const DEFAULTS = { mode: "Master Auto", detail: "Simple", after_prepare: "Review first", backend: "Transformers local", model_id: DEFAULT_MODEL, model_by_backend: { "Transformers local": DEFAULT_MODEL, Ollama: "", "llama.cpp": "", "Gemini API": "gemini-3.5-flash", "Grok API": "" }, endpoint: "http://127.0.0.1:11434", vibe: DEFAULT_VIBE, count: 4, include_selfie: true, unload_after: true };

function el(tag, className = "", text = "") { const item = document.createElement(tag); item.className = className; item.textContent = text; return item; }
function field(label, control) { const root = el("label", "onx2p-field"); root.append(el("span", "onx2p-label", label), control); return root; }
function button(text) { const item = el("button", "onx2p-button", text); item.type = "button"; return item; }
function input(type = "text") { const item = el("input", "onx2p-input"); item.type = type; return item; }
function select(options) { const item = el("select", "onx2p-input"); for (const value of options) { const option = el("option", "", value); option.value = value; item.append(option); } return item; }
function batchData(loader) { const widget = loader?.widgets?.find((w) => w.name === "batch_data") || loader?.widgets?.[0]; try { return JSON.parse(widget?.value || "{}"); } catch { return {}; } }
function orderedImages(data) { return (data.order || []).map((id) => (data.images || []).find((item) => item.id === id)).filter(Boolean); }
function dataUrl(blob) { return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(blob); }); }
function injectStyle() {
  if (document.getElementById("onx2-pose-prep-style")) return;
  const style = el("style"); style.id = "onx2-pose-prep-style";
  style.textContent = `.onx2p{box-sizing:border-box;padding:12px;background:linear-gradient(160deg,rgba(8,42,51,.34),rgba(7,26,33,.22) 60%,rgba(4,16,22,.34));color:#eaf8fa;border:1px solid #1ebbd8aa;border-radius:10px;font:12px/1.38 Segoe UI,Arial,sans-serif;min-height:650px}.onx2p-title{font-size:17px;font-weight:800;margin-bottom:3px}.onx2p-sub{font-size:10px;color:#9ac2c9;margin-bottom:12px}.onx2p-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.onx2p-field{display:block}.onx2p-label{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.05em;color:#9ed3dc;margin-bottom:3px}.onx2p-input,.onx2p-area{box-sizing:border-box;width:100%;padding:7px;border:1px solid #35636b;background:#0a1618;color:#f2fbff;border-radius:5px}.onx2p-area{min-height:100px;resize:vertical;font:11px/1.4 Consolas,monospace}.onx2p-section{border:1px solid #275057;border-radius:7px;padding:9px;margin-top:9px;background:rgba(10,29,33,0.28)}.onx2p-button{padding:7px 9px;border:1px solid #3694a6;border-radius:5px;background:#15444c;color:#e8fbff;cursor:pointer;font-weight:700}.onx2p-button:disabled{opacity:.4;cursor:not-allowed}.onx2p-row{display:flex;gap:7px;align-items:center;margin-top:8px}.onx2p-row>*{flex:1}.onx2p-status{margin-top:9px;padding:7px;background:#113124;border:1px solid #2b7052;border-radius:5px;white-space:pre-wrap}.onx2p-status.error{background:#36171c;border-color:#9b424b}.onx2p-status.wait{background:#34290d;border-color:#88701f}.onx2p-check{display:flex;gap:6px;align-items:center}.onx2p-check input{width:auto}.onx2p-hidden{display:none!important}`;
  document.head.append(style);
}

app.registerExtension({
  name: "Onyx.Krea2Carousel.PoseListPrep",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TYPE) return;
    injectStyle();
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      node.properties ||= {};
      const state = { ...DEFAULTS, ...(node.properties.pose_prep || {}), model_by_backend: { ...DEFAULTS.model_by_backend, ...(node.properties.pose_prep?.model_by_backend || {}) } };
      const widgets = Object.fromEntries((node.widgets || []).map((w) => [w.name, w]));
      for (const name of ["prepared_pose_list", "prepared_count"]) { const w = widgets[name]; if (w) { w.hidden = true; w.computeSize = () => [0, -4]; if (w.element) w.element.classList.add("onx2p-hidden"); } }
      const root = el("div", "onx2p");
      for (const event of ["mousedown", "pointerdown", "wheel", "keydown", "keyup"]) root.addEventListener(event, (e) => e.stopPropagation());
      root.append(el("div", "onx2p-title", "CAROUSEL POSE DIRECTOR · local / Gemini / Grok"), el("div", "onx2p-sub", "One master: same person, outfit, location and light. New poses, hands and framing for every slide."));
      const sourceStatus = el("div", "onx2p-sub"); root.append(sourceStatus);
      const mode = select(["Master Auto", "Reference Copy", "Vibe List"]); mode.value = state.mode;
      const detail = select(["Simple", "Detailed"]); detail.value = state.detail;
      const afterPrepare = select(["Review first", "Queue automatically"]); afterPrepare.value = state.after_prepare;
      const backend = select(["Transformers local", "Ollama", "llama.cpp", "Gemini API", "Grok API"]); backend.value = state.backend;
      const model = input(); model.value = state.model_id;
      const grokModels = select([]); grokModels.classList.add("onx2p-hidden");
      const modelWrap = el("div"); modelWrap.append(model, grokModels);
      const endpoint = input(); endpoint.value = state.endpoint;
      const key = input("password"); key.placeholder = "Private key · this browser tab only";
      const count = input("number"); count.min = "1"; count.max = "8"; count.value = String(state.count);
      const vibe = el("textarea", "onx2p-area"); vibe.value = state.vibe;
      const selfie = input("checkbox"); selfie.checked = state.include_selfie;
      const unload = input("checkbox"); unload.checked = state.unload_after;
      const controls = el("div", "onx2p-section");
      const grid = el("div", "onx2p-grid"); grid.append(field("Mode", mode), field("Backend", backend), field("Model", modelWrap), field("Number of poses · 1–8", count), field("Pose detail", detail), field("After preparation", afterPrepare), field("Local endpoint", endpoint), field("Private API key", key));
      const checks = el("div", "onx2p-row"); const selfieLabel = el("label", "onx2p-check", "Include selfie"); selfieLabel.prepend(selfie); const unloadLabel = el("label", "onx2p-check", "Unload local model"); unloadLabel.prepend(unload); checks.append(selfieLabel, unloadLabel);
      const actions = el("div", "onx2p-row"); const modelsButton = button("↻ Models"); const prepareButton = button("✦ Prepare pose list"); actions.append(modelsButton, prepareButton);
      controls.append(grid, field("Creative direction · Master Auto / Vibe List", vibe), checks, actions); root.append(controls);
      const listSection = el("div", "onx2p-section"); listSection.append(el("div", "onx2p-label", "Prepared pose list · editable before rendering"));
      const poses = el("textarea", "onx2p-area"); poses.value = widgets.prepared_pose_list?.value || ""; listSection.append(poses); root.append(listSection);
      const queueButton = button("▶ Queue carousel from fixed master"); root.append(queueButton);
      const status = el("div", "onx2p-status", "Master Auto: choose your master and press Prepare pose list. Reference Copy: upload poses and press Run pose analysis."); root.append(status);
      node.addDOMWidget("pose_prep_studio", "div", root, { serialize: false, hideOnZoom: false, getMinHeight: () => 720 });
      let busy = false;
      let queueing = false;
      let modelsLoading = false;
      let observedBatch = "";
      let grokIds = [];
      const keyName = () => state.backend === "Grok API" ? "onx2.krea.grok" : state.backend === "Gemini API" ? "onx2.krea.gemini" : "";
      const notify = (message, kind = "") => { status.textContent = message; status.className = `onx2p-status ${kind}`.trim(); };
      const save = () => { node.properties.pose_prep = { ...state }; node.graph?.setDirtyCanvas(true, true); };
      const setPrepared = (text, publish = true) => { poses.value = text; if (widgets.prepared_pose_list) widgets.prepared_pose_list.value = publish ? poseLines(text).join("\n") : ""; if (widgets.prepared_count) widgets.prepared_count.value = poseLines(text).length || 1; node.graph?.setDirtyCanvas(true, true); };
      const imageList = () => orderedImages(batchData(connectedNode(node, "pose_reference")));
      const signature = () => preparationSignature(state, masterSource(node), imageList());
      const ready = () => { const lines = poseLines(poses.value); return !busy && !queueing && !!lines.length && lines.length <= 8 && node.properties.pose_prep_signature === signature(); };
      const updateQueue = () => {
        const locked = busy || queueing;
        queueButton.disabled = !ready(); prepareButton.disabled = locked;
        for (const control of [mode, backend, model, grokModels, detail, afterPrepare, unload, poses]) control.disabled = locked;
        count.disabled = vibe.disabled = selfie.disabled = locked || state.mode === "Reference Copy";
        const cloud = ["Gemini API", "Grok API"].includes(state.backend);
        key.disabled = locked || !cloud; endpoint.disabled = locked || cloud || state.backend === "Transformers local";
        modelsButton.disabled = locked || modelsLoading || !["Grok API", "Ollama", "Transformers local"].includes(state.backend);
      };
      const invalidate = (message) => { node.properties.pose_prep_signature = ""; if (widgets.prepared_pose_list) widgets.prepared_pose_list.value = ""; updateQueue(); notify(message, "wait"); save(); };
      const refreshControls = () => {
        const cloud = state.backend === "Gemini API" || state.backend === "Grok API";
        key.disabled = !cloud; key.value = keyName() ? (sessionStorage.getItem(keyName()) || "") : "";
        endpoint.disabled = cloud || state.backend === "Transformers local";
        modelsButton.disabled = !["Grok API", "Ollama", "Transformers local"].includes(state.backend);
        model.classList.toggle("onx2p-hidden", state.backend === "Grok API"); grokModels.classList.toggle("onx2p-hidden", state.backend !== "Grok API");
        model.value = state.model_by_backend[state.backend] ?? state.model_id ?? "";
        state.model_id = model.value;
        if (state.backend === "Grok API") { grokModels.replaceChildren(); const blank = el("option", "", "Press Models, then select Grok"); blank.value = ""; grokModels.append(blank); for (const id of [...new Set([state.model_id, ...grokIds].filter(Boolean))]) { const option = el("option", "", id); option.value = id; grokModels.append(option); } grokModels.value = state.model_id || ""; }
        updateQueue(); save();
      };
      const requireModel = () => { if (!state.model_id.trim()) throw new Error("Selecciona un modelo antes de preparar las poses."); if (keyName() && !sessionStorage.getItem(keyName())) throw new Error(`Falta la API key de ${state.backend}.`); };
      const generateOne = async (imageData, generationMode, requestedCount, request) => {
        if (signature() !== request.signature) throw new Error("La fuente o los ajustes cambiaron. Prepara la lista de nuevo.");
        const response = await api.fetchApi("/onyx/nf/krea2/poses/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ backend: request.backend, model_id: request.model_id, endpoint: request.endpoint, api_key: request.api_key, vibe: request.vibe, count: requestedCount, mode: generationMode, detail: request.detail, include_selfie: request.include_selfie, unload_after: request.unload_after, image_b64: imageData }) });
        const data = await response.json(); if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`); return data.poses || [];
      };
      const prepare = async () => {
        if (busy || queueing) return;
        busy = true; updateQueue(); setPrepared("");
        const requestedSignature = signature();
        const request = { ...state, api_key: keyName() ? sessionStorage.getItem(keyName()) : "", signature: requestedSignature };
        node.properties.pose_prep_signature = "";
        try {
          requireModel();
          const lines = [];
          if (request.mode === "Reference Copy") {
            const refs = imageList();
            if (!refs.length) throw new Error("Sube al menos una imagen de pose al Batch Loader conectado.");
            if (refs.length > 8) throw new Error("Máximo 8 referencias de pose por carrusel.");
            for (let index = 0; index < refs.length; index++) {
              if (signature() !== requestedSignature) throw new Error("Las referencias o ajustes cambiaron. Prepara la lista de nuevo.");
              notify(`Analizando pose ${index + 1}/${refs.length}: ${refs[index].original_name || refs[index].filename}…`, "wait");
              const response = await api.fetchApi(`/onyx/view/${encodeURIComponent(refs[index].filename)}`);
              if (!response.ok) throw new Error(`No se pudo leer ${refs[index].original_name || refs[index].filename}.`);
              const imageData = await dataUrl(await response.blob());
              const result = await generateOne(imageData, "Reference Copy", 1, request);
              if (!result[0]) throw new Error(`El modelo no devolvió pose para referencia ${index + 1}.`);
              lines.push(result[0]); setPrepared(lines.join("\n"), false);
            }
          } else if (request.mode === "Master Auto") {
            const master = masterSource(node);
            if (!master) throw new Error("Conecta el Load Image de tu master a master_image y selecciona una imagen.");
            notify(`Analizando master y creando ${state.count} poses: ${master.filename}…`, "wait");
            const response = await api.fetchApi(`/view?${imageViewQuery(master)}`, { cache: "no-store" });
            if (!response.ok) throw new Error("No se pudo leer la imagen maestra seleccionada.");
            lines.push(...await generateOne(await dataUrl(await response.blob()), "Master Auto", request.count, request));
          } else {
            notify(`Generando ${state.count} poses desde el vibe…`, "wait");
            lines.push(...await generateOne(null, "Vibe List", request.count, request));
          }
          const expectedCount = request.mode === "Reference Copy" ? imageList().length : request.count;
          if (poseLines(lines.join("\n")).length !== expectedCount) throw new Error(`Se esperaban ${expectedCount} poses distintas. Reintenta o ajusta las referencias.`);
          if (signature() !== requestedSignature) throw new Error("Las referencias o ajustes cambiaron durante el análisis. Prepara la lista de nuevo.");
          setPrepared(lines.join("\n")); node.properties.pose_prep_signature = requestedSignature;
          notify(`${lines.length} poses preparadas. Revísalas y pulsa Queue carousel cuando estén listas.`);
          save();
        } catch (error) { node.properties.pose_prep_signature = ""; setPrepared(""); notify(error.message || String(error), "error"); }
        finally { busy = false; updateQueue(); }
        if (request.after_prepare === "Queue automatically" && ready()) await queueCarousel();
      };
      const setupReferenceLoader = () => {
        const loader = connectedNode(node, "pose_reference");
        const loaderRoot = loader?.widgets?.find((w) => w.name === "images_display")?.element;
        if (!loaderRoot) return;
        const uploadButton = [...loaderRoot.querySelectorAll("button")].find((item) => item.textContent.includes("Upload"));
        for (const item of loaderRoot.querySelectorAll("button")) if (item.textContent.includes("Queue All")) item.style.display = "none";
        if (!uploadButton) return;
        let runButton = loaderRoot.querySelector("[data-onx2-pose-run]");
        if (!runButton) {
          runButton = button("▶ Run pose analysis");
          runButton.dataset.onx2PoseRun = "1";
          runButton.title = "Send uploaded references to the selected local/Gemini/Grok model; does not queue Krea2.";
          runButton.addEventListener("click", () => { void prepare(); });
          uploadButton.after(runButton);
        }
        runButton.disabled = busy || queueing || state.mode !== "Reference Copy" || imageList().length === 0;
      };
      mode.addEventListener("change", () => { state.mode = mode.value; invalidate("Modo cambiado. Prepara la lista antes de renderizar."); refreshControls(); });
      detail.addEventListener("change", () => { state.detail = detail.value; invalidate("Detalle cambiado. Prepara la lista de nuevo."); });
      afterPrepare.addEventListener("change", () => { state.after_prepare = afterPrepare.value; save(); });
      backend.addEventListener("change", () => { state.model_by_backend[state.backend] = state.backend === "Grok API" ? grokModels.value : model.value; state.backend = backend.value; state.model_id = state.model_by_backend[state.backend] || ""; invalidate("Backend cambiado. Prepara la lista de nuevo."); refreshControls(); });
      model.addEventListener("input", () => { state.model_id = model.value; state.model_by_backend[state.backend] = model.value; invalidate("Modelo cambiado. Prepara la lista de nuevo."); });
      grokModels.addEventListener("change", () => { state.model_id = grokModels.value; state.model_by_backend[state.backend] = grokModels.value; invalidate("Modelo cambiado. Prepara la lista de nuevo."); });
      endpoint.addEventListener("input", () => { state.endpoint = endpoint.value; invalidate("Endpoint cambiado. Prepara la lista de nuevo."); });
      vibe.addEventListener("input", () => { state.vibe = vibe.value; invalidate("Dirección cambiada. Prepara la lista de nuevo."); });
      count.addEventListener("input", () => { state.count = Math.max(1, Math.min(8, Math.trunc(Number(count.value)) || 4)); invalidate("Cantidad cambiada. Prepara la lista de nuevo."); });
      selfie.addEventListener("change", () => { state.include_selfie = selfie.checked; invalidate("Regla de selfie cambiada. Prepara la lista de nuevo."); });
      unload.addEventListener("change", () => { state.unload_after = unload.checked; save(); });
      key.addEventListener("input", () => { if (keyName()) sessionStorage.setItem(keyName(), key.value); if (state.backend === "Grok API") { grokIds = []; state.model_id = ""; state.model_by_backend["Grok API"] = ""; refreshControls(); } invalidate("Clave cambiada. Revisa el modelo y prepara la lista."); });
      poses.addEventListener("input", () => { if (widgets.prepared_pose_list) widgets.prepared_pose_list.value = node.properties.pose_prep_signature === signature() ? poseLines(poses.value).join("\n") : ""; if (widgets.prepared_count) widgets.prepared_count.value = poseLines(poses.value).length || 1; updateQueue(); node.graph?.setDirtyCanvas(true, true); });
      modelsButton.addEventListener("click", async () => {
        if (modelsLoading) return;
        modelsLoading = true; updateQueue();
        const requestedBackend = state.backend;
        const requestedKey = sessionStorage.getItem("onx2.krea.grok") || "";
        const requestedSignature = signature();
        const unchanged = () => !busy && signature() === requestedSignature &&
          (requestedBackend !== "Grok API" || requestedKey === (sessionStorage.getItem("onx2.krea.grok") || ""));
        try {
          let response;
          if (requestedBackend === "Grok API") response = await api.fetchApi("/onyx/nf/grok/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: requestedKey }) });
          else if (requestedBackend === "Ollama") response = await api.fetchApi("/onyx/nf/ollama/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ host: state.endpoint }) });
          else response = await api.fetchApi("/onyx/nf/local/models");
          const result = await response.json();
          if (!unchanged()) return;
          if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
          const ids = (result.models || []).map((x) => x.id).filter(Boolean);
          if (requestedBackend === "Grok API") { grokIds = ids; if (!ids.includes(state.model_id)) state.model_id = ""; }
          else if (ids.length && !ids.includes(state.model_id)) state.model_id = ids[0];
          state.model_by_backend[requestedBackend] = state.model_id;
          refreshControls(); notify(`${ids.length} modelos disponibles. Elige uno y prepara la lista.`);
        } catch (error) { if (unchanged()) notify(error.message || String(error), "error"); }
        finally { modelsLoading = false; updateQueue(); }
      });
      prepareButton.addEventListener("click", prepare);
      const queueCarousel = async () => {
        if (!ready()) return;
        if (node.graph !== app.graph) { notify("Vuelve a este workflow para encolar el carrusel preparado.", "wait"); return; }
        queueing = true; updateQueue();
        try { await app.queuePrompt(0, 1); notify("Carrusel enviado a la cola. Revisa la cola de ComfyUI; el mismo master se usa en todos los slides."); }
        catch (error) { notify(error.message || String(error), "error"); }
        finally { queueing = false; updateQueue(); }
      };
      queueButton.addEventListener("click", queueCarousel);
      chainCallback(node, "onConfigure", (info) => { const saved = info?.properties?.pose_prep || {}; Object.assign(state, saved); state.model_by_backend = { ...DEFAULTS.model_by_backend, ...(saved.model_by_backend || {}) }; mode.value = state.mode; detail.value = state.detail; afterPrepare.value = state.after_prepare; backend.value = state.backend; endpoint.value = state.endpoint; vibe.value = state.vibe; count.value = String(state.count); selfie.checked = state.include_selfie; unload.checked = state.unload_after; poses.value = widgets.prepared_pose_list?.value || ""; observedBatch = ""; refreshControls(); });
      const scanBatch = () => {
        setupReferenceLoader();
        const refs = imageList();
        const master = masterSource(node);
        sourceStatus.textContent = state.mode === "Master Auto" ? (master ? `MASTER → ${master.filename} · ${state.count} new poses` : "MASTER → connect the fixed Load Image to master_image") : state.mode === "Reference Copy" ? `POSE REFERENCES → ${refs.length} image(s) · one pose each` : "VIBE LIST → text direction only";
        const marker = signature();
        if (marker === observedBatch) return;
        observedBatch = marker;
        if (node.properties.pose_prep_signature === signature() && poses.value.trim()) { updateQueue(); return; }
        invalidate(state.mode === "Reference Copy" ? (refs.length ? "Referencias listas. Pulsa Run pose analysis para enviarlas al modelo elegido." : "Sube referencias de pose para preparar la lista.") : "Master o ajustes listos. Pulsa Prepare pose list para crear el carrusel.");
      };
      const batchTimer = setInterval(scanBatch, 500);
      chainCallback(node, "onRemoved", () => clearInterval(batchTimer));
      chainCallback(node, "onDrawForeground", scanBatch);
      refreshControls(); updateQueue();
      node.setSize([Math.max(node.size[0], 710), Math.max(node.size[1], 860)]);
    });
  },
});
