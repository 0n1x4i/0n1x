import { app } from "../../scripts/app.js";
import { chainCallback } from "./onyx_nf_utility.js";

const NODE_ID = "OnyxPoseCarouselDirector";
const LOCK_KEYS = [
  "identity", "facial_structure", "body_proportions", "skin_details", "hairstyle",
  "outfit", "accessories", "location", "background_geometry", "camera_position",
  "framing", "lens", "expression", "gaze", "lighting_direction", "color_palette",
  "photographic_style",
];

function defaultLocks() {
  const locks = Object.fromEntries(LOCK_KEYS.map((key) => [key, `Locked to Master Image 1 (${key.replaceAll("_", " ")}).`]));
  locks.outfit = "Preserve every garment, design, color, texture, material and coverage from Master Image 1; allow only physically necessary fold and tension changes caused by the new pose.";
  return locks;
}

function defaultSlide(index) {
  return {
    index,
    purpose: `Independent re-pose edit using Pose Reference ${index}.`,
    action: "Transfer pose only.",
    pose: `Use Pose Reference ${index} as the sole pose guide.`,
    expression: "Locked to Master Image 1.",
    framing: "Locked to Master Image 1.",
    camera: "Locked to Master Image 1.",
    lens: "Locked to Master Image 1.",
    background: "Locked to Master Image 1.",
    allowed_changes: ["pose"],
    locked_attributes: [...LOCK_KEYS],
    transition: "Generate independently from the master, never from another slide.",
    prompt_delta: `Use DWPose Map ${index}, routed to Painter Image 2, only for body and hand joints, limb placement and weight distribution. It contributes geometry only.`,
  };
}

function freshState(previous = {}) {
  return {
    version: 2,
    project: String(previous.project || "FLUX.2 Klein pose-only carousel (1-4 slides)"),
    target: previous.target || "FLUX.2 Klein 9B",
    slide_count: 1,
    slide_index: Math.max(1, Math.min(4, Number(previous.slide_index) || 1)),
    seed_base: Math.max(0, Number(previous.seed_base) || 12345),
    seed_mode: previous.seed_mode || "Sequential",
    seed_stride: Math.max(1, Number(previous.seed_stride) || 1),
    grain_seed_offset: Math.max(0, Number(previous.grain_seed_offset) || 1000000),
    require_all_pose_references: true,
    base_prompt: previous.base_prompt || "",
    base_negative_prompt: previous.base_negative_prompt || "",
    global_lock: defaultLocks(),
    slides: [1, 2, 3, 4].map(defaultSlide),
    lock_generated_prompt: false,
    locked_prompts: {},
  };
}

function ensureState(candidate) {
  if (!candidate || Number(candidate.version) !== 2) return freshState(candidate || {});
  const state = candidate;
  state.version = 2;
  state.project ||= "FLUX.2 Klein pose-only carousel (1-4 slides)";
  state.target ||= "FLUX.2 Klein 9B";
  state.slide_count = Math.max(1, Math.min(4, Number(state.slide_count) || 1));
  state.slide_index = Math.max(1, Math.min(state.slide_count, Number(state.slide_index) || 1));
  state.seed_base = Math.max(0, Number(state.seed_base) || 0);
  state.seed_mode ||= "Sequential";
  state.seed_stride = Math.max(1, Number(state.seed_stride) || 1);
  state.grain_seed_offset = Math.max(0, Number(state.grain_seed_offset) || 1000000);
  state.require_all_pose_references = true;
  state.base_prompt ||= "";
  state.base_negative_prompt ||= "";
  state.global_lock = state.global_lock && typeof state.global_lock === "object" ? state.global_lock : defaultLocks();
  state.slides = Array.isArray(state.slides) ? state.slides.slice(0, 4) : [];
  while (state.slides.length < state.slide_count) state.slides.push(defaultSlide(state.slides.length + 1));
  state.slides.forEach((slide, index) => {
    slide.index = index + 1;
    slide.allowed_changes = Array.isArray(slide.allowed_changes) && slide.allowed_changes.length ? slide.allowed_changes : ["pose"];
    slide.locked_attributes = Array.isArray(slide.locked_attributes) && slide.locked_attributes.length ? slide.locked_attributes : [...LOCK_KEYS];
  });
  state.locked_prompts = state.locked_prompts && typeof state.locked_prompts === "object" ? state.locked_prompts : {};
  state.lock_generated_prompt = !!state.lock_generated_prompt;
  return state;
}

function parseState(value) {
  try {
    const parsed = JSON.parse(value || "{}");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch (_) {
    return {};
  }
}

function connected(node, name) {
  return node.inputs?.find((item) => item.name === name)?.link != null;
}

function control(tag, className = "onx2c-input") {
  const element = document.createElement(tag);
  element.className = className;
  return element;
}

function button(label, extra = "") {
  const element = control("button", `onx2c-btn ${extra}`.trim());
  element.type = "button";
  element.textContent = label;
  return element;
}

function field(label, element) {
  const root = document.createElement("label");
  const caption = document.createElement("span");
  caption.className = "onx2c-label";
  caption.textContent = label;
  root.append(caption, element);
  return root;
}

function select(options = []) {
  const element = control("select", "onx2c-input");
  for (const value of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    element.appendChild(option);
  }
  return element;
}

function injectStyle() {
  if (document.getElementById("onx2c-pose-style")) return;
  const style = document.createElement("style");
  style.id = "onx2c-pose-style";
  style.textContent = `
    .onx2c-wrap{box-sizing:border-box;width:100%;padding:10px;color:#dcf9ff;background:linear-gradient(160deg,rgba(8,42,51,.34),rgba(7,26,33,.22) 60%,rgba(4,16,22,.34));font:12px/1.35 Inter,Segoe UI,sans-serif;border:1px solid #1ebbd8aa;border-radius:8px}
    .onx2c-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}.onx2c-title{font-size:14px;font-weight:750;color:#ffffff}.onx2c-badge{padding:3px 7px;border-radius:999px;background:#165561;color:#79eaff;font-size:10px;font-weight:700}
    .onx2c-section{padding:9px;margin-top:8px;background:rgba(16,41,46,0.28);border:1px solid #25545d;border-radius:7px}.onx2c-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.onx2c-grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.onx2c-row{display:flex;gap:6px}.onx2c-row>*{flex:1}
    .onx2c-label{display:block;margin-bottom:3px;color:#90bec6;font-size:10px;text-transform:uppercase;letter-spacing:.04em}.onx2c-input,.onx2c-area{box-sizing:border-box;width:100%;border:1px solid #336f7a;border-radius:5px;background:#081517;color:#edfcff;padding:6px;font:12px/1.35 inherit}.onx2c-area{min-height:68px;resize:vertical}.onx2c-json{min-height:160px;font:11px/1.35 ui-monospace,Consolas,monospace}
    .onx2c-btn{border:1px solid #409bac;background:#164952;color:#eafcff;border-radius:5px;padding:6px 8px;cursor:pointer;font-weight:650}.onx2c-btn:hover{background:#217180}.onx2c-btn.warn{border-color:#9a6840;background:#472b18}.onx2c-btn.danger{border-color:#9b4d57;background:#421b22}
    .onx2c-refs{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-top:7px}.onx2c-ref{padding:5px 2px;text-align:center;border-radius:4px;background:#3b1d22;color:#ff9fa8;font-size:9px}.onx2c-ref.ok{background:#153726;color:#90e6b2}.onx2c-ref.off{background:#182231;color:#71839a}
    .onx2c-status{margin-top:8px;padding:7px;border-radius:5px;background:#102a20;color:#8fe2b1;white-space:pre-wrap}.onx2c-status.warn{background:#392c13;color:#ffd277}.onx2c-status.err{background:#3a171b;color:#ff9da8}.onx2c-check{display:flex;align-items:center;gap:6px;color:#b6dae1}.onx2c-check input{width:auto}
    .onx2c-locks{display:grid;grid-template-columns:1fr 1fr;gap:5px}.onx2c-lock{padding:5px;background:rgba(10,30,34,0.28);border-radius:4px;color:#9ec8d0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.onx2c-preview{max-height:150px;overflow:auto;white-space:pre-wrap;background:rgba(7,16,18,0.28);border:1px solid #23484f;border-radius:5px;padding:7px;color:#cae4e9}.onx2c-mini{font-size:10px;color:#81a8af}details.onx2c-section>summary{cursor:pointer;font-weight:700;color:#bbeef7}
  `;
  document.head.appendChild(style);
}

app.registerExtension({
  name: "Onyx.Krea2Carousel.PoseCarouselDirector",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    injectStyle();

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const stateWidget = node.widgets?.find((widget) => widget.name === "carousel_state");
      if (!stateWidget) return;
      stateWidget.hidden = true;
      stateWidget.computeSize = () => [0, -4];

      let state = ensureState(parseState(stateWidget.value));
      let lastPrompt = "";
      let lastNegative = "";
      const ui = {};
      const wrap = document.createElement("div");
      wrap.className = "onx2c-wrap";
      for (const event of ["mousedown", "pointerdown", "wheel", "keydown"]) wrap.addEventListener(event, (e) => e.stopPropagation());

      const persist = () => {
        stateWidget.value = JSON.stringify(state);
        stateWidget.callback?.(stateWidget.value);
        node.graph?.setDirtyCanvas(true, true);
      };
      const setStatus = (message, level = "") => {
        ui.status.textContent = message || "";
        ui.status.className = `onx2c-status ${level}`.trim();
      };
      const selectedSlide = () => state.slides[state.slide_index - 1];
      const updateRefs = () => {
        const slots = [["MASTER", "master_image"], ["POSE 1", "pose_1"], ["POSE 2", "pose_2"], ["POSE 3", "pose_3"], ["POSE 4", "pose_4"]];
        slots.forEach(([label, name], index) => {
          const ok = connected(node, name);
          const active = index === 0 || index <= state.slide_count;
          ui.refs[index].classList.toggle("ok", active && ok);
          ui.refs[index].classList.toggle("off", !active);
          ui.refs[index].textContent = `${label} · ${active ? (ok ? "OK" : "MISS") : "OFF"}`;
        });
      };
      const renderLocks = () => {
        ui.locks.innerHTML = "";
        for (const key of LOCK_KEYS) {
          const item = document.createElement("div");
          item.className = "onx2c-lock";
          item.title = state.global_lock[key] || "Not configured";
          item.textContent = `${key.replaceAll("_", " ")} · ${state.global_lock[key] ? "LOCK" : "MISSING"}`;
          ui.locks.appendChild(item);
        }
      };
      const renderSlide = () => {
        state = ensureState(state);
        ui.slide.innerHTML = "";
        for (let index = 1; index <= state.slide_count; index += 1) {
          const option = document.createElement("option");
          option.value = String(index);
          option.textContent = `Slide ${index} → Pose ${index}`;
          ui.slide.appendChild(option);
        }
        ui.slide.value = String(state.slide_index);
        ui.slideCount.value = String(state.slide_count);
        const slide = selectedSlide();
        ui.purpose.value = slide.purpose || "";
        ui.delta.value = slide.prompt_delta || "";
        ui.allowed.value = (slide.allowed_changes || []).join(", ");
        ui.locked.value = (slide.locked_attributes || []).join(", ");
        ui.lockMode.checked = !!state.lock_generated_prompt;
        ui.lockInfo.textContent = state.locked_prompts[String(state.slide_index)] ? `Slide ${state.slide_index} has an exact saved prompt.` : `Slide ${state.slide_index} is not saved.`;
        ui.json.value = JSON.stringify({ project: state.project, slide_count: state.slide_count, global_lock: state.global_lock, slides: state.slides }, null, 2);
        renderLocks();
      };

      const head = document.createElement("div");
      head.className = "onx2c-head";
      const title = document.createElement("div"); title.className = "onx2c-title"; title.textContent = "Pose-Only Carousel Director";
      const badge = document.createElement("div"); badge.className = "onx2c-badge"; badge.textContent = "KLEIN 9B · MASTER + POSE";
      head.append(title, badge);

      const projectSection = document.createElement("div"); projectSection.className = "onx2c-section";
      ui.project = control("input"); ui.project.value = state.project;
      ui.project.addEventListener("input", () => { state.project = ui.project.value; persist(); });
      ui.target = select(["FLUX.2 Klein 9B", "Krea 2 (future)"]); ui.target.value = state.target;
      ui.target.addEventListener("change", () => { state.target = ui.target.value; persist(); });
      const projectGrid = document.createElement("div"); projectGrid.className = "onx2c-grid";
      projectGrid.append(field("Project", ui.project), field("Target", ui.target));
      const refs = document.createElement("div"); refs.className = "onx2c-refs"; ui.refs = [];
      for (let index = 0; index < 5; index += 1) { const item = document.createElement("div"); item.className = "onx2c-ref"; ui.refs.push(item); refs.appendChild(item); }
      projectSection.append(projectGrid, refs);

      const slideSection = document.createElement("div"); slideSection.className = "onx2c-section";
      ui.slideCount = select(["1", "2", "3", "4"]); ui.slideCount.value = String(state.slide_count);
      ui.slideCount.addEventListener("change", () => {
        state.slide_count = Math.max(1, Math.min(4, Number(ui.slideCount.value) || 1));
        state.slide_index = Math.min(state.slide_index, state.slide_count);
        state.require_all_pose_references = true;
        persist(); renderSlide(); updateRefs();
        setStatus(`${state.slide_count} active slide(s) · only Pose 1-${state.slide_count} required.`);
      });
      const nav = document.createElement("div"); nav.className = "onx2c-row";
      const previous = button("◀ Previous"); const next = button("Next ▶"); ui.slide = select();
      previous.addEventListener("click", () => { state.slide_index = Math.max(1, state.slide_index - 1); persist(); renderSlide(); });
      next.addEventListener("click", () => { state.slide_index = Math.min(state.slide_count, state.slide_index + 1); persist(); renderSlide(); });
      ui.slide.addEventListener("change", () => { state.slide_index = Number(ui.slide.value); persist(); renderSlide(); });
      nav.append(previous, ui.slide, next);
      ui.purpose = control("input"); ui.delta = control("textarea", "onx2c-area"); ui.allowed = control("input"); ui.locked = control("input");
      ui.purpose.addEventListener("input", () => { selectedSlide().purpose = ui.purpose.value; persist(); });
      ui.delta.addEventListener("input", () => { selectedSlide().prompt_delta = ui.delta.value; persist(); });
      ui.allowed.addEventListener("change", () => { selectedSlide().allowed_changes = ui.allowed.value.split(",").map((v) => v.trim()).filter(Boolean); persist(); });
      ui.locked.addEventListener("change", () => { selectedSlide().locked_attributes = ui.locked.value.split(",").map((v) => v.trim()).filter(Boolean); persist(); });
      slideSection.append(field("Slides to generate · active pose references", ui.slideCount), nav, field("Purpose", ui.purpose), field("Pose-only direction", ui.delta), field("Allowed changes — must remain pose", ui.allowed), field("Locked attributes", ui.locked));

      const seedSection = document.createElement("div"); seedSection.className = "onx2c-section";
      ui.seed = control("input"); ui.seed.type = "number"; ui.seed.min = "0";
      ui.seedMode = select(["Sequential", "Fixed", "Stride"]); ui.stride = control("input"); ui.stride.type = "number"; ui.stride.min = "1";
      ui.seed.value = state.seed_base; ui.seedMode.value = state.seed_mode; ui.stride.value = state.seed_stride;
      ui.seed.addEventListener("change", () => { state.seed_base = Math.max(0, Number(ui.seed.value) || 0); persist(); });
      ui.seedMode.addEventListener("change", () => { state.seed_mode = ui.seedMode.value; persist(); });
      ui.stride.addEventListener("change", () => { state.seed_stride = Math.max(1, Number(ui.stride.value) || 1); persist(); });
      const seedGrid = document.createElement("div"); seedGrid.className = "onx2c-grid3";
      seedGrid.append(field("Seed base", ui.seed), field("Seed mode", ui.seedMode), field("Stride", ui.stride));
      seedSection.append(seedGrid);

      const lockSection = document.createElement("div"); lockSection.className = "onx2c-section";
      const lockLabel = document.createElement("label"); lockLabel.className = "onx2c-check";
      ui.lockMode = document.createElement("input"); ui.lockMode.type = "checkbox";
      ui.lockMode.addEventListener("change", () => { state.lock_generated_prompt = ui.lockMode.checked; persist(); });
      lockLabel.append(ui.lockMode, document.createTextNode("Use saved prompts exactly (byte-stable)"));
      const lockRow = document.createElement("div"); lockRow.className = "onx2c-row";
      const save = button("Save last prompt"); const clear = button("Clear slide", "danger"); ui.lockInfo = document.createElement("div"); ui.lockInfo.className = "onx2c-mini";
      save.addEventListener("click", () => {
        if (!lastPrompt.trim()) { setStatus("Run the selected slide once before saving its prompt.", "warn"); return; }
        state.locked_prompts[String(state.slide_index)] = { prompt: lastPrompt, negative_prompt: lastNegative };
        state.lock_generated_prompt = true; persist(); renderSlide(); setStatus(`Saved exact prompt for slide ${state.slide_index}.`);
      });
      clear.addEventListener("click", () => { delete state.locked_prompts[String(state.slide_index)]; if (!Object.keys(state.locked_prompts).length) state.lock_generated_prompt = false; persist(); renderSlide(); });
      lockRow.append(save, clear); lockSection.append(lockLabel, lockRow, ui.lockInfo);

      const continuity = document.createElement("details"); continuity.className = "onx2c-section";
      const continuitySummary = document.createElement("summary"); continuitySummary.textContent = "Global locks — everything except pose";
      ui.locks = document.createElement("div"); ui.locks.className = "onx2c-locks"; continuity.append(continuitySummary, ui.locks);

      const jsonSection = document.createElement("details"); jsonSection.className = "onx2c-section";
      const jsonSummary = document.createElement("summary"); jsonSummary.textContent = "Storyboard JSON";
      ui.json = control("textarea", "onx2c-area onx2c-json"); const apply = button("Apply storyboard JSON", "warn");
      apply.addEventListener("click", () => {
        try {
          const value = JSON.parse(ui.json.value);
          if (!value || !Array.isArray(value.slides) || !value.slides.length) throw new Error("slides[] is required");
          state.project = value.project || state.project;
          state.global_lock = value.global_lock || state.global_lock;
          state.slides = value.slides.slice(0, 4);
          state.slide_count = Math.max(1, Math.min(state.slides.length, Number(value.slide_count) || state.slide_count || 1));
          state.slide_index = Math.min(state.slide_index, state.slide_count);
          state = ensureState(state); ui.project.value = state.project; persist(); renderSlide(); setStatus("Storyboard JSON applied.");
        } catch (error) { setStatus(`Invalid storyboard JSON: ${error.message}`, "err"); }
      });
      jsonSection.append(jsonSummary, ui.json, apply);

      const previewSection = document.createElement("details"); previewSection.className = "onx2c-section"; previewSection.open = true;
      const previewSummary = document.createElement("summary"); previewSummary.textContent = "Last generated prompt";
      ui.preview = document.createElement("div"); ui.preview.className = "onx2c-preview"; ui.preview.textContent = "Run the node to preview the selected slide prompt.";
      previewSection.append(previewSummary, ui.preview);
      ui.status = document.createElement("div"); ui.status.className = "onx2c-status";

      wrap.append(head, projectSection, slideSection, seedSection, lockSection, continuity, jsonSection, previewSection, ui.status);
      node.addDOMWidget("onyx_nf_pose_carousel_director", "div", wrap, { serialize: false, hideOnZoom: false, getMinHeight: () => 930 });

      const applyState = () => {
        state = ensureState(parseState(stateWidget.value));
        ui.project.value = state.project; ui.target.value = state.target; ui.seed.value = state.seed_base;
        ui.seedMode.value = state.seed_mode; ui.stride.value = state.seed_stride; ui.slideCount.value = String(state.slide_count);
        renderSlide(); updateRefs();
      };
      chainCallback(node, "onConfigure", applyState);
      chainCallback(node, "onExecuted", function (message) {
        const prompt = message?.generated_prompt?.[0]; if (typeof prompt === "string") { lastPrompt = prompt; ui.preview.textContent = prompt; }
        const negative = message?.negative_prompt?.[0]; if (typeof negative === "string") lastNegative = negative;
        const status = message?.status?.[0]; if (status) setStatus(status);
        const report = message?.validation_report?.[0];
        if (report) try { const parsed = JSON.parse(report); if (parsed.warnings?.length) setStatus(`${status || "Ready"}\n${parsed.warnings.join("\n")}`, "warn"); } catch (_) {}
      });
      const oldConnections = node.onConnectionsChange;
      node.onConnectionsChange = function () { const result = oldConnections?.apply(this, arguments); updateRefs(); return result; };

      persist(); renderSlide(); updateRefs(); setStatus(`Ready · ${state.slide_count} active slide(s); only the matching pose references are required.`);
      setTimeout(() => { node.setSize([Math.max(node.size[0], 720), Math.max(node.size[1], 1040)]); node.graph?.setDirtyCanvas(true, true); }, 0);
    });
  },
});
