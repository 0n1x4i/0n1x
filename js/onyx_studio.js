// Onyx Studio ✦ — the creation hub UI.
// Modes: manual (optionally improved by a skill) · idea (skill) · recreate (Grok
// presets, image -> prompt). Tabs: Prompt · Gallery (send any saved prompt to
// the prompt) · Media (the Batch Loader's panel, embedded) · Character (LoRA +
// trigger) · Skills (import .txt/.md) · Post (look + motion for every Onyx
// post-processing node) · Settings (keys, Ollama) · History. Everything the run needs is kept in the hidden `studio_state`
// widget (JSON); API keys are saved on this computer, never in the workflow.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { mountBatchPanel } from "./onyx_image_batch_loader.js";

const NODE_SIZE = [780, 920];
const PANEL_MIN_H = 600;
const RATIOS = ["9:16", "3:4", "4:5", "1:1", "2:3", "16:9"];
const MODE_INFO = {
  manual:   { icon: "✍️", label: "Manual",   hint: "Your prompt. {trigger} = the character's trigger. Optionally improved by a skill." },
  idea:     { icon: "💡", label: "Idea",     hint: "Describe a scene: the skill + LLM writes the prompt." },
  recreate: { icon: "🎯", label: "Recreate", hint: "Each batch image becomes a prompt for your character, with a Grok preset." },
};
const LABEL = { manual: "Manual", gallery: "Gallery", idea: "Idea", recreate: "Recreate" };
const DEFAULT_STATE = {
  mode: "manual", manual: "", idea: "", extra: "", gallery: null,
  skill: "", character_id: "", provider: "ollama", model: "", ollama_url: "http://127.0.0.1:11434",
  temperature: 0.7, max_tokens: 2048, double_check: false, force_trigger: true, ratio: "9:16",
  extra_loras: [], tab: "prompt", enhance: false, preset: "", post_look: "none", post_motion: "none",
  res: { mp: 1.0, mult: 32, follow_skill: false, batch: 1 },
};

// ── tiny DOM helpers ─────────────────────────────────────────────────────────
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "value") e.value = v;
    else if (k === "checked") e.checked = !!v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}
// replaceChildren() would print a null / false child as the text "null".
const fill = (el, ...kids) => el.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
async function getJSON(url) { return (await api.fetchApi(url)).json(); }
async function postJSON(url, body) {
  return (await api.fetchApi(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
}
function toast(detail, severity = "info") {
  if (app.extensionManager?.toast?.add) app.extensionManager.toast.add({ severity, summary: "Onyx Studio", detail, life: 4500 });
  else console.log("[Onyx Studio] " + detail);
}
function fmtAge(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "now"; if (s < 3600) return `${Math.floor(s / 60)} min`; if (s < 86400) return `${Math.floor(s / 3600)} h`;
  return `${Math.floor(s / 86400)} d`;
}

// ── styles ───────────────────────────────────────────────────────────────────
function injectStyle() {
  if (document.getElementById("onyx-studio-style")) return;
  const css = `
.ost{--bg:#0c0c10;--p1:#121218;--p2:#17171f;--p3:#1d1d27;--ln:#24242f;--ln2:#33333f;--tx:#ececf3;--dim:#8a8c99;
  --ac:#9fb4ff;--ac2:#c7d2ff;--ok:#3ecf8e;--warn:#f0b43c;--bad:#ef5b5b;
  display:flex;flex-direction:column;width:100%;height:100%;box-sizing:border-box;background:var(--bg);color:var(--tx);
  border:1px solid var(--ln);border-radius:12px;overflow:hidden;font:12.5px/1.45 Inter,"Segoe UI",system-ui,sans-serif}
.ost *{box-sizing:border-box}
.ost-head{display:flex;align-items:center;gap:10px;padding:10px 12px;background:linear-gradient(120deg,#171a2b 0%,#141420 55%,#1b1426 100%);border-bottom:1px solid var(--ln)}
.ost-logo{font-weight:800;letter-spacing:.14em;font-size:13px;background:linear-gradient(90deg,#c7d2ff,#e4b8ff);-webkit-background-clip:text;background-clip:text;color:transparent;white-space:nowrap}
.ost-pill{display:flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;background:var(--p3);border:1px solid var(--ln2);font-size:11.5px;white-space:nowrap;cursor:pointer;max-width:220px;overflow:hidden;text-overflow:ellipsis}
.ost-pill b{color:var(--ac2)}
.ost-seg{display:flex;background:var(--p1);border:1px solid var(--ln2);border-radius:10px;padding:2px;margin-left:auto}
.ost-seg button{background:transparent;border:0;color:var(--dim);padding:5px 10px;border-radius:8px;cursor:pointer;font:600 11.5px Inter,system-ui,sans-serif;white-space:nowrap}
.ost-seg button.on{background:linear-gradient(135deg,#2a3156,#30264a);color:var(--tx);box-shadow:0 0 0 1px #46507e inset}
.ost-tabs{display:flex;gap:2px;padding:6px 10px 0;background:var(--p1);border-bottom:1px solid var(--ln)}
.ost-tab{padding:7px 12px;border-radius:8px 8px 0 0;cursor:pointer;color:var(--dim);font-weight:600;font-size:12px;border:1px solid transparent;border-bottom:0;white-space:nowrap}
.ost-tab.on{background:var(--bg);color:var(--tx);border-color:var(--ln)}
.ost-tab .n{display:inline-block;min-width:16px;padding:0 5px;margin-left:5px;border-radius:8px;background:var(--p3);color:var(--ac2);font-size:10.5px;text-align:center}
.ost-body{flex:1;min-height:0;position:relative}
.ost-page{position:absolute;inset:0;overflow:auto;padding:12px;display:none;scrollbar-width:thin;scrollbar-color:var(--ln2) transparent}
.ost-page.on{display:block}
.ost-page.media{padding:0;overflow:hidden}
.ost-page.media.on{display:flex}
.ost-page.media>*{flex:1;min-width:0}
.ost-card{background:var(--p1);border:1px solid var(--ln);border-radius:10px;padding:10px;margin-bottom:10px}
.ost-card h4{margin:0 0 8px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:700}
.ost-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.ost-row:last-child{margin-bottom:0}
.ost-col{display:flex;flex-direction:column;gap:4px;flex:1;min-width:120px}
.ost label{color:var(--dim);font-size:11px}
.ost input[type=text],.ost input[type=password],.ost input[type=number],.ost select,.ost textarea{
  background:var(--p2);color:var(--tx);border:1px solid var(--ln2);border-radius:8px;padding:6px 8px;font:12.5px Inter,system-ui,sans-serif;width:100%;outline:none}
.ost textarea{resize:vertical;min-height:70px;line-height:1.5}
.ost input:focus,.ost select:focus,.ost textarea:focus{border-color:#5b6aa8;box-shadow:0 0 0 2px rgba(159,180,255,.15)}
.ost-btn{background:var(--p3);color:var(--tx);border:1px solid var(--ln2);border-radius:8px;padding:6px 12px;cursor:pointer;font:600 12px Inter,system-ui,sans-serif;white-space:nowrap}
.ost-btn:hover{border-color:var(--ac)}
.ost-btn.pri{background:linear-gradient(135deg,#5468c9,#7a52c7);border-color:transparent;color:#fff}
.ost-btn.pri:hover{filter:brightness(1.1)}
.ost-btn.ghost{background:transparent}
.ost-btn.bad{color:#ff9b9b}
.ost-btn:disabled{opacity:.45;cursor:default}
.ost-hint{color:var(--dim);font-size:11.5px}
.ost-tog{display:flex;align-items:center;gap:7px;cursor:pointer;color:var(--tx);font-size:12px;user-select:none}
.ost-tog i{width:30px;height:16px;border-radius:8px;background:#2a2a35;position:relative;transition:background .15s;flex:none}
.ost-tog i:after{content:"";position:absolute;top:2px;left:2px;width:12px;height:12px;border-radius:50%;background:#9a9aa8;transition:left .15s,background .15s}
.ost-tog.on i{background:#5468c9}.ost-tog.on i:after{left:16px;background:#fff}
.ost-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.ost-gal{border:1px solid var(--ln);border-radius:9px;overflow:hidden;background:var(--p2);cursor:pointer;transition:border-color .12s}
.ost-gal:hover{border-color:var(--ln2)}
.ost-gal.on{border-color:var(--ac);box-shadow:0 0 0 2px rgba(159,180,255,.25)}
.ost-gal img,.ost-gal .ph{display:block;width:100%;aspect-ratio:4/3;object-fit:cover;background:#0f0f15}
.ost-gal .ph{display:flex;align-items:center;justify-content:center;color:var(--dim);font-size:22px}
.ost-gal .t{padding:5px 7px;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ost-list{display:flex;flex-direction:column;gap:6px}
.ost-item{display:flex;gap:8px;align-items:center;padding:8px 10px;border:1px solid var(--ln);border-radius:9px;background:var(--p2);cursor:pointer}
.ost-item.on{border-color:var(--ac)}
.ost-item .grow{flex:1;min-width:0}
.ost-item .ttl{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ost-item .sub{color:var(--dim);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ost-badge{font-size:10px;font-weight:700;padding:2px 6px;border-radius:5px;background:var(--p3);color:var(--ac2);text-transform:uppercase;letter-spacing:.05em;flex:none}
.ost-badge.grok{color:#ffc58a}.ost-badge.worker{color:#8affc8}
.ost-split{display:grid;grid-template-columns:220px 1fr;gap:10px;height:100%}
.ost-lora{display:grid;grid-template-columns:22px 1fr 70px 70px 28px;gap:6px;align-items:center;margin-bottom:6px}
.ost-pre{white-space:pre-wrap;background:var(--p2);border:1px solid var(--ln);border-radius:8px;padding:8px;max-height:220px;overflow:auto;font:11.5px/1.5 ui-monospace,Consolas,monospace;color:#d6d8e4;user-select:text}
.ost-foot{border-top:1px solid var(--ln);background:var(--p1);padding:8px 12px;display:flex;flex-direction:column;gap:6px}
.ost-out{font:12px/1.5 Inter,system-ui,sans-serif;color:#d9dbe6;max-height:62px;overflow:auto;white-space:pre-wrap;user-select:text;scrollbar-width:thin}
.ost-out.empty{color:var(--dim);font-style:italic}
.ost-chips{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.ost-chip{font-size:11px;padding:2px 8px;border-radius:999px;background:var(--p3);border:1px solid var(--ln2);color:var(--ac2)}
.ost-chip.post{color:#ffc58a}
.ost-spin{display:inline-block;width:12px;height:12px;border:2px solid #ffffff44;border-top-color:#fff;border-radius:50%;animation:ostspin .8s linear infinite;vertical-align:-2px;margin-right:6px}
@keyframes ostspin{to{transform:rotate(360deg)}}
.ost-drop{border:2px dashed var(--ln2);border-radius:10px;padding:14px;text-align:center;color:var(--dim);cursor:pointer}
.ost-drop.hot{border-color:var(--ac);color:var(--tx);background:rgba(159,180,255,.06)}
.ost-empty{color:var(--dim);text-align:center;padding:24px 10px}
.ost-sbs{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:10px;align-items:stretch}
.ost-sbs .img{background:#0f0f15;border:1px solid var(--ln);border-radius:9px;display:flex;align-items:center;justify-content:center;min-height:240px;overflow:hidden}
.ost-sbs .img img{max-width:100%;max-height:320px;object-fit:contain;display:block}
.ost-sbs .txt{background:var(--p2);border:1px solid var(--ln);border-radius:9px;padding:9px;font-size:12px;line-height:1.5;white-space:pre-wrap;overflow:auto;max-height:320px;user-select:text}
.ost-sbs .txt.empty{color:var(--dim);font-style:italic;display:flex;align-items:center;justify-content:center;text-align:center}
.ost-presets{display:flex;align-items:center;gap:5px;padding:6px 10px;background:var(--p1);border-bottom:1px solid var(--ln);flex-wrap:wrap}
.ost-presets .lbl{color:var(--dim);font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-right:4px}
.ost-ps{min-width:28px;max-width:118px;padding:4px 8px;border-radius:7px;border:1px solid var(--ln2);background:var(--p2);color:var(--dim);
  font:600 11px Inter,system-ui,sans-serif;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ost-ps.full{color:var(--tx);border-color:#46507e;background:#1c2038}
.ost-ps.cur{box-shadow:0 0 0 2px rgba(159,180,255,.45)}
.ost-ps.saving{border-style:dashed;border-color:#ffc58a;color:#ffc58a}
.ost-ps:hover{border-color:var(--ac)}
`;
  document.head.append(h("style", { id: "onyx-studio-style", html: css }));
}

function hideWidget(w) {
  if (!w) return;
  w.hidden = true; w.computeSize = () => [0, -4]; w.draw = () => {};
  if (w.element) w.element.style.cssText = "display:none!important";
}

function toggle(label, value, onChange, title) {
  const t = h("div", { class: "ost-tog" + (value ? " on" : ""), title }, h("i"), label);
  t.addEventListener("click", () => { t.classList.toggle("on"); onChange(t.classList.contains("on")); });
  return t;
}

// ── main ─────────────────────────────────────────────────────────────────────
function setupStudio(node) {
  injectStyle();
  const W = (n) => node.widgets?.find((w) => w.name === n);
  let state = { ...DEFAULT_STATE };
  let boot = { skills: [], presets: [], characters: [], keys: {}, history: [], loras: [], providers: [], models: {}, post_looks: [], post_motions: [] };
  const triggerCache = {};
  async function loraTrigger(name) {
    if (!name) return "";
    if (!(name in triggerCache)) {
      const r = await getJSON("/onyx/studio/lora_trigger?name=" + encodeURIComponent(name)).catch(() => ({}));
      triggerCache[name] = r.trigger || "";
    }
    return triggerCache[name];
  }
  let ollamaModels = [];
  let gallery = [];
  const mediaPrompts = {};      // filename -> last prompt written for it (preview or run)
  let runMedia = null;          // the batch item the current run is processing
  let galleryLoaded = false;
  let lastResult = null;
  let busy = false;

  const readState = () => {
    try { state = { ...DEFAULT_STATE, ...(JSON.parse(W("studio_state")?.value || "{}") || {}) }; }
    catch { state = { ...DEFAULT_STATE }; }
  };
  const saveState = () => {
    const w = W("studio_state");
    if (w) w.value = JSON.stringify(state);
    app.graph?.setDirtyCanvas(true, true);
  };
  const set = (patch, rerender = true) => { Object.assign(state, patch); saveState(); if (rerender) render(); };
  readState();

  // ── skeleton ───────────────────────────────────────────────────────────────
  const root = h("div", { class: "ost" });
  const stop = (e) => { if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) e.stopPropagation(); };
  root.addEventListener("keydown", stop, true);
  root.addEventListener("wheel", (e) => {
    const sc = e.target.closest?.(".ost-page,.ost-out,.ost-pre,textarea,.obl-grid-wrap,.obl-set-body");
    if (sc && sc.scrollHeight > sc.clientHeight) e.stopPropagation();
  }, { passive: true });

  const charPill = h("div", { class: "ost-pill", title: "Character used for this run", onclick: () => set({ tab: "character" }) });
  const modeSeg = h("div", { class: "ost-seg" });
  const head = h("div", { class: "ost-head" }, h("div", { class: "ost-logo" }, "ONYX STUDIO ✦"), charPill, modeSeg);

  const TABS = [["prompt", "✍️ Prompt"], ["gallery", "🖼️ Gallery"], ["media", "🎞️ Media"], ["character", "🧑 Character"], ["skills", "📚 Skills"], ["post", "🎛️ Post"], ["settings", "⚙️ Settings"], ["history", "🕘 History"]];
  const tabsBar = h("div", { class: "ost-tabs" });
  const body = h("div", { class: "ost-body" });
  const pages = {};
  for (const [id] of TABS) { pages[id] = h("div", { class: "ost-page" + (id === "media" ? " media" : "") }); body.append(pages[id]); }

  const outBox = h("div", { class: "ost-out empty" }, "No prompt yet — the last generated or typed prompt shows here.");
  const chips = h("div", { class: "ost-chips" });
  const genBtn = h("button", { class: "ost-btn pri", onclick: () => preview() }, "✦ Generate now");
  const useBtn = h("button", { class: "ost-btn", title: "Copy this prompt into Manual mode (locks it)", onclick: () => {
    if (!lastResult?.prompt) return; set({ mode: "manual", manual: lastResult.prompt, tab: "prompt" }); toast("Prompt locked in Manual mode.");
  } }, "⤵ Use as manual");
  const copyBtn = h("button", { class: "ost-btn ghost", onclick: () => { if (lastResult?.prompt) navigator.clipboard?.writeText(lastResult.prompt); } }, "⧉ Copy");
  const saveGalBtn = h("button", { class: "ost-btn ghost", title: "Save to the Onyx Prompt Gallery", onclick: async () => {
    if (!lastResult?.prompt) return;
    const r = await postJSON("/onyx/prompt_library/save", { node_id: String(node.id), prompt: lastResult.prompt, name: "", tags: "studio" });
    toast(r.success ? "Saved to the Prompt Gallery." : "Save failed: " + r.error, r.success ? "success" : "error");
    gallery = [];
  } }, "★ Save");
  const queueBtn = h("button", { class: "ost-btn", title: "Queue one run (Queue All is in the Media tab)", onclick: () => app.queuePrompt(0, 1) }, "▶ Queue");
  const foot = h("div", { class: "ost-foot" }, outBox,
    h("div", { class: "ost-row", style: "margin:0;justify-content:space-between" }, chips,
      h("div", { class: "ost-row", style: "margin:0" }, copyBtn, saveGalBtn, useBtn, genBtn, queueBtn)));

  // ── presets bar (10 slots, every setting of the node) ──────────────────────
  const presetBar = h("div", { class: "ost-presets" });
  let presetSaveMode = false;
  const SKIP_WIDGETS = new Set(["batch_data", "studio_state", "studio_panel", "seed", "control_after_generate"]);
  function widgetSnapshot() {
    const out = {};
    for (const w of node.widgets || []) {
      if (!w.name || SKIP_WIDGETS.has(w.name) || w.type === "button") continue;
      if (w.value === null || ["string", "number", "boolean"].includes(typeof w.value)) out[w.name] = w.value;
    }
    return out;
  }
  function applyPreset(slot, pr) {
    const keepTab = state.tab;
    state = { ...DEFAULT_STATE, ...(pr.state || {}), tab: keepTab, current_preset: slot };
    saveState();
    for (const [k, v] of Object.entries(pr.widgets || {})) {
      const w = W(k);
      if (w) { w.value = v; w.callback?.(v); }
    }
    node.onWidgetChanged?.();          // lets the media panel redraw its settings
    render();
    toast(`Preset ${slot} “${pr.name}” loaded.`, "success");
  }
  async function savePresetTo(slot) {
    const prev = (boot.presets_saved || {})[slot];
    const name = window.prompt(`Name for preset ${slot}`, prev?.name || `Preset ${slot}`);
    if (name === null) return;
    const r = await postJSON("/onyx/studio/preset/save", { slot, name, state: { ...state, current_preset: slot }, widgets: widgetSnapshot() });
    if (!r.ok) return toast(r.error, "error");
    boot.presets_saved = r.presets; presetSaveMode = false; state.current_preset = slot; saveState();
    renderPresets(); toast(`Saved to preset ${slot}.`, "success");
  }
  function renderPresets() {
    const saved = boot.presets_saved || {};
    const chips = [];
    for (let i = 1; i <= 10; i++) {
      const pr = saved[String(i)];
      const cls = "ost-ps" + (pr ? " full" : "") + (String(state.current_preset) === String(i) ? " cur" : "") + (presetSaveMode ? " saving" : "");
      chips.push(h("button", {
        class: cls, title: pr ? `${pr.name}\nclick: load · right-click: delete` : "empty slot",
        onclick: () => presetSaveMode ? savePresetTo(i) : pr ? applyPreset(i, pr) : toast("Empty slot: click 💾 Save, then this slot.", "info"),
        oncontextmenu: async (e) => {
          e.preventDefault(); e.stopPropagation();
          if (!pr || !confirm(`Delete preset ${i} “${pr.name}”?`)) return;
          const r = await postJSON("/onyx/studio/preset/delete", { slot: i });
          boot.presets_saved = r.presets; renderPresets();
        } }, pr ? `${i} · ${pr.name}` : String(i)));
    }
    fill(presetBar, h("span", { class: "lbl" }, "Presets"), ...chips,
      h("button", { class: "ost-btn" + (presetSaveMode ? " pri" : " ghost"), style: "margin-left:auto;padding:4px 10px",
        title: "Saves every setting of the node: mode, skill, preset, character, extra LoRAs, LLM, format, post look / motion, media settings (not the media list).",
        onclick: () => { presetSaveMode = !presetSaveMode; renderPresets(); } }, presetSaveMode ? "Click a slot…" : "💾 Save"));
  }

  root.append(head, tabsBar, presetBar, body, foot);

  // ── data ───────────────────────────────────────────────────────────────────
  async function loadBoot() {
    try {
      const r = await getJSON("/onyx/studio/bootstrap");
      if (r.ok) boot = r;
    } catch (e) { console.warn("[Onyx Studio] bootstrap failed", e); }
    if (state.provider === "ollama") refreshOllama(true);
    render();
  }
  async function refreshOllama(silent) {
    const r = await getJSON("/onyx/studio/ollama_models?url=" + encodeURIComponent(state.ollama_url || ""));
    if (r.ok) { ollamaModels = r.models; if (!state.model && ollamaModels.length) set({ model: ollamaModels[0] }, false); }
    else if (!silent) toast("Ollama: " + r.error, "warn");
    render();
  }
  async function loadGallery() {
    try { const r = await getJSON("/onyx/prompt_library/list"); gallery = r.entries || []; } catch { gallery = []; }
    render();
  }
  const batch = () => { try { return JSON.parse(W("batch_data")?.value || "{}"); } catch { return {}; } };
  const batchItems = () => { const b = batch(); return (b.order || []).map((id) => (b.images || []).find((m) => m.id === id)).filter(Boolean); };
  const curChar = () => boot.characters.find((c) => c.id === state.character_id) || null;
  const curSkill = () => boot.skills.find((s) => s.id === state.skill) || null;

  // ── preview (Generate now) ─────────────────────────────────────────────────
  async function preview() {
    if (busy) return;
    const items = batchItems();
    if (state.mode === "manual" && !state.enhance) {
      showResult({ prompt: state.manual, notes: "", mode: "manual" });
      return;
    }
    const media = state.mode === "recreate" ? (items.find((m) => m.filename === state.preview_media) || items[0])?.filename : undefined;
    if (state.mode === "recreate" && !media) { toast("Recreate: add an image to the batch (Media tab).", "warn"); return; }
    busy = true; genBtn.innerHTML = '<span class="ost-spin"></span>Writing…'; genBtn.disabled = true;
    try {
      const r = await postJSON("/onyx/studio/preview", { state, media, seed: Math.floor(Math.random() * 1e9) });
      if (!r.ok) throw new Error(r.error);
      showResult({ ...r, media });
      if (media) { mediaPrompts[media] = r.prompt; if (state.tab === "prompt" && state.mode === "recreate") renderPrompt(); }
    } catch (e) { toast(String(e.message || e), "error"); }
    finally { busy = false; genBtn.textContent = "✦ Generate now"; genBtn.disabled = false; }
  }
  function showResult(r) {
    lastResult = r;
    outBox.className = "ost-out" + (r.prompt ? "" : " empty");
    outBox.textContent = r.prompt || "(empty)";
    fill(chips, 
      h("span", { class: "ost-chip" }, LABEL[r.mode] || r.mode || ""),
      r.ratio ? h("span", { class: "ost-chip" }, "ratio " + r.ratio) : null,
      r.post && r.post !== "none" ? h("span", { class: "ost-chip post", title: "Connect post_preset to Onyx Camera Shake" }, "post: " + r.post) : null,
      r.prompt ? h("span", { class: "ost-chip", style: "color:var(--dim)" }, `${r.prompt.split(/\s+/).length} words`) : null,
    );
  }

  // ── render ─────────────────────────────────────────────────────────────────
  function render() {
    renderHeader();
    if (state.tab === "prompt") renderPrompt();
    if (state.tab === "gallery") renderGallery();
    if (state.tab === "post") renderPost();
    if (state.tab === "character") renderCharacter();
    if (state.tab === "skills") renderSkills();
    if (state.tab === "settings") renderSettings();
    if (state.tab === "history") renderHistory();
  }
  function renderHeader() {
    renderPresets();
    const c = curChar();
    fill(charPill, "🧑 ", c ? h("b", {}, c.name) : h("span", { style: "color:var(--dim)" }, "no character"),
      state.post_look && state.post_look !== "none" ? h("span", { style: "color:#ffc58a;margin-left:6px" }, "· 🎛️ " + state.post_look) : null);
    fill(modeSeg, ...Object.entries(MODE_INFO).map(([k, m]) =>
      h("button", { class: state.mode === k ? "on" : "", title: m.hint, onclick: () => set({ mode: k, tab: "prompt" }) }, `${m.icon} ${m.label}`)));
    const counts = { media: batchItems().length, skills: boot.skills.length, character: boot.characters.length };
    fill(tabsBar, ...TABS.map(([id, label]) =>
      h("div", { class: "ost-tab" + (state.tab === id ? " on" : ""), onclick: () => set({ tab: id }) }, label,
        counts[id] ? h("span", { class: "n" }, counts[id]) : null)));
    for (const [id] of TABS) pages[id].classList.toggle("on", state.tab === id);
    genBtn.style.display = state.mode === "idea" || state.mode === "recreate" || (state.mode === "manual" && state.enhance) ? "" : "none";
  }

  // ── Prompt tab ─────────────────────────────────────────────────────────────
  function llmCard() {
    const prov = h("select", { onchange: (e) => { set({ provider: e.target.value, model: "" }); if (e.target.value === "ollama") refreshOllama(true); } },
      boot.providers.map((p) => h("option", { value: p.id, selected: state.provider === p.id ? "selected" : null }, p.label)));
    let modelField;
    const list = state.provider === "ollama" ? ollamaModels : (boot.models[state.provider] || []);
    if (list.length) {
      modelField = h("select", { onchange: (e) => set({ model: e.target.value }, false) },
        list.map((m) => h("option", { value: m, selected: state.model === m ? "selected" : null }, m)));
      if (!list.includes(state.model)) set({ model: list[0] }, false);
    } else {
      modelField = h("input", { type: "text", value: state.model, placeholder: state.provider === "local_vlm" ? "default Qwen3-VL (leave empty)" : state.provider === "ollama" ? "no model found — is Ollama running?" : "model id, e.g. x-ai/grok-4.1-fast",
        onchange: (e) => set({ model: e.target.value.trim() }, false) });
    }
    const keyNeeded = ["xai", "gemini", "openrouter"].includes(state.provider);
    const keyInfo = keyNeeded ? h("span", { class: "ost-hint" }, boot.keys[state.provider] ? "🔑 key saved" : "⚠️ no key — Settings tab") : null;
    const temp = h("input", { type: "number", value: state.temperature, min: 0, max: 2, step: 0.05, onchange: (e) => set({ temperature: Number(e.target.value) }, false) });
    const maxT = h("input", { type: "number", value: state.max_tokens, min: 256, max: 16384, step: 256, onchange: (e) => set({ max_tokens: Number(e.target.value) }, false) });
    return h("div", { class: "ost-card" }, h("h4", {}, "Writer model"),
      h("div", { class: "ost-row" },
        h("div", { class: "ost-col" }, h("label", {}, "Provider"), prov),
        h("div", { class: "ost-col", style: "flex:2" }, h("label", {}, "Model ", state.provider === "ollama" ? h("a", { href: "#", style: "color:var(--ac)", onclick: (e) => { e.preventDefault(); refreshOllama(false); } }, "↻") : null), modelField),
        h("div", { class: "ost-col", style: "max-width:90px" }, h("label", {}, "Temperature"), temp),
        h("div", { class: "ost-col", style: "max-width:100px" }, h("label", {}, "Max tokens"), maxT)),
      keyInfo ? h("div", { class: "ost-row" }, keyInfo) : null);
  }
  function skillSelect(onPick) {
    return h("select", { onchange: (e) => onPick(e.target.value) },
      h("option", { value: "" }, "— choose a skill —"),
      ["imported", "worker"].map((src) => {
        const list = boot.skills.filter((sk) => sk.source === src);
        if (!list.length) return null;
        return h("optgroup", { label: src === "imported" ? "Imported skills" : "Onyx skills (Worker)" },
          list.map((sk) => h("option", { value: sk.id, selected: state.skill === sk.id ? "selected" : null }, sk.name)));
      }));
  }
  function skillCard(title = "Skill") {
    const sk = curSkill();
    return h("div", { class: "ost-card" }, h("h4", {}, title),
      h("div", { class: "ost-row" },
        h("div", { class: "ost-col", style: "flex:3" }, h("label", {}, "Prompting style"), skillSelect((v) => set({ skill: v })))),
      h("div", { class: "ost-row" },
        toggle("Double-check pass", state.double_check, (v) => set({ double_check: v }, false), "A second call checks the prompt against the skill's rules and returns the fixed version (the two-chat method)."),
        sk?.locked_sentence ? h("span", { class: "ost-hint" }, "rulebook with locked sentences") : null),
      !boot.skills.length ? h("div", { class: "ost-hint" }, "No skill yet: import one in the Skills tab (e.g. CANDID-SNAP-MASTER.txt).") : null);
  }
  function triggerHint() {
    const c = curChar();
    const span = h("span", { class: "ost-hint" }, c ? "trigger: …" : "no character — pick one in the Character tab");
    if (c) (async () => {
      let t = (c.trigger_line || "").trim(), src = "typed";
      if (!t) { for (const r of c.loras || []) { if (r.on !== false) { t = await loraTrigger(r.lora); if (t) { src = "from LoRA"; break; } } } }
      span.textContent = t ? `trigger: ${t} (${src})` : "⚠️ no trigger: the LoRA has none in its metadata — type it in the Character tab";
    })();
    return span;
  }
  function renderPrompt() {
    const p = pages.prompt;
    const m = state.mode === "gallery" ? "manual" : state.mode;
    const info = h("div", { class: "ost-hint", style: "margin-bottom:8px" }, MODE_INFO[m].icon, " ", MODE_INFO[m].hint);
    const extra = h("input", { type: "text", value: state.extra, placeholder: m === "recreate" ? 'Optional: "curvy", "night instead", "same outfit"…' : m === "manual" ? 'Optional note for the skill: "keep it at night"…' : 'Optional: "outdoor only", "flash"…',
      onchange: (e) => set({ extra: e.target.value }, false) });
    if (m === "manual") {
      const ta = h("textarea", { style: "min-height:240px", placeholder: "{trigger}\nWaist-up shot, A candid phone snapshot of …", value: state.manual,
        oninput: (e) => { state.manual = e.target.value; saveState(); } });
      fill(p, info, h("div", { class: "ost-card" }, h("h4", {}, "Prompt"), ta,
          h("div", { class: "ost-row", style: "margin-top:8px" },
            toggle("Force trigger line", state.force_trigger, (v) => set({ force_trigger: v }, false)),
            triggerHint(),
            h("span", { class: "ost-hint", style: "margin-left:auto" }, `${(state.manual || "").split(/\s+/).filter(Boolean).length} words`),
            h("button", { class: "ost-btn ghost", onclick: () => set({ tab: "gallery" }) }, "🖼️ From gallery…"))),
        h("div", { class: "ost-card" }, h("h4", {}, "Improve with a skill"),
          h("div", { class: "ost-row" },
            toggle("Improve my prompt with a skill", state.enhance, (v) => set({ enhance: v }), "The skill rewrites YOUR prompt to follow its rules (its fix / check path): same scene, outfit, pose and light."),
            state.enhance ? h("div", { class: "ost-col", style: "flex:2" }, skillSelect((v) => set({ skill: v }))) : null),
          state.enhance ? h("div", { class: "ost-row" }, extra) : h("div", { class: "ost-hint" }, "Off: the prompt is used exactly as typed.")),
        state.enhance ? llmCard() : null, formatCard(), postQuickCard());
      return;
    }
    if (m === "idea") {
      const idea = h("textarea", { style: "min-height:110px", placeholder: "gas station at night eating chips", value: state.idea,
        oninput: (e) => { state.idea = e.target.value; saveState(); } });
      fill(p, info, h("div", { class: "ost-card" }, h("h4", {}, "Idea"), idea, h("div", { class: "ost-row", style: "margin-top:8px" }, extra),
          h("div", { class: "ost-row" }, triggerHint())),
        skillCard(), llmCard(), formatCard(), postQuickCard());
      return;
    }
    // recreate
    const items = batchItems();
    const pick = h("select", { onchange: (e) => set({ preview_media: e.target.value }, false) },
      items.map((it) => h("option", { value: it.filename, selected: state.preview_media === it.filename ? "selected" : null }, (it.is_video ? "🎬 " : "🖼️ ") + (it.name || it.filename))));
    const presetSel = h("select", { onchange: (e) => set({ preset: e.target.value }) },
      h("option", { value: "" }, "— choose a Grok preset —"),
      boot.presets.map((pr) => h("option", { value: pr.slot, selected: String(state.preset) === pr.slot ? "selected" : null },
        `${pr.slot} · ${pr.name}${pr.skill ? "  ✦ " + pr.skill : ""}`)));
    const pr = boot.presets.find((x) => x.slot === String(state.preset));
    fill(p, info,
      h("div", { class: "ost-card" }, h("h4", {}, "Source images"),
        items.length
          ? h("div", { class: "ost-row" }, h("span", { class: "ost-hint" }, `${items.length} item(s) in the batch — each run rewrites the next one. Queue All in the Media tab runs them all.`))
          : h("div", { class: "ost-row" }, h("span", { class: "ost-hint" }, "The batch is empty."), h("button", { class: "ost-btn", onclick: () => set({ tab: "media" }) }, "Add media →")),
        items.length ? sideBySide(items, pick) : null),
      h("div", { class: "ost-card" }, h("h4", {}, "Grok preset (image → prompt)"),
        h("div", { class: "ost-row" }, presetSel),
        pr ? h("div", { class: "ost-row" },
          pr.embedded_trigger ? h("span", { class: "ost-hint" }, `preset trigger “${pr.embedded_trigger}” → replaced by the character's`) : null,
          pr.skill ? h("span", { class: "ost-hint" }, `✦ uses the “${pr.skill}” skill as rulebook (if imported)`) : null) : null,
        h("div", { class: "ost-row" }, triggerHint()),
        h("div", { class: "ost-row" }, extra),
        h("div", { class: "ost-row" }, toggle("Double-check pass", state.double_check, (v) => set({ double_check: v }, false), "Only for presets linked to a skill."))),
      llmCard(), formatCard(), postQuickCard());
  }

  // Format: ratio (also sent to the skill) + megapixels -> width / height outputs,
  // computed by Onyx Resolution (MP)'s own engine (same live preview route).
  function formatCard() {
    const res = (state.res ||= { mp: 1.0, mult: 32, follow_skill: false });
    const aspects = boot.res_aspects?.length ? boot.res_aspects : RATIOS;
    const preview = h("span", { class: "ost-hint" }, "…");
    const refresh = async () => {
      const ar = state.ratio;
      if (ar === "from image (input)") { preview.textContent = "from the batch image — computed at run time"; return; }
      try {
        const r = await getJSON(`/onyx/resolution_mp/preview?mp=${encodeURIComponent(res.mp)}&ar=${encodeURIComponent(ar)}&mult=${encodeURIComponent(res.mult)}`);
        preview.textContent = r.success ? "📐 " + r.text : (r.error || "?");
      } catch { preview.textContent = "preview unavailable"; }
    };
    refresh();
    return h("div", { class: "ost-card" }, h("h4", {}, "Format (latent · width / height outputs)"),
      h("div", { class: "ost-row" },
        h("div", { class: "ost-col" }, h("label", {}, "Aspect ratio"),
          h("select", { onchange: (e) => { set({ ratio: e.target.value }, false); refresh(); } },
            aspects.map((a) => h("option", { value: a, selected: state.ratio === a ? "selected" : null }, a)))),
        h("div", { class: "ost-col", style: "max-width:120px" }, h("label", {}, "Megapixels"),
          h("input", { type: "number", min: 0.05, max: 20, step: 0.01, value: res.mp, onchange: (e) => { res.mp = Number(e.target.value) || 1; saveState(); refresh(); } })),
        h("div", { class: "ost-col", style: "max-width:100px" }, h("label", {}, "Multiple of"),
          h("input", { type: "number", min: 1, max: 128, step: 1, value: res.mult, onchange: (e) => { res.mult = parseInt(e.target.value, 10) || 32; saveState(); refresh(); } })),
        h("div", { class: "ost-col", style: "max-width:90px" }, h("label", {}, "Batch size"),
          h("input", { type: "number", min: 1, max: 64, step: 1, value: res.batch || 1, title: "Images per run: the size of the latent output.", onchange: (e) => { res.batch = Math.max(1, parseInt(e.target.value, 10) || 1); saveState(); } }))),
      h("div", { class: "ost-row" }, preview,
        toggle("Follow the skill's ratio", res.follow_skill, (v) => { res.follow_skill = v; saveState(); }, "When the skill writes a ratio in its NOTES line, the width / height follow it.")));
  }

  // Look + motion chosen right where the prompt is written (same values as the Post tab).
  function postQuickCard() {
    const looks = boot.post_looks.length ? boot.post_looks.map((l) => l.id) : ["none"];
    const motions = boot.post_motions.length ? boot.post_motions : ["none"];
    const motionLabel = (m) => m === "auto" ? "auto (the skill decides)" : m;
    return h("div", { class: "ost-card" }, h("h4", {}, "Post look & motion"),
      h("div", { class: "ost-row" },
        h("div", { class: "ost-col" }, h("label", {}, "Look"),
          h("select", { onchange: (e) => set({ post_look: e.target.value }) },
            looks.map((l) => h("option", { value: l, selected: state.post_look === l ? "selected" : null }, l)))),
        h("div", { class: "ost-col" }, h("label", {}, "Motion (Camera Shake)"),
          h("select", { onchange: (e) => set({ post_motion: e.target.value }) },
            motions.map((m) => h("option", { value: m, selected: state.post_motion === m ? "selected" : null }, motionLabel(m)))))),
      h("div", { class: "ost-hint" }, "Pick a motion yourself, or auto to use the one the skill writes in its NOTES. Sent through the post and post_preset outputs."));
  }

  // Recreate: the source image and the prompt written for it, side by side.
  function sideBySide(items, pick) {
    const idx = Math.max(0, items.findIndex((it) => it.filename === state.preview_media));
    const it = items[idx] || items[0];
    if (!state.preview_media && it) { state.preview_media = it.filename; saveState(); }
    const step = (d) => { const n = items[(idx + d + items.length) % items.length]; set({ preview_media: n.filename }); };
    const text = mediaPrompts[it.filename];
    return h("div", {},
      h("div", { class: "ost-row" },
        h("button", { class: "ost-btn ghost", onclick: () => step(-1), title: "Previous item" }, "◀"),
        h("div", { class: "ost-col", style: "flex:1" }, pick),
        h("button", { class: "ost-btn ghost", onclick: () => step(1), title: "Next item" }, "▶"),
        h("span", { class: "ost-hint" }, `${idx + 1} / ${items.length}`)),
      h("div", { class: "ost-sbs" },
        h("div", { class: "img" }, h("img", { src: api.apiURL("/onyx/view/" + encodeURIComponent(it.thumbnail || it.filename)), alt: it.name || it.filename })),
        text ? h("div", { class: "txt" }, text)
             : h("div", { class: "txt empty" }, "No prompt written for this image yet.\n✦ Generate now writes it here, and every queued run fills it in live.")));
  }

  // ── Gallery tab ────────────────────────────────────────────────────────────
  function renderGallery() {
    const p = pages.gallery;
    if (!gallery.length && !galleryLoaded) { galleryLoaded = true; loadGallery(); }
    const q = (state.gallery_q || "").toLowerCase();
    const items = gallery.filter((e) => !q || (e.name + " " + e.prompt + " " + (e.tags || []).join(" ")).toLowerCase().includes(q));
    const search = h("input", { type: "text", style: "flex:1;width:auto", placeholder: "Search the Prompt Gallery…", value: state.gallery_q || "",
      oninput: (e) => { state.gallery_q = e.target.value; renderGallery(); setTimeout(() => { const i = p.querySelector("input"); i?.focus(); i?.setSelectionRange(i.value.length, i.value.length); }, 0); } });
    const sel = gallery.find((e) => e.id === state.gallery_sel);
    const send = (how) => {
      if (!sel) return;
      const cur = state.manual || "";
      const text = how === "append" && cur.trim() ? cur.replace(/\s+$/, "") + "\n" + sel.prompt : sel.prompt;
      set({ mode: "manual", manual: text, tab: "prompt" });
      toast(how === "append" ? "Gallery prompt appended." : "Gallery prompt sent to the prompt.", "success");
    };
    fill(p,
      h("div", { class: "ost-row", style: "flex-wrap:nowrap" }, search, h("button", { class: "ost-btn ghost", onclick: () => { galleryLoaded = false; gallery = []; renderGallery(); } }, "↻")),
      sel ? h("div", { class: "ost-card" }, h("h4", {}, sel.name), h("div", { class: "ost-pre" }, sel.prompt),
        h("div", { class: "ost-row", style: "margin-top:8px" },
          h("button", { class: "ost-btn pri", onclick: () => send("replace") }, "⤵ Use as prompt"),
          h("button", { class: "ost-btn", onclick: () => send("append") }, "＋ Append to my prompt"),
          h("button", { class: "ost-btn ghost", onclick: () => navigator.clipboard?.writeText(sel.prompt) }, "⧉ Copy"))) : null,
      items.length ? h("div", { class: "ost-grid" }, items.map((e) => h("div", {
          class: "ost-gal" + (state.gallery_sel === e.id ? " on" : ""), title: e.prompt,
          onclick: () => { state.gallery_sel = e.id; saveState(); renderGallery(); },
          ondblclick: () => { state.gallery_sel = e.id; send("replace"); } },
          e.thumb ? h("img", { src: api.apiURL("/onyx/prompt_library/thumb?id=" + e.id), loading: "lazy" }) : h("div", { class: "ph" }, "✦"),
          h("div", { class: "t" }, e.name))))
        : h("div", { class: "ost-empty" }, gallery.length ? "No match." : "The gallery is empty — save prompts with ★ Save or the Onyx Prompt Saver node."),
      h("div", { class: "ost-hint", style: "margin-top:8px" }, "Click to preview, double-click to use it directly."));
  }

  // ── Post tab ───────────────────────────────────────────────────────────────
  function renderPost() {
    const p = pages.post;
    const looks = boot.post_looks.length ? boot.post_looks : [{ id: "none", info: "" }];
    const motions = boot.post_motions.length ? boot.post_motions : ["none"];
    const motionInfo = { none: "No camera shake.", auto: "The preset the skill wrote in its NOTES line.", soft: "Slightly soft / hazy, no smear.",
      shake_med: "Diagonal hand-shake, eyes still readable.", shake_med_h: "Side-to-side smear + faint double image.",
      shake_med_v: "Up-and-down smear.", shake_med_rot: "Smear + tilted frame.", shake_heavy: "Strong smear, flash-in-the-dark style." };
    fill(p,
      h("div", { class: "ost-hint", style: "margin-bottom:8px" }, "🎛️ Wire the node's ", h("b", {}, "post"), " output to the onyx_post input of your post-processing nodes. A look other than none overrides their settings; none = each node keeps its own."),
      h("div", { class: "ost-card" }, h("h4", {}, "Look"),
        h("div", { class: "ost-grid", style: "grid-template-columns:repeat(auto-fill,minmax(170px,1fr))" }, looks.map((l) => h("div", {
          class: "ost-item" + (state.post_look === l.id ? " on" : ""), style: "flex-direction:column;align-items:flex-start;gap:3px",
          onclick: () => set({ post_look: l.id }) }, h("div", { class: "ttl" }, l.id), h("div", { class: "sub", style: "white-space:normal" }, l.info || ""))))),
      h("div", { class: "ost-card" }, h("h4", {}, "Motion (Onyx Camera Shake)"),
        h("div", { class: "ost-grid", style: "grid-template-columns:repeat(auto-fill,minmax(170px,1fr))" }, motions.map((m) => h("div", {
          class: "ost-item" + (state.post_motion === m ? " on" : ""), style: "flex-direction:column;align-items:flex-start;gap:3px",
          onclick: () => set({ post_motion: m }) }, h("div", { class: "ttl" }, m), h("div", { class: "sub", style: "white-space:normal" }, motionInfo[m] || ""))))));
  }

  // ── Character tab ──────────────────────────────────────────────────────────
  let draft = null;
  function loraRowsEditor(rows, onChange, autoTrigger) {
    const box = h("div", {});
    const draw = () => fill(box,
      h("div", { class: "ost-lora", style: "color:var(--dim);font-size:11px" }, h("span"), h("span", {}, "LoRA file"), h("span", {}, "model"), h("span", {}, "clip"), h("span")),
      ...rows.map((r, i) => h("div", { class: "ost-lora" },
        h("input", { type: "checkbox", checked: r.on !== false, onchange: (e) => { r.on = e.target.checked; onChange(); } }),
        h("select", { onchange: async (e) => { r.lora = e.target.value; onChange(); if (autoTrigger) autoTrigger(r.lora); } },
          [r.lora, ...boot.loras.filter((x) => x !== r.lora)].filter(Boolean).map((f) => h("option", { value: f, selected: f === r.lora ? "selected" : null }, f))),
        h("input", { type: "number", step: 0.05, value: r.sm ?? 1, onchange: (e) => { r.sm = Number(e.target.value); onChange(); } }),
        h("input", { type: "number", step: 0.05, value: r.sc ?? r.sm ?? 1, onchange: (e) => { r.sc = Number(e.target.value); onChange(); } }),
        h("button", { class: "ost-btn ghost bad", style: "padding:4px 6px", onclick: () => { rows.splice(i, 1); onChange(); draw(); } }, "✕"))),
      h("button", { class: "ost-btn", onclick: () => { rows.push({ lora: boot.loras[0] || "", sm: 1, sc: 1, on: true }); onChange(); draw(); if (autoTrigger) autoTrigger(boot.loras[0]); } }, "+ LoRA"));
    draw();
    return box;
  }
  function renderCharacter() {
    const p = pages.character;
    const list = h("div", { class: "ost-list" },
      boot.characters.map((c) => h("div", { class: "ost-item" + (state.character_id === c.id ? " on" : ""),
        onclick: () => { draft = null; set({ character_id: c.id }); } },
        h("div", { class: "grow" }, h("div", { class: "ttl" }, c.name), h("div", { class: "sub" }, c.trigger_line || "trigger from LoRA")),
        h("span", { class: "ost-badge" }, `${(c.loras || []).length} LoRA`))),
      h("button", { class: "ost-btn", onclick: () => { draft = { name: "New character", trigger_line: "", loras: [] }; renderCharacter(); } }, "+ New character"),
      state.character_id ? h("button", { class: "ost-btn ghost", onclick: () => set({ character_id: "" }) }, "Use no character") : null);
    const c = draft || (curChar() ? JSON.parse(JSON.stringify(curChar())) : null);
    if (!c) {
      fill(p, h("div", { class: "ost-split" }, list, h("div", { class: "ost-empty" }, "A character = its LoRA + its trigger word. Create one: the LoRA is applied to MODEL / CLIP and the trigger starts every prompt.")));
      return;
    }
    draft = c;
    const trigInput = h("input", { type: "text", value: c.trigger_line || "", placeholder: "empty = read from the LoRA", oninput: (e) => { c.trigger_line = e.target.value; } });
    const trigInfo = h("span", { class: "ost-hint" });
    const autoTrigger = async (name) => {
      const t = await loraTrigger(name);
      trigInfo.textContent = t ? `LoRA trigger: ${t}` : "this LoRA has no trigger in its metadata — type it";
      if (t && !c.trigger_line) { trigInput.placeholder = `${t} (from LoRA)`; }
    };
    const firstOn = (c.loras || []).find((r) => r.on !== false);
    if (firstOn) autoTrigger(firstOn.lora);
    const editor = h("div", {},
      h("div", { class: "ost-card" }, h("h4", {}, "Character"),
        h("div", { class: "ost-row" },
          h("div", { class: "ost-col" }, h("label", {}, "Name"), h("input", { type: "text", value: c.name || "", oninput: (e) => { c.name = e.target.value; } })),
          h("div", { class: "ost-col", style: "flex:2" }, h("label", {}, "Trigger word"), trigInput)),
        h("div", { class: "ost-row" }, trigInfo,
          h("button", { class: "ost-btn ghost", style: "margin-left:auto", onclick: async () => { const r = (c.loras || []).find((x) => x.on !== false); const t = r ? await loraTrigger(r.lora) : ""; if (t) { c.trigger_line = t; trigInput.value = t; } else toast("No trigger in this LoRA's metadata.", "warn"); } }, "↧ Fill from LoRA"))),
      h("div", { class: "ost-card" }, h("h4", {}, "Character LoRA(s) — applied to MODEL / CLIP"), loraRowsEditor((c.loras ||= []), () => {}, autoTrigger)),
      h("div", { class: "ost-row" },
        h("button", { class: "ost-btn pri", onclick: async () => {
          const r = await postJSON("/onyx/studio/character/save", { character: c });
          if (!r.ok) return toast(r.error, "error");
          boot.characters = boot.characters.filter((x) => x.id !== r.character.id).concat(r.character);
          draft = null; set({ character_id: r.character.id }); toast("Character saved.", "success");
        } }, "Save character"),
        c.id ? h("button", { class: "ost-btn ghost bad", onclick: async () => {
          if (!confirm(`Delete ${c.name}?`)) return;
          await postJSON("/onyx/studio/character/delete", { id: c.id });
          boot.characters = boot.characters.filter((x) => x.id !== c.id); draft = null; set({ character_id: "" });
        } }, "Delete") : null),
      h("div", { class: "ost-card", style: "margin-top:10px" }, h("h4", {}, "Extra LoRAs — this workflow only"),
        loraRowsEditor((state.extra_loras ||= []), () => saveState()),
        h("div", { class: "ost-hint", style: "margin-top:6px" }, "Style / realism LoRAs for this workflow. Saved in the workflow, not in the character.")));
    fill(p, h("div", { class: "ost-split" }, list, editor));
  }

  // ── Skills tab ─────────────────────────────────────────────────────────────
  let openSkill = null, openText = "";
  async function importFiles(files) {
    for (const f of files) {
      if (!/\.(txt|md)$/i.test(f.name)) { toast(`${f.name}: only .txt / .md`, "warn"); continue; }
      const text = await f.text();
      const r = await postJSON("/onyx/studio/skill/import", { name: f.name, text });
      if (!r.ok) toast(`${f.name}: ${r.error}`, "error");
      else toast(`Imported ${r.skill.name}${r.skill.locked_sentence ? " (locked sentence found)" : ""}.`, "success");
    }
    await loadBoot();
  }
  function renderSkills() {
    const p = pages.skills;
    const input = h("input", { type: "file", accept: ".txt,.md", multiple: true, style: "display:none", onchange: (e) => importFiles([...e.target.files]) });
    const drop = h("div", { class: "ost-drop", onclick: () => input.click() }, "📥 Drop .txt / .md skill files here, or click to import (e.g. CANDID-SNAP-MASTER.txt)");
    drop.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.add("hot"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.remove("hot"); importFiles([...e.dataTransfer.files]); });
    const list = h("div", { class: "ost-list" }, boot.skills.map((s) => h("div", {
      class: "ost-item" + (openSkill === s.id ? " on" : ""),
      onclick: async () => { openSkill = s.id; const r = await getJSON("/onyx/studio/skill/text?id=" + encodeURIComponent(s.id)); openText = r.ok ? r.text : r.error; renderSkills(); } },
      h("span", { class: "ost-badge " + s.source }, s.source),
      h("div", { class: "grow" }, h("div", { class: "ttl" }, s.name), h("div", { class: "sub" }, s.kind === "instruction" ? "instruction + image" : "system prompt", s.locked_sentence ? " · locked: " + s.locked_sentence.slice(0, 60) + "…" : "")),
      h("button", { class: "ost-btn ghost", onclick: (e) => { e.stopPropagation(); set({ skill: s.id, tab: "prompt" }); } }, "Use"))));
    const sk = boot.skills.find((s) => s.id === openSkill);
    let detail = null;
    if (sk) {
      const name = h("input", { type: "text", value: sk.name });
      const locked = h("textarea", { style: "min-height:50px", value: sk.locked_sentence || "", placeholder: "Sentence of the skill that the character's hair lock replaces" });
      detail = h("div", { class: "ost-card", style: "margin-top:10px" }, h("h4", {}, sk.name),
        sk.source === "imported" ? h("div", {},
          h("div", { class: "ost-row" }, h("div", { class: "ost-col" }, h("label", {}, "Name"), name)),
          h("div", { class: "ost-row" }, h("div", { class: "ost-col" }, h("label", {}, "Locked sentence (replaced by the character's hair lock)"), locked)),
          h("div", { class: "ost-row" },
            h("button", { class: "ost-btn pri", onclick: async () => { await postJSON("/onyx/studio/skill/meta", { id: sk.id, name: name.value, locked_sentence: locked.value }); await loadBoot(); toast("Skill updated.", "success"); } }, "Save"),
            h("button", { class: "ost-btn ghost bad", onclick: async () => { if (!confirm("Delete this skill file?")) return; await postJSON("/onyx/studio/skill/delete", { id: sk.id }); openSkill = null; await loadBoot(); } }, "Delete"))) : null,
        h("div", { class: "ost-pre" }, openText));
    }
    fill(p, input, drop, h("div", { style: "height:10px" }), list.childNodes.length ? list : h("div", { class: "ost-empty" }, "No skill yet."), detail,
      h("div", { class: "ost-hint", style: "margin-top:10px" }, "Imported skills are stored in ", h("code", {}, (boot.studio_dir || "") + "/skills"), ", on this computer only. Skills improve a manual prompt, write from an idea, or serve as the rulebook of a Grok preset (Recreate)."));
  }

  // ── Settings tab ───────────────────────────────────────────────────────────
  function renderSettings() {
    const p = pages.settings;
    const keyRow = (prov, label) => {
      const inp = h("input", { type: "password", placeholder: boot.keys[prov] ? "•••••••• saved — type to replace" : "paste the key" });
      return h("div", { class: "ost-row" }, h("div", { class: "ost-col", style: "flex:2" }, h("label", {}, label, boot.keys[prov] ? " ✓" : ""), inp),
        h("button", { class: "ost-btn", onclick: async () => { const r = await postJSON("/onyx/studio/key", { provider: prov, key: inp.value }); if (r.ok) { boot.keys = r.keys; inp.value = ""; toast(`${label} key ${r.keys[prov] ? "saved" : "removed"}.`, "success"); renderSettings(); } } }, "Save"),
        boot.keys[prov] ? h("button", { class: "ost-btn ghost bad", onclick: async () => { const r = await postJSON("/onyx/studio/key", { provider: prov, key: "" }); boot.keys = r.keys; renderSettings(); } }, "Remove") : null);
    };
    const url = h("input", { type: "text", value: state.ollama_url, onchange: (e) => set({ ollama_url: e.target.value.trim() }, false) });
    fill(p, 
      h("div", { class: "ost-card" }, h("h4", {}, "API keys — saved on this computer, never in the workflow"),
        keyRow("xai", "Grok (xAI)"), keyRow("gemini", "Gemini"), keyRow("openrouter", "OpenRouter"),
        h("div", { class: "ost-hint" }, "A key connected to the node's input wins over the saved one.")),
      h("div", { class: "ost-card" }, h("h4", {}, "Ollama (local)"),
        h("div", { class: "ost-row" }, h("div", { class: "ost-col", style: "flex:2" }, h("label", {}, "Server URL"), url),
          h("button", { class: "ost-btn", onclick: () => refreshOllama(false) }, "Test & list models")),
        h("div", { class: "ost-hint" }, ollamaModels.length ? `${ollamaModels.length} model(s): ${ollamaModels.slice(0, 8).join(", ")}${ollamaModels.length > 8 ? "…" : ""}` : "No model listed. For recreate mode use a vision model: qwen3-vl, qwen2.5vl, gemma3, llava.")),
      h("div", { class: "ost-card" }, h("h4", {}, "Storage"),
        h("div", { class: "ost-hint" }, "Skills, characters, keys and history: ", h("code", {}, boot.studio_dir || "?"))));
  }

  // ── History tab ────────────────────────────────────────────────────────────
  async function renderHistory() {
    const p = pages.history;
    const r = await getJSON("/onyx/studio/history").catch(() => ({ history: [] }));
    const hist = r.history || [];
    fill(p, hist.length ? h("div", { class: "ost-list" }, hist.map((e) => h("div", { class: "ost-item", style: "align-items:flex-start", onclick: () => showResult(e) },
      h("span", { class: "ost-badge" }, (MODE_INFO[e.mode]?.icon || "") + " " + (e.preview ? "preview" : "run")),
      h("div", { class: "grow" }, h("div", { class: "sub" }, `${fmtAge(e.ts)} · ${e.preset ? "preset " + e.preset : (e.skill || e.mode)}${e.notes ? " · " + e.notes : ""}`),
        h("div", { style: "white-space:pre-wrap;font-size:12px;max-height:54px;overflow:hidden" }, e.prompt)),
      h("button", { class: "ost-btn ghost", onclick: (ev) => { ev.stopPropagation(); set({ mode: "manual", manual: e.prompt, tab: "prompt" }); } }, "Use")))) :
      h("div", { class: "ost-empty" }, "No prompt generated yet."));
  }

  // ── mount ──────────────────────────────────────────────────────────────────
  const batchPanel = mountBatchPanel(node, pages.media);   // the Batch Loader's own panel, inside the Media tab

  // Drop anywhere on the Studio: .txt / .md -> skills, images / videos / links ->
  // the media batch (the Media tab and the Skills drop zone handle their own
  // drops first and stop them there). Without this, a drop outside the Media
  // tab fell through to ComfyUI, which made a Load Image node of it.
  const isSkillFile = (f) => /\.(txt|md)$/i.test(f.name);
  root.addEventListener("dragover", (e) => {
    if (!batchPanel?.isExternalDrag?.(e)) return;
    e.preventDefault(); e.stopPropagation();
    root.style.outline = "2px dashed #9fb4ff"; root.style.outlineOffset = "-4px";
    try { e.dataTransfer.dropEffect = "copy"; } catch (_) {}
  });
  root.addEventListener("dragleave", (e) => { if (!root.contains(e.relatedTarget)) root.style.outline = ""; });
  root.addEventListener("drop", (e) => {
    root.style.outline = "";
    if (e.defaultPrevented || !batchPanel?.isExternalDrag?.(e)) return;
    e.preventDefault(); e.stopPropagation();
    const files = [...(e.dataTransfer.files || [])];
    if (files.length && files.every(isSkillFile)) { importFiles(files); set({ tab: "skills" }); return; }
    if (batchPanel.acceptDrop(e.dataTransfer)) {
      toast("Added to the media batch.", "success");
      if (state.mode !== "recreate" || state.tab !== "prompt") set({ tab: "media" });
    }
  });
  hideWidget(W("studio_state"));
  setTimeout(() => hideWidget(W("studio_state")), 300);
  node.addDOMWidget("studio_panel", "ONYX_STUDIO", root, {
    serialize: false, getValue: () => undefined, setValue: () => {}, getMinHeight: () => PANEL_MIN_H,
  });
  node.size = [...NODE_SIZE];

  const origConfigure = node.onConfigure?.bind(node);
  node.onConfigure = function (info) { origConfigure?.(info); readState(); render(); };

  const onResult = ({ detail }) => {
    if (String(detail?.node) !== String(node.id)) return;
    const media = runMedia;
    if (media) mediaPrompts[media] = detail.prompt;
    showResult({ ...detail, media });
    if (media && state.mode === "recreate" && state.tab === "prompt") { state.preview_media = media; saveState(); renderPrompt(); }
  };
  api.addEventListener("onyx.studio.result", onResult);
  // The batch engine announces which item it loads; remember it for the result that follows.
  const onBatchItem = ({ detail }) => {
    if (String(detail?.node_id) !== String(node.id)) return;
    const b = batch(); const id = (b.order || [])[detail.current_index ?? -1];
    const meta = (b.images || []).find((m) => m.id === id);
    runMedia = meta ? meta.filename : null;
  };
  api.addEventListener("onyx_batch_loader_update", onBatchItem);
  const origRemoved = node.onRemoved?.bind(node);
  node.onRemoved = function () {
    api.removeEventListener("onyx.studio.result", onResult);
    api.removeEventListener("onyx_batch_loader_update", onBatchItem);
    origRemoved?.();
  };

  // The media panel changes batch_data on its own: refresh counts now and then.
  let lastBatch = "";
  const poll = setInterval(() => {
    if (!node.graph) return clearInterval(poll);
    const v = W("batch_data")?.value || "";
    if (v !== lastBatch) {
      lastBatch = v;
      renderHeader();                                   // tab counters only: never steal the focus
      if (state.tab === "prompt" && state.mode === "recreate") renderPrompt();
    }
  }, 1500);

  loadBoot();
}

app.registerExtension({
  name: "Onyx.Studio",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "OnyxStudio") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      setupStudio(this);
      return r;
    };
  },
});
