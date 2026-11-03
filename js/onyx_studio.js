// Onyx Studio ✦ — the creation hub UI.
// Tabs: Prompt (manual / gallery / idea / recreate) · Media (the batch panel of
// the Batch Loader, embedded) · Character (LoRAs, trigger, locks) · Skills
// (import .txt/.md, Grok presets, Worker skills) · Settings (keys, Ollama) ·
// History. Everything the run needs is kept in the hidden `studio_state`
// widget (JSON); API keys are saved on this computer, never in the workflow.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { mountBatchPanel } from "./onyx_image_batch_loader.js";

const NODE_SIZE = [780, 920];
const PANEL_MIN_H = 600;
const RATIOS = ["9:16", "3:4", "4:5", "1:1", "2:3", "16:9"];
const MODE_INFO = {
  manual:   { icon: "✍️", label: "Manual",   hint: "Type the prompt. {trigger} = the character's trigger line." },
  gallery:  { icon: "🖼️", label: "Gallery",  hint: "Pick a prompt from the Onyx Prompt Gallery library." },
  idea:     { icon: "💡", label: "Idea",     hint: "Describe a scene: the skill + LLM writes the prompt." },
  recreate: { icon: "🎯", label: "Recreate", hint: "Each batch image is rewritten as a prompt for your character." },
};
const DEFAULT_STATE = {
  mode: "manual", manual: "", idea: "", extra: "", gallery: null,
  skill: "", character_id: "", provider: "ollama", model: "", ollama_url: "http://127.0.0.1:11434",
  temperature: 0.7, max_tokens: 2048, double_check: false, force_trigger: true, ratio: "9:16",
  extra_loras: [], tab: "prompt",
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
  let boot = { skills: [], characters: [], keys: {}, history: [], loras: [], providers: [], models: {} };
  let ollamaModels = [];
  let gallery = [];
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

  const TABS = [["prompt", "✍️ Prompt"], ["media", "🎞️ Media"], ["character", "🧑 Character"], ["skills", "📚 Skills"], ["settings", "⚙️ Settings"], ["history", "🕘 History"]];
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

  root.append(head, tabsBar, body, foot);

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
    if (state.mode === "manual" || state.mode === "gallery") {
      showResult({ prompt: state.mode === "manual" ? state.manual : (state.gallery?.text || ""), notes: "", mode: state.mode });
      return;
    }
    const items = batchItems();
    const media = state.mode === "recreate" ? (items.find((m) => m.filename === state.preview_media) || items[0])?.filename : undefined;
    if (state.mode === "recreate" && !media) { toast("Recreate: add an image to the batch (Media tab).", "warn"); return; }
    busy = true; genBtn.innerHTML = '<span class="ost-spin"></span>Writing…'; genBtn.disabled = true;
    try {
      const r = await postJSON("/onyx/studio/preview", { state, media, seed: Math.floor(Math.random() * 1e9) });
      if (!r.ok) throw new Error(r.error);
      showResult(r);
    } catch (e) { toast(String(e.message || e), "error"); }
    finally { busy = false; genBtn.textContent = "✦ Generate now"; genBtn.disabled = false; }
  }
  function showResult(r) {
    lastResult = r;
    outBox.className = "ost-out" + (r.prompt ? "" : " empty");
    outBox.textContent = r.prompt || "(empty)";
    fill(chips, 
      h("span", { class: "ost-chip" }, MODE_INFO[r.mode]?.label || r.mode || ""),
      r.ratio ? h("span", { class: "ost-chip" }, "ratio " + r.ratio) : null,
      r.post && r.post !== "none" ? h("span", { class: "ost-chip post", title: "Connect post_preset to Onyx Camera Shake" }, "post: " + r.post) : null,
      r.prompt ? h("span", { class: "ost-chip", style: "color:var(--dim)" }, `${r.prompt.split(/\s+/).length} words`) : null,
    );
  }

  // ── render ─────────────────────────────────────────────────────────────────
  function render() {
    renderHeader();
    if (state.tab === "prompt") renderPrompt();
    if (state.tab === "character") renderCharacter();
    if (state.tab === "skills") renderSkills();
    if (state.tab === "settings") renderSettings();
    if (state.tab === "history") renderHistory();
  }
  function renderHeader() {
    const c = curChar();
    fill(charPill, "🧑 ", c ? h("b", {}, c.name) : h("span", { style: "color:var(--dim)" }, "no character"));
    fill(modeSeg, ...Object.entries(MODE_INFO).map(([k, m]) =>
      h("button", { class: state.mode === k ? "on" : "", title: m.hint, onclick: () => set({ mode: k, tab: "prompt" }) }, `${m.icon} ${m.label}`)));
    const counts = { media: batchItems().length, skills: boot.skills.length, character: boot.characters.length };
    fill(tabsBar, ...TABS.map(([id, label]) =>
      h("div", { class: "ost-tab" + (state.tab === id ? " on" : ""), onclick: () => set({ tab: id }) }, label,
        counts[id] ? h("span", { class: "n" }, counts[id]) : null)));
    for (const [id] of TABS) pages[id].classList.toggle("on", state.tab === id);
    genBtn.style.display = state.mode === "idea" || state.mode === "recreate" ? "" : "none";
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
  function skillCard() {
    const sel = h("select", { onchange: (e) => set({ skill: e.target.value }) },
      h("option", { value: "" }, "— choose a skill —"),
      ["imported", "grok", "worker"].map((src) => {
        const list = boot.skills.filter((s) => s.source === src);
        if (!list.length) return null;
        return h("optgroup", { label: src === "imported" ? "Imported skills" : src === "grok" ? "Grok Prompt Generator presets" : "Onyx skills (Worker)" },
          list.map((s) => h("option", { value: s.id, selected: state.skill === s.id ? "selected" : null }, s.name)));
      }));
    const ratio = h("select", { onchange: (e) => set({ ratio: e.target.value }, false) }, RATIOS.map((r) => h("option", { value: r, selected: state.ratio === r ? "selected" : null }, r)));
    const sk = curSkill();
    return h("div", { class: "ost-card" }, h("h4", {}, "Skill"),
      h("div", { class: "ost-row" },
        h("div", { class: "ost-col", style: "flex:3" }, h("label", {}, "Prompting style"), sel),
        h("div", { class: "ost-col", style: "max-width:90px" }, h("label", {}, "Ratio"), ratio)),
      h("div", { class: "ost-row" },
        toggle("Double-check pass", state.double_check, (v) => set({ double_check: v }, false), "A second call checks the prompt against the skill's rules and returns the fixed version (the two-chat method)."),
        toggle("Force trigger line", state.force_trigger, (v) => set({ force_trigger: v }, false), "Line 1 is always the character's trigger line."),
        sk ? h("span", { class: "ost-hint" }, sk.kind === "instruction" ? "instruction sent with the image" : "used as system prompt", sk.locked_sentence ? " · locked sentence swapped" : "") : null),
      !boot.skills.length ? h("div", { class: "ost-hint" }, "No skill yet: import one in the Skills tab (e.g. CANDID-SNAP-MASTER.txt).") : null);
  }
  function renderPrompt() {
    const p = pages.prompt;
    const m = state.mode;
    const info = h("div", { class: "ost-hint", style: "margin-bottom:8px" }, MODE_INFO[m].icon, " ", MODE_INFO[m].hint);
    if (m === "manual") {
      const ta = h("textarea", { style: "min-height:260px", placeholder: "{trigger}\nWaist-up shot, A candid phone snapshot of …", value: state.manual,
        oninput: (e) => { state.manual = e.target.value; saveState(); } });
      fill(p, info, h("div", { class: "ost-card" }, h("h4", {}, "Prompt"), ta,
        h("div", { class: "ost-row", style: "margin-top:8px" }, toggle("Force trigger line", state.force_trigger, (v) => set({ force_trigger: v }, false)),
          h("span", { class: "ost-hint" }, `${(state.manual || "").split(/\s+/).filter(Boolean).length} words`))));
      return;
    }
    if (m === "gallery") {
      if (!gallery.length) loadGallery();
      const q = (state.gallery_q || "").toLowerCase();
      const items = gallery.filter((e) => !q || (e.name + " " + e.prompt + " " + (e.tags || []).join(" ")).toLowerCase().includes(q));
      const search = h("input", { type: "text", style: "flex:1;width:auto", placeholder: "Search the gallery…", value: state.gallery_q || "",
        oninput: (e) => { state.gallery_q = e.target.value; saveState(); renderPrompt(); setTimeout(() => p.querySelector("input")?.focus(), 0); } });
      const grid = h("div", { class: "ost-grid" }, items.map((e) => h("div", {
        class: "ost-gal" + (state.gallery?.id === e.id ? " on" : ""), title: e.prompt,
        onclick: () => { set({ gallery: { id: e.id, text: e.prompt } }); showResult({ prompt: e.prompt, mode: "gallery" }); } },
        e.thumb ? h("img", { src: api.apiURL("/onyx/prompt_library/thumb?id=" + e.id), loading: "lazy" }) : h("div", { class: "ph" }, "✦"),
        h("div", { class: "t" }, e.name))));
      fill(p, info, h("div", { class: "ost-row", style: "flex-wrap:nowrap" }, search, h("button", { class: "ost-btn ghost", onclick: loadGallery }, "↻")),
        items.length ? grid : h("div", { class: "ost-empty" }, gallery.length ? "No match." : "The gallery is empty — save prompts with ★ Save or the Onyx Prompt Saver node."),
        state.gallery ? h("div", { class: "ost-card", style: "margin-top:10px" }, h("h4", {}, "Selected"), h("div", { class: "ost-pre" }, state.gallery.text)) : null);
      return;
    }
    const extra = h("input", { type: "text", value: state.extra, placeholder: m === "recreate" ? 'Optional: "curvy", "night instead", "same outfit"…' : 'Optional: "10 different lights", "outdoor only"…',
      onchange: (e) => set({ extra: e.target.value }, false) });
    if (m === "idea") {
      const idea = h("textarea", { style: "min-height:110px", placeholder: "gas station at night eating chips", value: state.idea,
        oninput: (e) => { state.idea = e.target.value; saveState(); } });
      fill(p, info, h("div", { class: "ost-card" }, h("h4", {}, "Idea"), idea, h("div", { class: "ost-row", style: "margin-top:8px" }, extra)),
        skillCard(), llmCard());
      return;
    }
    // recreate
    const items = batchItems();
    const pick = h("select", { onchange: (e) => set({ preview_media: e.target.value }, false) },
      items.map((it) => h("option", { value: it.filename, selected: state.preview_media === it.filename ? "selected" : null }, (it.is_video ? "🎬 " : "🖼️ ") + (it.name || it.filename))));
    fill(p, info,
      h("div", { class: "ost-card" }, h("h4", {}, "Source images"),
        items.length
          ? h("div", { class: "ost-row" }, h("span", { class: "ost-hint" }, `${items.length} item(s) in the batch — each run rewrites the next one. Queue All in the Media tab runs them all.`))
          : h("div", { class: "ost-row" }, h("span", { class: "ost-hint" }, "The batch is empty."), h("button", { class: "ost-btn", onclick: () => set({ tab: "media" }) }, "Add media →")),
        items.length ? h("div", { class: "ost-row" }, h("div", { class: "ost-col" }, h("label", {}, "Preview with"), pick)) : null,
        h("div", { class: "ost-row" }, extra)),
      skillCard(), llmCard());
  }

  // ── Character tab ──────────────────────────────────────────────────────────
  let draft = null;
  function renderCharacter() {
    const p = pages.character;
    const list = h("div", { class: "ost-list" },
      boot.characters.map((c) => h("div", { class: "ost-item" + (state.character_id === c.id ? " on" : ""),
        onclick: () => { draft = null; set({ character_id: c.id }); } },
        h("div", { class: "grow" }, h("div", { class: "ttl" }, c.name), h("div", { class: "sub" }, c.trigger_line || "no trigger")),
        h("span", { class: "ost-badge" }, `${(c.loras || []).length} LoRA`))),
      h("button", { class: "ost-btn", onclick: () => { draft = { name: "New character", trigger_line: "", body: "", hair_lock: "", extra_rules: "", loras: [] }; renderCharacter(); } }, "+ New character"),
      state.character_id ? h("button", { class: "ost-btn ghost", onclick: () => set({ character_id: "" }) }, "Use no character") : null);
    const c = draft || (curChar() ? JSON.parse(JSON.stringify(curChar())) : null);
    if (!c) { fill(p, h("div", { class: "ost-split" }, list, h("div", { class: "ost-empty" }, "Select or create a character: its LoRAs are applied to MODEL / CLIP, and its trigger line, body and hair lock fill the skill."))); return; }
    draft = c;
    const field = (label, key, ph, area) => h("div", { class: "ost-col" }, h("label", {}, label),
      area ? h("textarea", { style: "min-height:56px", value: c[key] || "", placeholder: ph, oninput: (e) => { c[key] = e.target.value; } })
           : h("input", { type: "text", value: c[key] || "", placeholder: ph, oninput: (e) => { c[key] = e.target.value; } }));
    const loraRows = h("div", {});
    const drawLoras = () => fill(loraRows, 
      h("div", { class: "ost-lora", style: "color:var(--dim);font-size:11px" }, h("span"), h("span", {}, "LoRA file"), h("span", {}, "model"), h("span", {}, "clip"), h("span")),
      ...(c.loras || []).map((r, i) => h("div", { class: "ost-lora" },
        h("input", { type: "checkbox", checked: r.on !== false, onchange: (e) => { r.on = e.target.checked; } }),
        h("select", { onchange: (e) => { r.lora = e.target.value; } }, [r.lora, ...boot.loras.filter((x) => x !== r.lora)].filter(Boolean).map((f) => h("option", { value: f, selected: f === r.lora ? "selected" : null }, f))),
        h("input", { type: "number", step: 0.05, value: r.sm ?? 1, onchange: (e) => { r.sm = Number(e.target.value); } }),
        h("input", { type: "number", step: 0.05, value: r.sc ?? r.sm ?? 1, onchange: (e) => { r.sc = Number(e.target.value); } }),
        h("button", { class: "ost-btn ghost bad", style: "padding:4px 6px", onclick: () => { c.loras.splice(i, 1); drawLoras(); } }, "✕"))),
      h("button", { class: "ost-btn", onclick: () => { (c.loras ||= []).push({ lora: boot.loras[0] || "", sm: 1, sc: 1, on: true }); drawLoras(); } }, "+ LoRA"));
    drawLoras();
    const editor = h("div", {},
      h("div", { class: "ost-card" }, h("h4", {}, "Identity"),
        h("div", { class: "ost-row" }, field("Name", "name", "Mia"), field("Trigger line", "trigger_line", "candid_snap, mia woman")),
        h("div", { class: "ost-row" }, field("Body (fills BODY in the skill)", "body", "a slim woman with …", true)),
        h("div", { class: "ost-row" }, field("Hair lock — replaces the skill's locked sentence", "hair_lock", "Her long wavy auburn hair falls past her shoulders.", true)),
        h("div", { class: "ost-row" }, field("Extra rules (appended to the skill)", "extra_rules", "Feet in frame: white sneakers. Never glasses.", true))),
      h("div", { class: "ost-card" }, h("h4", {}, "Character LoRAs (applied to MODEL / CLIP)"), loraRows),
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
        } }, "Delete") : null));
    // Workflow-only LoRAs
    const extra = h("div", { class: "ost-card" }, h("h4", {}, "Extra LoRAs — this workflow only"),
      ...(state.extra_loras || []).map((r, i) => h("div", { class: "ost-lora" },
        h("input", { type: "checkbox", checked: r.on !== false, onchange: (e) => { r.on = e.target.checked; saveState(); } }),
        h("select", { onchange: (e) => { r.lora = e.target.value; saveState(); } }, [r.lora, ...boot.loras.filter((x) => x !== r.lora)].filter(Boolean).map((f) => h("option", { value: f, selected: f === r.lora ? "selected" : null }, f))),
        h("input", { type: "number", step: 0.05, value: r.sm ?? 1, onchange: (e) => { r.sm = Number(e.target.value); saveState(); } }),
        h("input", { type: "number", step: 0.05, value: r.sc ?? 1, onchange: (e) => { r.sc = Number(e.target.value); saveState(); } }),
        h("button", { class: "ost-btn ghost bad", style: "padding:4px 6px", onclick: () => { state.extra_loras.splice(i, 1); set({}); } }, "✕"))),
      h("button", { class: "ost-btn", onclick: () => { (state.extra_loras ||= []).push({ lora: boot.loras[0] || "", sm: 1, sc: 1, on: true }); set({}); } }, "+ LoRA"),
      h("div", { class: "ost-hint", style: "margin-top:6px" }, "Style / realism LoRAs for this workflow (e.g. candid_snap 1.0, RealisticSnapshot 0.6). Saved in the workflow, not in the character."));
    fill(p, h("div", { class: "ost-split" }, list, h("div", {}, editor, extra)));
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
      h("div", { class: "ost-hint", style: "margin-top:10px" }, "Imported skills are stored in ", h("code", {}, (boot.studio_dir || "") + "/skills"), ", on this computer only. Grok presets come from the Grok Prompt Generator node."));
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
      h("div", { class: "grow" }, h("div", { class: "sub" }, `${fmtAge(e.ts)} · ${e.skill || e.mode}${e.notes ? " · " + e.notes : ""}`),
        h("div", { style: "white-space:pre-wrap;font-size:12px;max-height:54px;overflow:hidden" }, e.prompt)),
      h("button", { class: "ost-btn ghost", onclick: (ev) => { ev.stopPropagation(); set({ mode: "manual", manual: e.prompt, tab: "prompt" }); } }, "Use")))) :
      h("div", { class: "ost-empty" }, "No prompt generated yet."));
  }

  // ── mount ──────────────────────────────────────────────────────────────────
  mountBatchPanel(node, pages.media);   // the Batch Loader's own panel, inside the Media tab
  hideWidget(W("studio_state"));
  setTimeout(() => hideWidget(W("studio_state")), 300);
  node.addDOMWidget("studio_panel", "ONYX_STUDIO", root, {
    serialize: false, getValue: () => undefined, setValue: () => {}, getMinHeight: () => PANEL_MIN_H,
  });
  node.size = [...NODE_SIZE];

  const origConfigure = node.onConfigure?.bind(node);
  node.onConfigure = function (info) { origConfigure?.(info); readState(); render(); };

  const onResult = ({ detail }) => { if (String(detail?.node) === String(node.id)) showResult(detail); };
  api.addEventListener("onyx.studio.result", onResult);
  const origRemoved = node.onRemoved?.bind(node);
  node.onRemoved = function () { api.removeEventListener("onyx.studio.result", onResult); origRemoved?.(); };

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
