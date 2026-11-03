// Onyx Post Studio ✦ — the whole post-processing chain in one node.
// Stages, in the Crispy workflow's order (motion last): color finish → skin &
// eyes → grade & optics → camera sensor → grain & noise → JPEG & contrast →
// motion. One tab per stage (+ the Chain overview), on/off per stage, live
// before/after preview, 10 presets. Settings are kept in the hidden
// `post_state` widget (JSON); the Python side runs the existing nodes in order.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_SIZE = [760, 1000];
const PANEL_MIN_H = 640;
const PREVIEW_SIZE = 1024;
const GROUPS = { scene: "The scene", phone: "The phone", upload: "The upload" };

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
const fill = (el, ...kids) => el.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
async function getJSON(url) { return (await api.fetchApi(url)).json(); }
async function postJSON(url, body) {
  return (await api.fetchApi(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
}
function toast(detail, severity = "info") {
  if (app.extensionManager?.toast?.add) app.extensionManager.toast.add({ severity, summary: "Onyx Post Studio", detail, life: 4000 });
  else console.log("[Onyx Post Studio] " + detail);
}
const human = (s) => String(s).replace(/^enable_/, "").replace(/_/g, " ").replace(/\bca\b/g, "CA").replace(/\bnr\b/g, "NR").replace(/^./, (c) => c.toUpperCase());
const same = (a, b) => (typeof a === "number" && typeof b === "number") ? Math.abs(a - b) < 1e-9 : a === b;

function injectStyle() {
  if (document.getElementById("onyx-post-studio-style")) return;
  const css = `
.ops{--bg:#0c0c10;--p1:#121218;--p2:#17171f;--p3:#1d1d27;--ln:#24242f;--ln2:#33333f;--tx:#ececf3;--dim:#8a8c99;--ac:#9fb4ff;--ac2:#c7d2ff;--ok:#3ecf8e;--warm:#ffc58a;
  display:flex;flex-direction:column;position:relative;width:100%;height:100%;box-sizing:border-box;background:var(--bg);color:var(--tx);
  border:1px solid var(--ln);border-radius:12px;overflow:hidden;font:12.5px/1.45 Inter,"Segoe UI",system-ui,sans-serif}
.ops *{box-sizing:border-box}
.ops-head{display:flex;align-items:center;gap:10px;padding:10px 12px;background:linear-gradient(120deg,#1b1a2b 0%,#141420 55%,#261a14 100%);border-bottom:1px solid var(--ln)}
.ops-logo{font-weight:800;letter-spacing:.14em;font-size:13px;background:linear-gradient(90deg,#ffd7a8,#e4b8ff);-webkit-background-clip:text;background-clip:text;color:transparent;white-space:nowrap}
.ops-pill{padding:4px 10px;border-radius:999px;background:var(--p3);border:1px solid var(--ln2);font-size:11.5px;white-space:nowrap}
.ops-pill b{color:var(--ac2)}.ops-pill.link{color:var(--warm);border-color:#5a4630}
.ops-presets{display:flex;align-items:center;gap:5px;padding:6px 10px;background:var(--p1);border-bottom:1px solid var(--ln);flex-wrap:wrap}
.ops-presets .lbl{color:var(--dim);font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-right:4px}
.ops-ps{min-width:28px;max-width:118px;padding:4px 8px;border-radius:7px;border:1px solid var(--ln2);background:var(--p2);color:var(--dim);font:600 11px Inter,system-ui,sans-serif;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ops-ps.full{color:var(--tx);border-color:#46507e;background:#1c2038}.ops-ps.cur{box-shadow:0 0 0 2px rgba(159,180,255,.45)}
.ops-ps.saving{border-style:dashed;border-color:var(--warm);color:var(--warm)}.ops-ps:hover{border-color:var(--ac)}
.ops-prev{display:grid;grid-template-columns:1fr 170px;grid-template-rows:minmax(0,1fr);gap:10px;padding:10px;border-bottom:1px solid var(--ln);background:var(--p1);height:330px;flex:none}
.ops-prev.min{height:auto;grid-template-columns:1fr auto}.ops-prev.min .ops-cmp{display:none}
.ops-cmp{position:relative;border:1px solid var(--ln);border-radius:9px;overflow:hidden;cursor:ew-resize;user-select:none;
  background:#0a0a0e repeating-conic-gradient(#13131a 0 25%,#0d0d12 0 50%) 0 0/18px 18px}
.ops-cmp img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none}
.ops-cmp .bar{position:absolute;top:0;bottom:0;width:2px;background:#fff;box-shadow:0 0 6px #000;pointer-events:none}
.ops-cmp .bar:after{content:"⟷";position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;color:#111;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:12px}
.ops-cmp .tag{position:absolute;top:6px;padding:2px 7px;border-radius:6px;background:#000a;font-size:10.5px;font-weight:700;letter-spacing:.06em;pointer-events:none}
.ops-cmp .msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:20px;color:var(--dim)}
.ops-cmp.empty img,.ops-cmp.empty .bar,.ops-cmp.empty .tag{display:none}
.ops-cmp.hot{outline:2px dashed var(--ac);outline-offset:-4px}
.ops-side{display:flex;flex-direction:column;gap:7px;min-width:0;min-height:0;overflow:hidden}
.ops-side .t{font-size:11px;color:var(--dim)}
.ops-tim{font:11px/1.5 ui-monospace,Consolas,monospace;color:#c9cbd8;overflow:auto;flex:1;min-height:0}
.ops-seg{display:flex;background:var(--p2);border:1px solid var(--ln2);border-radius:9px;padding:2px}
.ops-seg button{flex:1;background:transparent;border:0;color:var(--dim);padding:4px 6px;border-radius:7px;cursor:pointer;font:600 11.5px Inter,system-ui,sans-serif}
.ops-seg button.on{background:linear-gradient(135deg,#2a3156,#30264a);color:var(--tx)}
.ops-main{flex:1;min-height:0;display:grid;grid-template-columns:150px minmax(0,1fr)}
.ops-tabs{overflow:auto;background:var(--p1);border-right:1px solid var(--ln);padding:6px;scrollbar-width:thin;scrollbar-color:var(--ln2) transparent}
.ops-tabs .grp{color:var(--dim);font-size:9.5px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;padding:8px 6px 3px}
.ops-tab{padding:5px 7px;border-radius:7px;cursor:pointer;color:var(--dim);font-weight:600;font-size:11.5px;white-space:nowrap;display:flex;align-items:center;gap:6px;border:1px solid transparent}
.ops-tab:hover{color:var(--tx)}
.ops-tab.on{background:var(--p3);color:var(--tx);border-color:var(--ln2)}
.ops-tab .dot{width:6px;height:6px;border-radius:50%;background:#444;flex:none}.ops-tab .dot.on{background:var(--ok)}
.ops-tab .num{font-size:10px;color:var(--ac2);min-width:13px;text-align:right}
.ops-body{min-height:0;position:relative}
.ops-ghd{color:var(--dim);font-size:10px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin:6px 2px 4px}
.ops-page{position:absolute;inset:0;overflow:auto;padding:12px;display:none;scrollbar-width:thin;scrollbar-color:var(--ln2) transparent}
.ops-page.on{display:block}
.ops-card{background:var(--p1);border:1px solid var(--ln);border-radius:10px;padding:10px;margin-bottom:10px}
.ops-card h4{margin:0 0 8px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:700;display:flex;align-items:center;gap:8px}
.ops-card h4 .sp{flex:1}
.ops-card.off .ops-params{opacity:.4}
.ops-params{display:flex;flex-direction:column;gap:5px}
.ops-p{display:grid;grid-template-columns:150px 1fr 78px 18px;gap:8px;align-items:center}
.ops-p .lb{color:var(--dim);font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ops-p .lb.mod{color:var(--ac2)}
.ops-p .rs{color:var(--dim);cursor:pointer;font-size:12px;visibility:hidden}.ops-p .rs.on{visibility:visible}.ops-p .rs:hover{color:var(--tx)}
.ops-p .wide{grid-column:2 / 4}
.ops input[type=range]{width:100%;accent-color:#8a7cf0}
.ops input[type=number],.ops input[type=text],.ops select{background:var(--p2);color:var(--tx);border:1px solid var(--ln2);border-radius:7px;padding:4px 6px;font:12px Inter,system-ui,sans-serif;width:100%;outline:none}
.ops input:focus,.ops select:focus{border-color:#5b6aa8}
.ops-btn{background:var(--p3);color:var(--tx);border:1px solid var(--ln2);border-radius:8px;padding:5px 11px;cursor:pointer;font:600 11.5px Inter,system-ui,sans-serif;white-space:nowrap}
.ops-btn:hover{border-color:var(--ac)}.ops-btn.pri{background:linear-gradient(135deg,#c9814f,#9a52c7);border-color:transparent;color:#fff}
.ops-btn.ghost{background:transparent}.ops-btn:disabled{opacity:.45;cursor:default}
.ops-hint{color:var(--dim);font-size:11.5px}
.ops-tog{display:inline-flex;align-items:center;gap:7px;cursor:pointer;color:var(--tx);font-size:12px;user-select:none;text-transform:none;letter-spacing:0;font-weight:600}
.ops-tog i{width:30px;height:16px;border-radius:8px;background:#2a2a35;position:relative;transition:background .15s;flex:none}
.ops-tog i:after{content:"";position:absolute;top:2px;left:2px;width:12px;height:12px;border-radius:50%;background:#9a9aa8;transition:left .15s,background .15s}
.ops-tog.on i{background:#5468c9}.ops-tog.on i:after{left:16px;background:#fff}
.ops-chain{display:flex;flex-direction:column;gap:0}
.ops-st{display:grid;grid-template-columns:26px 22px 1fr auto auto;gap:8px;align-items:center;padding:6px 9px;border:1px solid var(--ln);border-radius:10px;background:var(--p2)}
.ops-st.off{opacity:.5}
.ops-st .n{width:24px;height:24px;border-radius:50%;background:var(--p3);color:var(--ac2);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:11.5px}
.ops-st .ic{font-size:17px;text-align:center}
.ops-st .tt{font-weight:700}.ops-st .sub{color:var(--dim);font-size:11px}
.ops-arrow{text-align:center;color:var(--ln2);font-size:10px;line-height:10px}
.ops-looks{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:7px}
.ops-look{border:1px solid var(--ln);border-radius:9px;background:var(--p2);padding:8px;cursor:pointer}
.ops-look:hover{border-color:var(--ac)}.ops-look b{display:block;margin-bottom:2px}.ops-look span{color:var(--dim);font-size:11px}
.ops-spin{display:inline-block;width:11px;height:11px;border:2px solid #ffffff44;border-top-color:#fff;border-radius:50%;animation:opsspin .8s linear infinite;vertical-align:-2px;margin-right:5px}
@keyframes opsspin{to{transform:rotate(360deg)}}
`;
  document.head.append(h("style", { id: "onyx-post-studio-style", html: css }));
}

// The browser's prompt dialog does not exist in ComfyUI Desktop (Electron): a
// small inline dialog inside the node instead. Resolves to the text, or null.
function askText(host, title, value, cls) {
  return new Promise((resolve) => {
    const inp = document.createElement("input");
    inp.type = "text"; inp.value = value || "";
    inp.style.cssText = "width:100%;background:#17171f;color:#ececf3;border:1px solid #46507e;border-radius:8px;padding:7px 9px;font:13px Inter,system-ui,sans-serif;outline:none;margin:10px 0";
    const box = document.createElement("div");
    box.style.cssText = "position:absolute;inset:0;z-index:50;background:rgba(5,5,9,.6);display:flex;align-items:center;justify-content:center";
    const card = document.createElement("div");
    card.style.cssText = "width:min(340px,90%);background:#121218;border:1px solid #33333f;border-radius:12px;padding:14px;box-shadow:0 10px 40px #000a;color:#ececf3;font:600 12.5px Inter,system-ui,sans-serif";
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:8px;justify-content:flex-end";
    const done = (v) => { box.remove(); resolve(v); };
    const mk = (label, primary, fn) => { const b = document.createElement("button"); b.className = cls + (primary ? " pri" : " ghost"); b.textContent = label; b.onclick = fn; return b; };
    row.append(mk("Cancel", false, () => done(null)), mk("Save", true, () => done(inp.value.trim() || value || "")));
    card.append(title, inp, row);
    box.append(card);
    box.addEventListener("pointerdown", (e) => { e.stopPropagation(); if (e.target === box) done(null); });
    inp.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") done(inp.value.trim() || value || ""); if (e.key === "Escape") done(null); });
    host.append(box);
    setTimeout(() => { inp.focus(); inp.select(); }, 0);
  });
}
// Tell the workflow tab its graph changed (autosave / "modified" dot), debounced.
let _trackTimer = null;
function markChanged(app) {
  clearTimeout(_trackTimer);
  _trackTimer = setTimeout(() => {
    try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.(); } catch { /* older frontend */ }
  }, 250);
}

function hideWidget(w) {
  if (!w) return;
  w.hidden = true; w.computeSize = () => [0, -4]; w.draw = () => {};
  if (w.element) w.element.style.cssText = "display:none!important";
}
function toggle(label, value, onChange, title) {
  const t = h("span", { class: "ops-tog" + (value ? " on" : ""), title }, h("i"), label);
  t.addEventListener("click", (e) => { e.stopPropagation(); t.classList.toggle("on"); onChange(t.classList.contains("on")); });
  return t;
}

function setupPostStudio(node) {
  injectStyle();
  const W = (n) => node.widgets?.find((w) => w.name === n);
  let spec = { stages: [], looks: {}, look_info: {}, presets: {} };
  let state = { stages: {}, tab: "chain", view: { mode: "fit", cx: 0.5, cy: 0.5, auto: true, split: 50, open: true } };
  const readState = () => {
    try {
      const s = JSON.parse(W("post_state")?.value || "{}") || {};
      state = { stages: s.stages || {}, tab: s.tab || "chain", view: { ...state.view, ...(s.view || {}) }, current_preset: s.current_preset };
    } catch { /* keep */ }
  };
  const saveState = () => { const w = W("post_state"); if (w) w.value = JSON.stringify(state); app.graph?.setDirtyCanvas(true, true); markChanged(app); };
  readState();

  const stageDef = (k) => spec.stages.find((s) => s.key === k);
  const st = (k) => (state.stages[k] ||= { on: stageDef(k)?.default_on !== false, v: {} });
  const isOn = (k) => state.stages[k] ? state.stages[k].on !== false : stageDef(k)?.default_on !== false;
  const val = (k, p) => { const v = state.stages[k]?.v?.[p.name]; return v === undefined ? p.default : v; };
  function setVal(k, p, v) {
    const s = st(k); s.v ||= {};
    if (same(v, p.default)) delete s.v[p.name]; else s.v[p.name] = v;
    state.current_preset = null; saveState(); schedulePreview();
  }

  // ── skeleton ───────────────────────────────────────────────────────────────
  const root = h("div", { class: "ops" });
  root.addEventListener("keydown", (e) => { if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) e.stopPropagation(); });   // bubble phase: the input's own keys run first
  root.addEventListener("wheel", (e) => {
    const sc = e.target.closest?.(".ops-page,.ops-tim");
    if (sc && sc.scrollHeight > sc.clientHeight) e.stopPropagation();
  }, { passive: true });

  const chainPill = h("div", { class: "ops-pill" });
  const linkPill = h("div", { class: "ops-pill link", title: "The post input is connected: a look other than none overrides these settings at run time." }, "🔗 look from Studio");
  const head = h("div", { class: "ops-head" }, h("div", { class: "ops-logo" }, "ONYX POST STUDIO ✦"), chainPill, linkPill);

  const presetBar = h("div", { class: "ops-presets" });
  // preview
  const imgBefore = h("img", { alt: "" }), imgAfter = h("img", { alt: "" });
  const bar = h("div", { class: "bar" });
  const msg = h("div", { class: "msg" }, "Run the workflow once, or drop an image here, to preview the chain.");
  const cmp = h("div", { class: "ops-cmp empty", title: "Drag: before / after · double-click: 100% view there · drop an image to preview it" },
    imgAfter, imgBefore, bar, h("div", { class: "tag", style: "left:6px" }, "BEFORE"), h("div", { class: "tag", style: "right:6px" }, "AFTER"), msg);
  const timBox = h("div", { class: "ops-tim" });
  const modeSeg = h("div", { class: "ops-seg" });
  const refreshBtn = h("button", { class: "ops-btn pri", onclick: () => runPreview() }, "↻ Preview");
  const autoTog = h("span");
  const sizeInfo = h("div", { class: "t" });
  const openBtn = h("button", { class: "ops-btn ghost", style: "padding:3px 8px", onclick: () => { state.view.open = !state.view.open; saveState(); layoutPreview(); } });
  const side = h("div", { class: "ops-side" }, h("div", { style: "display:flex;gap:6px;align-items:center" }, refreshBtn, openBtn), modeSeg, autoTog, sizeInfo, timBox);
  const prevBox = h("div", { class: "ops-prev" }, cmp, side);

  const tabsBar = h("div", { class: "ops-tabs" });
  const body = h("div", { class: "ops-body" });
  const pages = {};
  root.append(head, presetBar, prevBox, h("div", { class: "ops-main" }, tabsBar, body));

  // ── compare slider ─────────────────────────────────────────────────────────
  const setSplit = (x) => {
    state.view.split = Math.max(0, Math.min(100, x));
    imgBefore.style.clipPath = `inset(0 ${100 - state.view.split}% 0 0)`;
    bar.style.left = `calc(${state.view.split}% - 1px)`;
  };
  let dragging = false;
  const splitFromEvent = (e) => { const r = cmp.getBoundingClientRect(); setSplit(((e.clientX - r.left) / r.width) * 100); };
  cmp.addEventListener("pointerdown", (e) => { dragging = true; cmp.setPointerCapture?.(e.pointerId); splitFromEvent(e); e.stopPropagation(); });
  cmp.addEventListener("pointermove", (e) => { if (dragging) { splitFromEvent(e); e.stopPropagation(); } });
  cmp.addEventListener("pointerup", (e) => { dragging = false; saveState(); e.stopPropagation(); });
  cmp.addEventListener("dblclick", (e) => {
    // where in the IMAGE (object-fit: contain) was the click?
    const r = cmp.getBoundingClientRect(), nw = imgAfter.naturalWidth, nh = imgAfter.naturalHeight;
    if (!nw || !nh) return;
    if (state.view.mode === "crop") { state.view.mode = "fit"; saveState(); renderSide(); return runPreview(); }
    const s = Math.min(r.width / nw, r.height / nh), ox = (r.width - nw * s) / 2, oy = (r.height - nh * s) / 2;
    state.view.cx = Math.max(0, Math.min(1, (e.clientX - r.left - ox) / (nw * s)));
    state.view.cy = Math.max(0, Math.min(1, (e.clientY - r.top - oy) / (nh * s)));
    state.view.mode = "crop"; saveState(); renderSide(); runPreview();
  });
  // drop an image: becomes the preview source (never reaches ComfyUI's canvas)
  const isFileDrag = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  root.addEventListener("dragover", (e) => { if (!isFileDrag(e)) return; e.preventDefault(); e.stopPropagation(); cmp.classList.add("hot"); });
  root.addEventListener("dragleave", () => cmp.classList.remove("hot"));
  root.addEventListener("drop", async (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault(); e.stopPropagation(); cmp.classList.remove("hot");
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
    if (!f) return toast("Drop an image file.", "warn");
    const data_url = await new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(f); });
    const r = await postJSON("/onyx/post_studio/source", { node_id: String(node.id), data_url });
    if (!r.ok) return toast(r.error, "error");
    if (!state.view.open) { state.view.open = true; layoutPreview(); }
    toast(`Preview image: ${f.name} (${r.size[0]}×${r.size[1]})`, "success");
    runPreview();
  }, true);

  // ── preview ────────────────────────────────────────────────────────────────
  let busy = false, again = false, timer = null;
  function schedulePreview() {
    renderChrome();
    if (!state.view.auto || !state.view.open) return;
    clearTimeout(timer); timer = setTimeout(runPreview, 450);
  }
  async function runPreview() {
    if (busy) { again = true; return; }
    busy = true; refreshBtn.disabled = true; fill(refreshBtn, h("span", { class: "ops-spin" }), "Rendering");
    try {
      const r = await postJSON("/onyx/post_studio/preview", { node_id: String(node.id), state: { stages: state.stages }, mode: state.view.mode, cx: state.view.cx, cy: state.view.cy, size: PREVIEW_SIZE });
      if (!r.ok) { showEmpty(r.error || "Preview failed."); fill(timBox); return; }
      msg.style.display = "none"; cmp.classList.remove("empty");
      imgBefore.src = r.before; imgAfter.src = r.after; setSplit(state.view.split);
      const total = r.timings.reduce((a, [, t]) => a + (typeof t === "number" ? t : 0), 0);
      fill(timBox, r.timings.map(([k, t]) => typeof t === "number"
        ? h("div", {}, `${stageDef(k)?.icon || ""} ${stageDef(k)?.short || k}`.padEnd(10, " ") + ` ${(t * 1000).toFixed(0)} ms`)
        : h("div", { style: "color:#ff9b9b;white-space:normal", title: t }, `⚠️ ${stageDef(k)?.short || k} skipped (hover)`)),
        h("div", { style: "color:var(--dim);margin-top:3px" }, `total ${(total * 1000).toFixed(0)} ms`),
        r.post_look && r.post_look !== "none" ? h("div", { style: "color:var(--warm)" }, `look '${r.post_look}' from Studio applied`) : null);
      fill(sizeInfo, `${r.source_size[0]}×${r.source_size[1]} · ${state.view.mode === "crop" ? "100% crop" : "fit"} ${r.view_size[0]}×${r.view_size[1]}`);
    } catch (e) {
      showEmpty(String(e));
    } finally {
      busy = false; refreshBtn.disabled = false; fill(refreshBtn, "↻ Preview");
      if (again) { again = false; runPreview(); }
    }
  }
  function showEmpty(text) {
    cmp.classList.add("empty"); msg.style.display = ""; fill(msg, text);
    imgBefore.removeAttribute("src"); imgAfter.removeAttribute("src");
  }
  function renderSide() {
    fill(modeSeg, [["fit", "Fit"], ["crop", "100%"]].map(([m, l]) => h("button", {
      class: state.view.mode === m ? "on" : null, title: m === "crop" ? "Pixel-exact view (grain, sharpness). Double-click the image to choose where." : "Whole image, downscaled",
      onclick: () => { state.view.mode = m; saveState(); renderSide(); runPreview(); } }, l)));
    fill(autoTog, toggle("Auto preview", state.view.auto, (v) => { state.view.auto = v; saveState(); if (v) runPreview(); }, "Re-render after every change"));
  }
  function layoutPreview() {
    prevBox.classList.toggle("min", !state.view.open);
    fill(openBtn, state.view.open ? "▴ Hide" : "▾ Show");
    modeSeg.style.display = autoTog.style.display = timBox.style.display = sizeInfo.style.display = state.view.open ? "" : "none";
  }

  // ── presets ────────────────────────────────────────────────────────────────
  let saveMode = false;
  function renderPresets() {
    const saved = spec.presets || {};
    const chips = [];
    for (let i = 1; i <= 10; i++) {
      const pr = saved[String(i)];
      chips.push(h("button", {
        class: "ops-ps" + (pr ? " full" : "") + (String(state.current_preset) === String(i) ? " cur" : "") + (saveMode ? " saving" : ""),
        title: pr ? `${pr.name}\nclick: load · right-click: delete` : "empty slot",
        onclick: async () => {
          if (saveMode) {
            const name = await askText(root, `Name for preset ${i}`, pr?.name || `Preset ${i}`, "ops-btn");
            if (name === null) return;
            const r = await postJSON("/onyx/post_studio/preset/save", { slot: i, name, state: { stages: state.stages } });
            if (!r.ok) return toast(r.error, "error");
            spec.presets = r.presets; saveMode = false; state.current_preset = i; saveState(); renderPresets();
            return toast(`Saved to preset ${i}.`, "success");
          }
          if (!pr) return toast("Empty slot: click 💾 Save, then this slot.");
          state.stages = structuredClone(pr.stages || {}); state.current_preset = i; saveState(); render(); schedulePreview();
          toast(`Preset ${i} “${pr.name}” loaded.`, "success");
        },
        oncontextmenu: async (e) => {
          e.preventDefault(); e.stopPropagation();
          if (!pr || !confirm(`Delete preset ${i} “${pr.name}”?`)) return;
          const r = await postJSON("/onyx/post_studio/preset/delete", { slot: i });
          spec.presets = r.presets || {}; renderPresets();
        } }, pr ? `${i} · ${pr.name}` : String(i)));
    }
    fill(presetBar, h("span", { class: "lbl" }, "Presets"), chips,
      h("button", { class: "ops-btn" + (saveMode ? " pri" : " ghost"), style: "margin-left:auto;padding:4px 10px", title: "Saves every stage: on/off + all settings",
        onclick: () => { saveMode = !saveMode; renderPresets(); } }, saveMode ? "Click a slot…" : "💾 Save"));
  }

  // ── tabs ───────────────────────────────────────────────────────────────────
  function renderChrome() {
    const on = spec.stages.filter((s) => isOn(s.key)).length;
    fill(chainPill, h("b", {}, `${on}/${spec.stages.length}`), " stages on");
    const linked = node.inputs?.find((i) => i.name === "onyx_post")?.link != null;
    linkPill.style.display = linked ? "" : "none";
    const items = [h("div", { class: "ops-tab" + (state.tab === "chain" ? " on" : ""), onclick: () => setTab("chain") }, "⛓ Chain")];
    let grp = null;
    spec.stages.forEach((s, i) => {
      if (s.group !== grp) { grp = s.group; items.push(h("div", { class: "grp" }, GROUPS[grp] || grp)); }
      items.push(h("div", { class: "ops-tab" + (state.tab === s.key ? " on" : ""), title: s.info, onclick: () => setTab(s.key) },
        h("span", { class: "dot" + (isOn(s.key) ? " on" : "") }), h("span", { class: "num" }, i + 1), `${s.icon} ${s.short || s.title}`));
    });
    fill(tabsBar, items);
    for (const [k, p] of Object.entries(pages)) p.classList.toggle("on", k === state.tab);
  }
  function setTab(t) { state.tab = t; saveState(); render(); }

  function renderChain() {
    const p = pages.chain;
    const rows = [];
    let grp = null;
    spec.stages.forEach((s, i) => {
      if (s.group !== grp) { grp = s.group; rows.push(h("div", { class: "ops-ghd" }, GROUPS[grp] || grp)); }
      else if (i) rows.push(h("div", { class: "ops-arrow" }, "↓"));
      rows.push(h("div", { class: "ops-st" + (isOn(s.key) ? "" : " off") },
        h("div", { class: "n" }, i + 1), h("div", { class: "ic" }, s.icon),
        h("div", {}, h("div", { class: "tt" }, s.title, s.available ? "" : " — not available"), h("div", { class: "sub" }, s.info)),
        toggle("", isOn(s.key), (v) => { st(s.key).on = v; state.current_preset = null; saveState(); render(); schedulePreview(); }, "Stage on / off"),
        h("button", { class: "ops-btn ghost", onclick: () => setTab(s.key) }, "Edit ›")));
    });
    fill(p,
      h("div", { class: "ops-card" }, h("h4", {}, "The chain, in order"), h("div", { class: "ops-chain" }, rows),
        h("div", { class: "ops-hint", style: "margin-top:8px" }, "The 4K Crispy workflow's order: first the scene (colour, skin, grade, LUT), then what the phone does to it (lens, sensor clipping, sensor, HDR processing, grain, JPEG, shake), and last what the upload does to the file. New stages start switched off.")),
      h("div", { class: "ops-card" }, h("h4", {}, "Start from a look", h("span", { class: "sp" }),
        h("button", { class: "ops-btn ghost", onclick: () => { if (confirm("Reset every stage to the Crispy workflow settings?")) { state.stages = {}; state.current_preset = null; saveState(); render(); schedulePreview(); } } }, "↺ Reset all (Crispy)")),
        h("div", { class: "ops-looks" }, Object.keys(spec.looks).map((name) => h("div", { class: "ops-look", onclick: () => loadLook(name) },
          h("b", {}, name), h("span", {}, spec.look_info[name] || "")))),
        h("div", { class: "ops-hint", style: "margin-top:8px" }, "Copies the look's values into the tabs, to tweak from there. To change the look per run instead, wire Onyx Studio's post output to this node's onyx_post input.")));
  }
  function loadLook(name) {
    const look = spec.looks[name] || {};
    const stages = {};
    for (const s of spec.stages) {
      const vals = look[s.key] || {};
      const v = {};
      for (const p of s.params) if (p.name in vals && !same(vals[p.name], p.default)) v[p.name] = vals[p.name];
      // a stage the look does not mention keeps its own on/off (a look never switches on the LUT...)
      const on = s.key in look ? !vals._bypass : isOn(s.key);
      stages[s.key] = { on, v };
    }
    stages.motion = state.stages.motion || stages.motion;     // a look never changes the motion
    state.stages = stages; state.current_preset = null; saveState(); render(); schedulePreview();
    toast(`Look “${name}” loaded into the settings.`, "success");
  }

  function control(s, p, onAny) {
    const v = val(s.key, p);
    const lb = h("div", { class: "lb" + (same(v, p.default) ? "" : " mod"), title: (p.tooltip ? p.tooltip + "\n" : "") + `default: ${p.default}` }, human(p.name));
    const rs = h("span", { class: "rs" + (same(v, p.default) ? "" : " on"), title: "Back to default" }, "↺");
    const mark = (nv) => { lb.classList.toggle("mod", !same(nv, p.default)); rs.classList.toggle("on", !same(nv, p.default)); };
    const commit = (nv) => { setVal(s.key, p, nv); mark(nv); onAny?.(); };
    if (p.type === "BOOLEAN") {
      const t = toggle("", !!v, (nv) => commit(nv));
      rs.onclick = () => { commit(p.default); t.classList.toggle("on", !!p.default); };
      return h("div", { class: "ops-p" }, lb, h("div", { class: "wide" }, t), rs);
    }
    if (p.type === "COMBO") {
      const sel = h("select", { onchange: (e) => commit(e.target.value) }, p.options.map((o) => h("option", { value: o, selected: o === String(v) ? "selected" : null }, o)));
      rs.onclick = () => { sel.value = p.default; commit(p.default); };
      return h("div", { class: "ops-p" }, lb, h("div", { class: "wide" }, sel), rs);
    }
    if (p.type === "STRING") {
      const inp = h("input", { type: "text", value: v ?? "", onchange: (e) => commit(e.target.value) });
      rs.onclick = () => { inp.value = p.default; commit(p.default); };
      return h("div", { class: "ops-p" }, lb, h("div", { class: "wide" }, inp), rs);
    }
    const isInt = p.type === "INT";
    const step = p.step ?? (isInt ? 1 : 0.01);
    const parse = (x) => { let n = isInt ? parseInt(x, 10) : parseFloat(x); if (!isFinite(n)) n = p.default; if (p.min != null) n = Math.max(p.min, n); if (p.max != null) n = Math.min(p.max, n); return n; };
    const num = h("input", { type: "number", value: v, step, min: p.min, max: p.max });
    const huge = p.max == null || p.min == null || (p.max - p.min) / step > 1e6;
    if (huge) {   // seeds and the like: number + dice
      num.onchange = (e) => commit(parse(e.target.value));
      const dice = h("button", { class: "ops-btn ghost", style: "padding:3px 8px", title: "Random", onclick: () => { const n = Math.floor(Math.random() * 2 ** 32); num.value = n; commit(n); } }, "🎲");
      rs.onclick = () => { num.value = p.default; commit(p.default); };
      return h("div", { class: "ops-p" }, lb, h("div", { style: "display:flex;gap:6px;grid-column:2 / 4" }, num, dice), rs);
    }
    const rng = h("input", { type: "range", min: p.min, max: p.max, step, value: v });
    rng.oninput = (e) => { num.value = e.target.value; mark(parse(e.target.value)); };
    rng.onchange = (e) => commit(parse(e.target.value));
    num.onchange = (e) => { const n = parse(e.target.value); num.value = n; rng.value = n; commit(n); };
    rs.onclick = () => { num.value = p.default; rng.value = p.default; commit(p.default); };
    return h("div", { class: "ops-p" }, lb, rng, num, rs);
  }

  function renderStage(key) {
    const s = stageDef(key), p = pages[key];
    if (!s) return;
    const groups = [];
    for (const prm of s.params) {
      let g = groups.find((x) => x.id === prm.group);
      if (!g) groups.push(g = { id: prm.group, switch: null, params: [] });
      if (prm.switch) g.switch = prm; else g.params.push(prm);
    }
    groups.sort((a, b) => (a.id === "main" ? -1 : 0) - (b.id === "main" ? -1 : 0));
    const cards = groups.map((g) => {
      const on = g.switch ? !!val(key, g.switch) : true;
      const card = h("div", { class: "ops-card" + (on ? "" : " off") });
      const title = g.id === "main" ? "Settings" : human(g.id);
      const sw = g.switch ? toggle(on ? "on" : "off", on, (v) => { setVal(key, g.switch, v); card.classList.toggle("off", !v); sw.lastChild.textContent = v ? "on" : "off"; }) : null;
      fill(card, h("h4", {}, title, h("span", { class: "sp" }), sw), g.params.length ? h("div", { class: "ops-params" }, g.params.map((prm) => control(s, prm))) : null);
      return card;
    });
    fill(p,
      h("div", { class: "ops-card", style: "background:linear-gradient(120deg,#16161f,#1a1622)" },
        h("h4", { style: "font-size:12px;color:var(--tx)" }, `${s.icon} ${s.title}`, h("span", { class: "sp" }),
          toggle(isOn(key) ? "Stage on" : "Stage off", isOn(key), (v) => { st(key).on = v; state.current_preset = null; saveState(); render(); schedulePreview(); }),
          h("button", { class: "ops-btn ghost", title: "Every setting of this stage back to the Crispy workflow values", onclick: () => { st(key).v = {}; state.current_preset = null; saveState(); renderStage(key); schedulePreview(); } }, "↺ Reset")),
        h("div", { class: "ops-hint" }, s.info, s.available ? "" : " ⚠️ This node is not available in this install — the stage is skipped.")),
      h("div", { style: isOn(key) ? "" : "opacity:.45" }, cards));
  }

  function render() {
    for (const s of [{ key: "chain" }, ...spec.stages]) if (!pages[s.key]) { pages[s.key] = h("div", { class: "ops-page" }); body.append(pages[s.key]); }
    if (!pages[state.tab]) state.tab = "chain";
    renderChrome(); renderPresets(); renderSide(); layoutPreview(); setSplit(state.view.split);
    if (state.tab === "chain") renderChain(); else renderStage(state.tab);
  }

  // ── mount ──────────────────────────────────────────────────────────────────
  hideWidget(W("post_state"));
  setTimeout(() => hideWidget(W("post_state")), 300);
  node.addDOMWidget("post_panel", "ONYX_POST_STUDIO", root, {
    serialize: false, getValue: () => undefined, setValue: () => {}, getMinHeight: () => PANEL_MIN_H,
  });
  node.size = [...NODE_SIZE];
  const origConfigure = node.onConfigure?.bind(node);
  node.onConfigure = function (info) { origConfigure?.(info); readState(); render(); };
  const origConn = node.onConnectionsChange?.bind(node);
  node.onConnectionsChange = function (...a) { origConn?.(...a); renderChrome(); };
  // A finished run gave this node a new preview source: refresh the preview.
  let ranThisNode = false;
  const onExecuting = ({ detail }) => {
    const id = detail && typeof detail === "object" ? detail.node : detail;
    if (id != null) { if (String(id) === String(node.id)) ranThisNode = true; return; }
    if (ranThisNode && state.view.auto && state.view.open) runPreview();       // the queue item is done
    ranThisNode = false;
  };
  api.addEventListener("executing", onExecuting);
  const origRemoved = node.onRemoved?.bind(node);
  node.onRemoved = function () { api.removeEventListener("executing", onExecuting); origRemoved?.(); };

  render();
  getJSON("/onyx/post_studio/spec").then((r) => {
    if (!r.ok) return toast(r.error || "Could not load the stages.", "error");
    spec = r; render(); runPreview();
  }).catch((e) => toast(String(e), "error"));
}

app.registerExtension({
  name: "Onyx.PostStudio",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "OnyxPostStudio") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      setupPostStudio(this);
      return r;
    };
  },
});
