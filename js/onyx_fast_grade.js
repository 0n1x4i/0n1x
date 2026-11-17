// Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
// Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
//
// Onyx Fast Video Grade — panel layout: settings on the left, a large live viewer on the right
// (any frame of the timeline, re-rendered with the exact export grade on every change).
// The node resizes freely; the viewer also goes full screen.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { makeCompact } from "./onyx_compact.js";

const NODE = "OnyxFastVideoGrade";
const NODE_SIZE = [1180, 760];
const PANEL_MIN_H = 420;
const VIDEO_EXT = /\.(mp4|mov|mkv|webm|avi|m4v|gif)(\s*\[(input|output|temp)\])?$/i;
const SOURCE_WIDGETS = ["video_path", "file", "video", "path", "filename", "file_path", "video_file"];
const PREVIEW_PARAMS = ["exposure", "contrast", "highlights", "shadows", "whites", "blacks", "temperature", "tint",
  "saturation", "vibrance", "lut", "lut_strength", "grain_amount", "grain_size", "grain_color", "seed"];

const GROUPS = [
  { title: "Light", items: [
    { k: "exposure", label: "Exposure", min: -3, max: 3, step: 0.05, def: 0 },
    { k: "contrast", label: "Contrast", def: 0 },
    { k: "highlights", label: "Highlights", def: 0 },
    { k: "shadows", label: "Shadows", def: 0 },
    { k: "whites", label: "Whites", def: 0 },
    { k: "blacks", label: "Blacks", def: 0 },
  ] },
  { title: "Color", items: [
    { k: "temperature", label: "Temperature", def: 0, track: "linear-gradient(90deg,#5d8bd6,#8a8c96,#e0a85c)" },
    { k: "tint", label: "Tint", def: 0, track: "linear-gradient(90deg,#6dbf72,#8a8c96,#c86cc0)" },
    { k: "saturation", label: "Saturation", def: 0 },
    { k: "vibrance", label: "Vibrance", def: 0 },
  ] },
  { title: "Look", items: [
    { k: "lut", label: "LUT", kind: "select" },
    { k: "lut_strength", label: "Strength", min: 0, max: 1, step: 0.05, def: 1 },
  ] },
  { title: "Grain", items: [
    { k: "grain_amount", label: "Amount", min: 0, max: 100, step: 1, def: 0 },
    { k: "grain_size", label: "Size", min: 0.5, max: 4, step: 0.1, def: 1.2 },
    { k: "grain_color", label: "Color", kind: "bool", def: false },
    { k: "seed", label: "Seed", kind: "seed", def: 0 },
  ] },
  { title: "Export", items: [
    { k: "encoder", label: "Encoder", kind: "select" },
    { k: "quality", label: "Quality", min: 10, max: 35, step: 1, def: 18, int: true },
    { k: "filename_prefix", label: "Name", kind: "text" },
    { k: "video_path", label: "Video path", kind: "text" },
    { k: "frame_rate", label: "Image fps", min: 1, max: 120, step: 0.01, def: 24 },
  ] },
];

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "value") e.value = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}

function injectStyle() {
  if (document.getElementById("onyx-fast-grade-style")) return;
  const css = `
.ofg{--bg:#0d0d10;--well:#111114;--p2:#16161b;--ln:rgba(255,255,255,.10);--ln2:rgba(255,255,255,.18);--tx:#f2f2f4;--dim:#9a9ba3;--faint:#5e5f68;--ac:#b9c5e6;
  display:flex;flex-direction:column;width:100%;height:100%;box-sizing:border-box;background:var(--bg);color:var(--tx);
  border:1px solid var(--ln);border-radius:12px;overflow:hidden;font:12px/1.4 Inter,"Segoe UI",system-ui,sans-serif}
.ofg *{box-sizing:border-box}
.ofg:fullscreen{border-radius:0;border:none}
.ofg-head{display:flex;align-items:center;gap:10px;padding:9px 12px;border-bottom:1px solid var(--ln);background:#0a0a0d;flex:0 0 auto}
.ofg-logo{font:800 12px Inter,system-ui,sans-serif;letter-spacing:.18em;color:var(--tx);white-space:nowrap}
.ofg-title{font-weight:600;color:var(--dim);white-space:nowrap}
.ofg-src{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:11px ui-monospace,"JetBrains Mono",Consolas,monospace;color:var(--faint);text-align:right}
.ofg-body{flex:1 1 auto;min-height:0;display:flex}
.ofg-side{flex:0 0 290px;min-width:0;overflow-y:auto;overflow-x:hidden;border-right:1px solid var(--ln);background:var(--well);padding:8px 10px 14px}
.ofg-side::-webkit-scrollbar{width:6px}.ofg-side::-webkit-scrollbar-thumb{background:#2a2a32;border-radius:3px}
.ofg-grp{margin-top:6px}
.ofg-grp h4{margin:8px 0 4px;font:700 10px Inter,system-ui,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);display:flex;align-items:center}
.ofg-grp h4 .sp{flex:1}
.ofg-grp h4 button{all:unset;cursor:pointer;color:var(--faint);font:600 10px Inter,system-ui,sans-serif;letter-spacing:.04em;text-transform:none;padding:0 2px}
.ofg-grp h4 button:hover{color:var(--tx)}
.ofg-row{display:grid;grid-template-columns:78px 1fr 54px;align-items:center;gap:8px;padding:3px 0}
.ofg-row.wide{grid-template-columns:78px 1fr}
.ofg-row label{color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:default;user-select:none}
.ofg-row.changed label{color:var(--tx)}
.ofg-row input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:3px;border-radius:2px;background:#2b2b33;outline:none;cursor:pointer;margin:0}
.ofg-row input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:#e6eaf6;border:none;box-shadow:0 0 0 3px rgba(185,197,230,.18)}
.ofg-row input[type=range]::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:#e6eaf6;border:none}
.ofg-row input[type=number],.ofg-row input[type=text],.ofg-row select{width:100%;height:24px;padding:0 6px;border-radius:6px;border:1px solid var(--ln);background:#0c0c0f;color:var(--tx);
  font:11.5px ui-monospace,"JetBrains Mono",Consolas,monospace;outline:none}
.ofg-row input[type=number]{text-align:right;-moz-appearance:textfield}
.ofg-row input[type=number]::-webkit-inner-spin-button{display:none}
.ofg-row select{font-family:Inter,system-ui,sans-serif;cursor:pointer}
.ofg-row input:focus,.ofg-row select:focus{border-color:rgba(185,197,230,.55)}
.ofg-seed{display:flex;gap:6px}
.ofg-tog{width:34px;height:18px;border-radius:9px;background:#2b2b33;position:relative;cursor:pointer;border:1px solid var(--ln)}
.ofg-tog::after{content:"";position:absolute;top:2px;left:2px;width:12px;height:12px;border-radius:50%;background:#8a8c96;transition:left .12s}
.ofg-tog.on{background:#b7c3e5}.ofg-tog.on::after{left:18px;background:#0b0b0f}
.ofg-main{flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:8px;padding:10px}
.ofg-view{position:relative;flex:1 1 auto;min-height:0;background:#060608;border:1px solid var(--ln);border-radius:10px;overflow:hidden;
  display:flex;align-items:center;justify-content:center;cursor:zoom-in}
.ofg-view.zoomed{cursor:zoom-out}
.ofg-view img{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;display:block;user-select:none;-webkit-user-drag:none}
.ofg-view.zoomed img{width:100%;height:100%;max-width:none;max-height:none;object-fit:contain}
.ofg-msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:16px;text-align:center;color:var(--faint);font-size:12px}
.ofg-tag{position:absolute;top:10px;left:10px;padding:2px 9px;border-radius:999px;background:rgba(10,10,13,.75);border:1px solid var(--ln2);
  font:600 10px ui-monospace,"JetBrains Mono",Consolas,monospace;letter-spacing:.08em;color:var(--dim);text-transform:uppercase;pointer-events:none}
.ofg-tag.before{color:#e0c380;border-color:rgba(224,195,128,.45)}
.ofg-busy{position:absolute;top:11px;right:11px;width:11px;height:11px;border-radius:50%;border:1.5px solid rgba(255,255,255,.18);
  border-top-color:var(--ac);animation:ofgspin .7s linear infinite;opacity:0;transition:opacity .15s;pointer-events:none}
.ofg-busy.on{opacity:1}
@keyframes ofgspin{to{transform:rotate(360deg)}}
.ofg-bar{flex:0 0 auto;display:flex;align-items:center;gap:8px;padding:6px 8px;background:var(--well);border:1px solid var(--ln);border-radius:9px}
.ofg-btn{flex:0 0 auto;height:26px;min-width:28px;padding:0 10px;border-radius:7px;border:1px solid var(--ln2);background:var(--p2);color:var(--tx);
  font:600 11px Inter,system-ui,sans-serif;cursor:pointer;display:flex;align-items:center;justify-content:center;user-select:none}
.ofg-btn:hover{border-color:rgba(185,197,230,.6)}
.ofg-btn.on{background:linear-gradient(180deg,#e6eaf6,#b7c3e5);color:#0b0b0f;border-color:transparent}
.ofg-range{flex:1 1 auto;min-width:60px;-webkit-appearance:none;appearance:none;height:4px;border-radius:2px;background:#2b2b33;cursor:pointer;margin:0}
.ofg-range::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#e6eaf6;border:none;box-shadow:0 0 0 4px rgba(185,197,230,.18)}
.ofg-range::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#e6eaf6;border:none}
.ofg-tc{flex:0 0 auto;font:11.5px ui-monospace,"JetBrains Mono",Consolas,monospace;color:var(--dim);white-space:nowrap;min-width:120px;text-align:right}
.ofg-tc b{color:var(--tx);font-weight:500}
`;
  document.head.append(h("style", { id: "onyx-fast-grade-style" }, css));
}

const fmt = (sec) => {
  if (!isFinite(sec)) sec = 0;
  const m = Math.floor(sec / 60), s = sec - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
};

function hideWidget(w) {
  if (!w || w.__ofgHidden) return;
  w.__ofgHidden = true;
  w.hidden = true; w.computeSize = () => [0, -4]; w.draw = () => {};
  if (w.element) w.element.style.cssText = "display:none!important";
}

async function post(url, body) {
  return api.fetchApi(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

function setupGrade(node) {
  injectStyle();
  const W = (name) => node.widgets?.find((w) => w.name === name);
  const graph = () => node.graph || app.graph;
  const linkOf = (id) => { const g = graph(); return g.links?.[id] ?? g.links?.get?.(id); };

  // Where the video comes from: the node's own video_path, else a file widget upstream.
  function findSource() {
    const own = W("video_path");
    const ownLinked = node.inputs?.find((i) => i.name === "video_path" && i.link != null);
    if (own && !ownLinked && String(own.value || "").trim()) return String(own.value).trim();
    // images in: the frames to grade only exist after a run (the server keeps them)
    const linked = (n) => node.inputs?.some((i) => i.name === n && i.link != null);
    if (linked("images") && !linked("video") && !linked("video_path")) return "";
    const seen = new Set();
    const walk = (n, depth) => {
      if (!n || depth > 4 || seen.has(n.id)) return "";
      seen.add(n.id);
      for (const name of SOURCE_WIDGETS) {
        const w = n.widgets?.find((x) => x.name === name);
        if (w && typeof w.value === "string" && VIDEO_EXT.test(w.value.trim())) return w.value.trim();
      }
      for (const w of n.widgets || []) if (typeof w.value === "string" && VIDEO_EXT.test(w.value.trim())) return w.value.trim();
      for (const inp of n.inputs || []) {
        if (inp.link == null) continue;
        const l = linkOf(inp.link);
        const r = walk(l && graph().getNodeById(l.origin_id), depth + 1);
        if (r) return r;
      }
      return "";
    };
    for (const name of ["video", "video_info", "video_path"]) {
      const inp = node.inputs?.find((i) => i.name === name);
      if (!inp || inp.link == null) continue;
      const l = linkOf(inp.link);
      const r = walk(l && graph().getNodeById(l.origin_id), 0);
      if (r) return r;
    }
    return "";
  }

  // ── settings column ────────────────────────────────────────────────────────
  const setters = {};                 // key -> (value) => update the control from the widget
  const setValue = (k, v) => {
    const w = W(k);
    if (!w) return;
    w.value = v;
    try { w.callback?.(v, app.canvas, node); } catch { /* ignore */ }
    node.setDirtyCanvas?.(true, true);
  };
  const decimals = (step) => (String(step).split(".")[1] || "").length;

  function sliderRow(it) {
    const w = W(it.k);
    const min = it.min ?? w?.options?.min ?? -100, max = it.max ?? w?.options?.max ?? 100;
    const step = it.step ?? 1, dec = it.int ? 0 : decimals(step);
    const row = h("div", { class: "ofg-row" });
    const lbl = h("label", {}, it.label);
    const rng = h("input", { type: "range", min, max, step });
    if (it.track) rng.style.background = it.track;
    const num = h("input", { type: "number", min, max, step });
    const clamp = (x) => { let n = it.int ? parseInt(x, 10) : parseFloat(x); if (!isFinite(n)) n = it.def; return Math.min(max, Math.max(min, n)); };
    const show = (v) => { rng.value = v; num.value = Number(v).toFixed(dec); row.classList.toggle("changed", Math.abs(v - it.def) > 1e-9); };
    rng.addEventListener("input", () => { const v = clamp(rng.value); num.value = v.toFixed(dec); row.classList.toggle("changed", Math.abs(v - it.def) > 1e-9); setValue(it.k, v); });
    num.addEventListener("change", () => { const v = clamp(num.value); show(v); setValue(it.k, v); });
    lbl.addEventListener("dblclick", () => { show(it.def); setValue(it.k, it.def); });
    rng.addEventListener("dblclick", () => { show(it.def); setValue(it.k, it.def); });
    setters[it.k] = (v) => show(v ?? it.def);
    row.append(lbl, rng, num);
    return row;
  }
  function selectRow(it) {
    const w = W(it.k);
    const sel = h("select", {});
    const fillOpts = () => {
      const vals = w?.options?.values;
      const list = typeof vals === "function" ? vals() : (vals || []);
      sel.replaceChildren(...list.map((v) => h("option", { value: v }, v)));
    };
    fillOpts();
    sel.addEventListener("focus", fillOpts);
    sel.addEventListener("change", () => setValue(it.k, sel.value));
    setters[it.k] = (v) => { if (![...sel.options].some((o) => o.value === v)) fillOpts(); sel.value = v; };
    return h("div", { class: "ofg-row wide" }, h("label", {}, it.label), sel);
  }
  function boolRow(it) {
    const t = h("div", { class: "ofg-tog" });
    t.addEventListener("click", () => { const v = !t.classList.contains("on"); t.classList.toggle("on", v); setValue(it.k, v); });
    setters[it.k] = (v) => t.classList.toggle("on", !!v);
    return h("div", { class: "ofg-row wide" }, h("label", {}, it.label), t);
  }
  function seedRow(it) {
    const num = h("input", { type: "number", min: 0, max: 4294967295, step: 1 });
    num.addEventListener("change", () => { const v = Math.max(0, Math.min(4294967295, parseInt(num.value, 10) || 0)); num.value = v; setValue(it.k, v); });
    const dice = h("button", { class: "ofg-btn", style: "height:24px;padding:0 8px" }, "🎲");
    dice.addEventListener("click", () => { const v = Math.floor(Math.random() * 2 ** 32); num.value = v; setValue(it.k, v); });
    setters[it.k] = (v) => { num.value = v ?? 0; };
    return h("div", { class: "ofg-row wide" }, h("label", {}, it.label), h("div", { class: "ofg-seed" }, num, dice));
  }
  function textRow(it) {
    const inp = h("input", { type: "text", spellcheck: "false" });
    inp.addEventListener("change", () => setValue(it.k, inp.value));
    setters[it.k] = (v) => { if (document.activeElement !== inp) inp.value = v ?? ""; };
    return h("div", { class: "ofg-row wide" }, h("label", {}, it.label), inp);
  }
  const resetGroup = (g) => {
    for (const it of g.items) if ("def" in it && it.kind !== "seed") { setValue(it.k, it.def); setters[it.k]?.(it.def); }
  };
  const side = h("div", { class: "ofg-side" }, GROUPS.map((g) => {
    const canReset = g.items.some((it) => "def" in it && it.kind !== "seed") && g.title !== "Export";
    return h("div", { class: "ofg-grp" },
      h("h4", {}, g.title, h("span", { class: "sp" }), canReset ? h("button", { onclick: () => resetGroup(g) }, "Reset") : null),
      g.items.map((it) => it.kind === "select" ? selectRow(it) : it.kind === "bool" ? boolRow(it)
        : it.kind === "seed" ? seedRow(it) : it.kind === "text" ? textRow(it) : sliderRow(it)));
  }));
  const syncControls = () => { for (const [k, set] of Object.entries(setters)) set(W(k)?.value); };

  // ── viewer ─────────────────────────────────────────────────────────────────
  const img = h("img", { draggable: "false", alt: "" });
  const msg = h("div", { class: "ofg-msg" }, "Connect a video");
  const tag = h("div", { class: "ofg-tag" }, "After");
  const busyDot = h("div", { class: "ofg-busy" });
  const view = h("div", { class: "ofg-view" }, img, msg, tag, busyDot);
  const range = h("input", { class: "ofg-range", type: "range", min: 0, max: 0, step: 1, value: 0 });
  const tc = h("div", { class: "ofg-tc" });
  const prev = h("button", { class: "ofg-btn" }, "‹");
  const next = h("button", { class: "ofg-btn" }, "›");
  const before = h("button", { class: "ofg-btn" }, "Before");
  const full = h("button", { class: "ofg-btn" }, "⛶");
  const bar = h("div", { class: "ofg-bar" }, prev, range, next, tc, before, full);
  const main = h("div", { class: "ofg-main" }, view, bar);
  const srcLabel = h("div", { class: "ofg-src" });
  const head = h("div", { class: "ofg-head" }, h("span", { class: "ofg-logo" }, "ONYX"), h("span", { class: "ofg-title" }, "Fast Video Grade"), srcLabel);
  const root = h("div", { class: "ofg" }, head, h("div", { class: "ofg-body" }, side, main));
  for (const ev of ["wheel", "keydown"]) root.addEventListener(ev, (e) => e.stopPropagation());
  side.addEventListener("pointerdown", (e) => e.stopPropagation());
  main.addEventListener("pointerdown", (e) => e.stopPropagation());

  const st = { source: "", info: null, index: 0, zoom: false, cx: 0.5, cy: 0.5, before: false, sig: "", size: 1024 };
  tag.style.display = "none";
  const params = () => { const p = {}; for (const k of PREVIEW_PARAMS) { const w = W(k); if (w) p[k] = w.value; } return p; };
  const showMsg = (t) => { msg.textContent = t; msg.style.display = ""; img.style.display = "none"; tag.style.display = "none"; };
  const updateTc = () => {
    if (!st.info) { tc.textContent = "--:--.-- / --:--.--"; return; }
    const { fps, frames } = st.info;
    tc.replaceChildren(h("b", {}, fmt(st.index / fps)), ` / ${fmt(frames / fps)}`);
  };
  // render at the size the viewer really shows (sharp when the node is big or full screen)
  const viewPx = () => {
    // on-screen size (already includes the canvas zoom), in device pixels
    const r = view.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    return [Math.max(64, Math.round(r.width * dpr)), Math.max(64, Math.round(r.height * dpr))];
  };
  const wantedSize = () => {
    const [vw, vh] = viewPx();
    const iw = st.info?.width || 16, ih = st.info?.height || 9;
    const s = Math.min(vw / iw, vh / ih);                       // fitted image, in device px
    const longSide = Math.max(iw, ih) * s;
    return Math.max(512, Math.min(3072, Math.ceil(longSide / 256) * 256));
  };

  let busy = false, again = false, lastUrl = null;
  async function render() {
    if (!st.info) return;
    if (busy) { again = true; return; }
    busy = true; busyDot.classList.add("on");
    try {
      do {
        again = false;
        st.size = wantedSize();
        const [vw, vh] = viewPx();
        const r = await post("/onyx/fast_grade/frame", {
          node_id: String(node.id), path: st.source, index: st.index, params: params(), original: st.before,
          size: st.size, zoom: st.zoom, cx: st.cx, cy: st.cy, view_w: vw, view_h: vh,
        });
        if (!r.ok) { const j = await r.json().catch(() => ({})); showMsg(j.error || "Preview unavailable"); break; }
        const url = URL.createObjectURL(await r.blob());
        await new Promise((res) => { img.onload = img.onerror = res; img.src = url; });
        if (lastUrl) URL.revokeObjectURL(lastUrl);
        lastUrl = url;
        msg.style.display = "none"; img.style.display = ""; tag.style.display = "";
        tag.textContent = st.before ? "Before" : "After"; tag.classList.toggle("before", st.before);
      } while (again);
    } catch (e) {
      showMsg(String(e));
    } finally {
      busy = false; busyDot.classList.remove("on");
    }
  }

  async function loadSource(force = false) {
    const src = findSource();
    if (!force && src === st.source && st.info) return;
    st.source = src;
    try {
      const r = await post("/onyx/fast_grade/info", { node_id: String(node.id), path: src });
      const j = await r.json();
      if (!j.ok) { st.info = null; srcLabel.textContent = ""; updateTc(); return showMsg(r.status === 404 && !src
        ? (node.inputs?.some((i) => i.name === "images" && i.link != null) ? "Run once to load the frames" : "Connect a video")
        : (j.error || "Video unavailable")); }
      const changed = !st.info || st.info.name !== j.name || st.info.frames !== j.frames;
      st.info = j;
      srcLabel.textContent = `${j.name} · ${j.width}×${j.height} · ${Math.round(j.fps * 100) / 100} fps`;
      range.max = Math.max(0, j.frames - 1);
      if (changed) { st.index = Math.min(st.index, j.frames - 1); st.zoom = false; view.classList.remove("zoomed"); }
      range.value = st.index;
      updateTc();
      render();
    } catch (e) { st.info = null; showMsg(String(e)); }
  }

  // timeline
  let scrubTimer = null;
  range.addEventListener("input", () => {
    st.index = parseInt(range.value, 10) || 0; updateTc();
    clearTimeout(scrubTimer); scrubTimer = setTimeout(render, 30);
  });
  const step = (d) => {
    if (!st.info) return;
    st.index = Math.max(0, Math.min(st.info.frames - 1, st.index + d));
    range.value = st.index; updateTc(); render();
  };
  prev.addEventListener("click", () => step(-1));
  next.addEventListener("click", () => step(1));
  main.tabIndex = 0;
  main.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target !== range) return;
    if (e.key === "ArrowLeft") { step(e.shiftKey ? -10 : -1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { step(e.shiftKey ? 10 : 1); e.preventDefault(); }
  });

  // before / after: hold the button (or the B key)
  const setBefore = (v) => { if (st.before === v) return; st.before = v; before.classList.toggle("on", v); render(); };
  before.addEventListener("pointerdown", (e) => { e.preventDefault(); setBefore(true); });
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) before.addEventListener(ev, () => setBefore(false));
  root.addEventListener("keydown", (e) => { if ((e.key === "b" || e.key === "B") && e.target.tagName !== "INPUT") setBefore(true); });
  root.addEventListener("keyup", (e) => { if (e.key === "b" || e.key === "B") setBefore(false); });

  // full screen
  full.addEventListener("click", () => {
    if (document.fullscreenElement === root) document.exitFullscreen?.();
    else root.requestFullscreen?.().catch(() => {});
  });
  document.addEventListener("fullscreenchange", () => {
    full.classList.toggle("on", document.fullscreenElement === root);
    setTimeout(render, 80);
  });

  // click: 1:1 zoom on that spot, click again: fit
  view.addEventListener("click", (e) => {
    if (!st.info || img.style.display === "none") return;
    if (st.zoom) { st.zoom = false; view.classList.remove("zoomed"); return render(); }
    const r = img.getBoundingClientRect();
    if (!r.width || !r.height) return;
    st.cx = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    st.cy = Math.max(0, Math.min(1, (e.clientY - r.top) / r.height));
    st.zoom = true; view.classList.add("zoomed"); render();
  });

  // a bigger viewer gets a sharper frame
  let lastSize = 0, roTimer = null;
  new ResizeObserver(() => {
    clearTimeout(roTimer);
    roTimer = setTimeout(() => { const s = wantedSize(); if (st.info && (s !== lastSize || st.zoom)) { lastSize = s; render(); } }, 200);
  }).observe(view);

  // settings: watched by value, so the panel, loaded workflows and pasted nodes all refresh
  const sig = () => JSON.stringify(params());
  const poll = setInterval(() => {
    if (!node.graph) return;
    const s = sig();
    if (s !== st.sig) { st.sig = s; syncControls(); render(); }
    if (findSource() !== st.source) loadSource();
  }, 120);

  // ── mount ──────────────────────────────────────────────────────────────────
  const hideAll = () => { for (const w of node.widgets || []) if (!w.element || w.element !== root) hideWidget(w); };
  hideAll();
  setTimeout(hideAll, 300);
  node.addDOMWidget("grade_panel", "ONYX_FAST_GRADE", root, {
    serialize: false, getValue: () => undefined, setValue: () => {}, getMinHeight: () => PANEL_MIN_H,
  });
  // the seed must not move after each run, or the export grain would not be the one previewed
  const ctrl = node.widgets?.find((w) => w.name === "control_after_generate" || w.name === "control_after_generate_seed");
  if (ctrl && ctrl.value === "randomize") ctrl.value = "fixed";
  if (typeof node.setSize === "function") node.setSize([...NODE_SIZE]); else node.size = [...NODE_SIZE];
  makeCompact(node, head);

  const origConn = node.onConnectionsChange?.bind(node);
  node.onConnectionsChange = function (...a) { origConn?.(...a); setTimeout(() => loadSource(true), 50); };
  let ranThisNode = false;
  const onExecuting = ({ detail }) => {
    const id = detail && typeof detail === "object" ? detail.node : detail;
    if (id != null) { if (String(id) === String(node.id)) ranThisNode = true; return; }
    if (ranThisNode) loadSource(true);
    ranThisNode = false;
  };
  api.addEventListener("executing", onExecuting);
  const origRemoved = node.onRemoved?.bind(node);
  node.onRemoved = function () {
    clearInterval(poll); api.removeEventListener("executing", onExecuting);
    if (lastUrl) URL.revokeObjectURL(lastUrl);
    origRemoved?.();
  };
  const origConfigure = node.onConfigure?.bind(node);
  node.onConfigure = function (info) {
    origConfigure?.(info); hideAll(); syncControls();
    if (Array.isArray(info?.size) && info.size[0] < 900) node.setSize?.([...NODE_SIZE]);     // older inline layout
    setTimeout(() => loadSource(true), 100);
  };

  syncControls(); updateTc();
  setTimeout(() => loadSource(true), 100);
}

app.registerExtension({
  name: "Onyx.FastVideoGrade",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      setupGrade(this);
      return r;
    };
  },
});
