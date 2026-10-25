// Onyx Glass Lab — "obsidian" node theme preview, calibrated on onyx-ai.eu.
// Targets ONLY the OnyxGlassLabDemo node. Everything is namespaced (ogl-*)
// so it can run next to the real Onyx pack without touching it.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "OnyxGlassLabDemo";

// ── Palette — the only block to edit ────────────────────────────────────────
// Monochrome obsidian like the site: near-black surfaces in layers, 1px white
// hairlines, white glow. The pale steel blue is the single accent and only
// shows on selection and primary actions.
const PALETTE = {
    canvasInk:   "#0a0a0c",
    body:        "#101013",   // node body (site card)
    well:        "#141418",   // nested panels (site inner card)
    title:       "#0d0d10",   // title band
    text:        "#f2f2f4",
    muted:       "#9a9ba3",
    faint:       "#5e5f68",
    line:        "rgba(255,255,255,0.10)",
    lineStrong:  "rgba(255,255,255,0.18)",
    highlight:   "rgba(255,255,255,0.06)",
    accent:      "#b9c5e6",
    accentRGB:   "185, 197, 230",
    ctaTop:      "#e6eaf6",
    ctaBottom:   "#b7c3e5",
    ctaInk:      "#0b0b0f",
    ok:          "#9fd4a8",
    warn:        "#e0c380",
    err:         "#e5737f",
    errRGB:      "229, 115, 127",
    ledOff:      "#3a3b42",
    // Background motif strength (0 = flat). Stars + one orbital arc.
    texture:     1,
};

// Pastel family so slot colours stay readable but never shout.
const SLOT_COLORS = {
    IMAGE: "#b9c5e6", MASK: "#c9b9e6", LATENT: "#e6b9d2", STRING: "#e6dcb9",
    MODEL: "#b9e6c4", INT: "#e6cbb9", FLOAT: "#e6cbb9", CLIP: "#b9dbe6",
    VAE: "#d2b9e6", CONDITIONING: "#e6c0b9", AUDIO: "#e6b9d2", VIDEO: "#b9e6dc",
};
const SLOT_FALLBACK = "#9a9ba3";

const REDUCED_MOTION = typeof matchMedia === "function"
    && matchMedia("(prefers-reduced-motion: reduce)").matches;
const FRAME_MS = 1000 / 30;
const SUPPORTS_ZOOM = typeof CSS !== "undefined" && CSS.supports?.("zoom", "1");

// ── Styles ──────────────────────────────────────────────────────────────────
function injectStyles() {
    if (document.getElementById("ogl-style")) return;
    const P = PALETTE, a = P.accentRGB, r = P.errRGB, t = P.texture;
    const css = `
.ogl-root{position:fixed;inset:0;pointer-events:none;z-index:2;overflow:hidden}
.ogl{position:absolute;left:0;top:0;pointer-events:none;will-change:transform}
.ogl__in{position:relative;transform-origin:0 0;border-radius:12px;overflow:hidden;
  box-shadow:0 0 0 1px ${P.line} inset,0 1px 0 ${P.highlight} inset,0 14px 34px rgba(0,0,0,.5);transition:box-shadow 160ms ease}
.ogl.is-selected .ogl__in{box-shadow:0 0 0 1px rgba(255,255,255,.32) inset,0 1px 0 rgba(255,255,255,.10) inset,0 0 0 1px rgba(${a},.25),0 0 22px rgba(255,255,255,.10),0 14px 34px rgba(0,0,0,.5)}
.ogl.is-error .ogl__in{box-shadow:0 0 0 1px rgba(${r},.55) inset,0 0 0 1px rgba(${r},.18),0 0 18px rgba(${r},.14),0 14px 34px rgba(0,0,0,.5)}
/* Motif: scattered star dots + one orbital arc, like the site background. */
.ogl__tex{position:absolute;inset:0;pointer-events:none;opacity:${t};
  background-image:
    radial-gradient(rgba(255,255,255,.07) .7px,transparent 1px),
    radial-gradient(rgba(255,255,255,.05) .7px,transparent 1px);
  background-size:38px 38px,61px 61px;background-position:7px 11px,23px 29px}
.ogl__arc{position:absolute;width:220%;aspect-ratio:1;right:-150%;top:-60%;border-radius:50%;border:1px solid rgba(255,255,255,.045);pointer-events:none;opacity:${t}}
.ogl__arc--2{width:160%;right:auto;left:-120%;top:40%;border-color:rgba(255,255,255,.035)}
/* Title band: slightly lighter top, hairline under. Title text is drawn by LiteGraph below. */
.ogl__head{position:absolute;left:0;right:0;top:0;height:var(--ogl-title,20px);pointer-events:none;
  background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,0));border-bottom:1px solid ${P.line}}
.ogl__chip{position:absolute;right:7px;top:3px;display:inline-flex;align-items:center;gap:6px;padding:2px 7px;border-radius:999px;
  background:#0b0b0e;border:1px solid ${P.line};font:600 8.5px/1 ui-monospace,"JetBrains Mono",Consolas,monospace;letter-spacing:.14em;color:${P.muted};white-space:nowrap}
.ogl.is-selected .ogl__chip{color:${P.text};border-color:${P.lineStrong}}
.ogl.is-error .ogl__chip{border-color:rgba(${r},.5)}
.ogl__led{width:6px;height:6px;border-radius:50%;background:${P.ledOff};transition:background 200ms,box-shadow 200ms}
.ogl__led.ok{background:${P.ok};box-shadow:0 0 6px rgba(159,212,168,.55)}
.ogl__led.warn{background:${P.warn};box-shadow:0 0 6px rgba(224,195,128,.55)}
.ogl__led.err{background:${P.err};box-shadow:0 0 6px rgba(${r},.6)}
.ogl__lock{opacity:.7;font-size:9px;padding-left:6px;border-left:1px solid ${P.line}}
/* Bottom state rail — the site's thin progress bar. */
.ogl__rail{position:absolute;left:10px;right:10px;bottom:5px;height:2px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;pointer-events:none}
.ogl__rail i{position:absolute;inset:0;border-radius:inherit;background:rgba(255,255,255,.28);transform-origin:left;transition:background 200ms}
.ogl.is-selected .ogl__rail i{background:${P.accent}}
.ogl.is-error .ogl__rail i{background:${P.err}}
.ogl.is-running .ogl__rail i{background:linear-gradient(90deg,rgba(255,255,255,.1),#fff,rgba(255,255,255,.1));background-size:200% 100%;animation:ogl-run 1.4s linear infinite}
@keyframes ogl-run{from{background-position:200% 0}to{background-position:-200% 0}}
.ogl.is-reduced .ogl__rail i{animation:none}
.ogl__err{position:absolute;left:8px;right:8px;bottom:12px;display:none;padding:6px 9px;border-radius:8px;background:#15090b;
  border:1px solid rgba(${r},.45);color:#f1c9ce;font:500 10px/1.35 ui-monospace,"JetBrains Mono",Consolas,monospace;overflow:hidden;max-height:46%}
.ogl.is-error .ogl__err{display:block}
.ogl.is-collapsed .ogl__err,.ogl.is-collapsed .ogl__rail,.ogl.is-collapsed .ogl__arc,.ogl.is-collapsed .ogl__tex{display:none}

/* ── Glass widget library (DOM widgets) — the site's nested card ── */
.oglw{box-sizing:border-box;width:100%;padding:10px 10px 8px;display:grid;gap:9px;color:${P.text};
  font:500 11px/1.3 Inter,"Segoe UI",system-ui,sans-serif;background:${P.well};border:1px solid ${P.line};border-radius:10px;
  box-shadow:0 1px 0 ${P.highlight} inset}
.oglw *{box-sizing:border-box}
.oglw__row{display:flex;align-items:center;gap:10px;min-width:0}
.oglw__label{font:500 9px/1 ui-monospace,"JetBrains Mono",Consolas,monospace;letter-spacing:.12em;text-transform:uppercase;color:${P.faint};min-width:64px}
.oglw-seg{display:inline-flex;padding:2px;gap:2px;border-radius:8px;background:#0b0b0e;border:1px solid ${P.line}}
.oglw-seg button{border:0;background:transparent;color:${P.muted};padding:5px 10px;border-radius:6px;font:600 10px/1 inherit;letter-spacing:.02em;cursor:pointer}
.oglw-seg button:hover{color:${P.text}}
.oglw-seg button.on{background:rgba(255,255,255,.09);color:${P.text};box-shadow:0 0 0 1px ${P.lineStrong} inset}
.oglw-combo{width:100%;appearance:none;-webkit-appearance:none;padding:6px 26px 6px 9px;border-radius:8px;color:${P.text};
  background:#0b0b0e;border:1px solid ${P.line};font:inherit;outline:none;cursor:pointer}
.oglw-combo:focus{border-color:${P.lineStrong};box-shadow:0 0 0 2px rgba(${a},.15)}
.oglw-combo-wrap{position:relative;flex:1;min-width:0}
.oglw-combo-wrap::after{content:"";position:absolute;right:10px;top:50%;width:6px;height:6px;border-right:1.5px solid ${P.muted};border-bottom:1.5px solid ${P.muted};transform:translateY(-70%) rotate(45deg);pointer-events:none}
.oglw-slider{flex:1;min-width:0;display:flex;align-items:center;gap:8px}
.oglw-slider input{-webkit-appearance:none;appearance:none;flex:1;height:2px;border-radius:999px;background:rgba(255,255,255,.12);outline:none}
.oglw-slider input::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.4),0 0 8px rgba(255,255,255,.35);cursor:pointer}
.oglw-slider input::-moz-range-thumb{width:12px;height:12px;border:0;border-radius:50%;background:#fff;box-shadow:0 0 8px rgba(255,255,255,.35);cursor:pointer}
.oglw-slider output{min-width:36px;text-align:right;font:500 10px/1 ui-monospace,"JetBrains Mono",Consolas,monospace;font-variant-numeric:tabular-nums;color:${P.text}}
.oglw-input{flex:1;min-width:0;padding:6px 9px;border-radius:8px;color:${P.text};background:#0b0b0e;border:1px solid ${P.line};font:inherit;outline:none}
.oglw-input::placeholder{color:${P.faint}}
.oglw-input:focus{border-color:${P.lineStrong};box-shadow:0 0 0 2px rgba(${a},.15)}
.oglw-switch{position:relative;width:32px;height:18px;flex:none;cursor:pointer}
.oglw-switch input{opacity:0;width:0;height:0;position:absolute}
.oglw-switch i{position:absolute;inset:0;border-radius:999px;background:#0b0b0e;border:1px solid ${P.line};transition:background 160ms}
.oglw-switch i::after{content:"";position:absolute;left:2px;top:2px;width:12px;height:12px;border-radius:50%;background:${P.muted};transition:transform 160ms,background 160ms}
.oglw-switch input:checked+i{background:rgba(255,255,255,.12);border-color:${P.lineStrong}}
.oglw-switch input:checked+i::after{transform:translateX(14px);background:#fff;box-shadow:0 0 6px rgba(255,255,255,.4)}
.oglw-btn{border:0;border-radius:8px;padding:7px 13px;font:600 10.5px/1 inherit;cursor:pointer;
  background:linear-gradient(180deg,${P.ctaTop},${P.ctaBottom});color:${P.ctaInk};box-shadow:0 1px 0 rgba(255,255,255,.35) inset,0 4px 14px rgba(0,0,0,.35)}
.oglw-btn:hover{filter:brightness(1.04)}
.oglw-btn.ghost{background:#0b0b0e;color:${P.text};border:1px solid ${P.line};box-shadow:none}
.oglw__foot{display:flex;justify-content:space-between;align-items:center;padding-top:6px;border-top:1px solid ${P.line};
  font:500 9px/1 ui-monospace,"JetBrains Mono",Consolas,monospace;letter-spacing:.08em;color:${P.faint}}
.oglw__foot b{color:${P.text};font-weight:500}
`;
    const style = document.createElement("style");
    style.id = "ogl-style";
    style.textContent = css;
    document.head.appendChild(style);
}

// ── Overlay ─────────────────────────────────────────────────────────────────
let root = null;
function ensureRoot() {
    if (!root) {
        root = document.createElement("div");
        root.className = "ogl-root";
        document.body.appendChild(root);
    }
    return root;
}

class Overlay {
    constructor() {
        const el = document.createElement("div");
        el.className = "ogl" + (REDUCED_MOTION ? " is-reduced" : "");
        el.innerHTML = `<div class="ogl__in">
            <div class="ogl__tex"></div><div class="ogl__arc"></div><div class="ogl__arc ogl__arc--2"></div>
            <div class="ogl__head"></div>
            <div class="ogl__chip"><span class="ogl__led" title="Onyx session: not synced"></span><span>ONYX</span><span class="ogl__lock">🔒</span></div>
            <div class="ogl__err"></div>
            <div class="ogl__rail"><i></i></div></div>`;
        this.el = el;
        this.inner = el.firstElementChild;
        this.led = el.querySelector(".ogl__led");
        this.err = el.querySelector(".ogl__err");
        this.last = {};
        ensureRoot().appendChild(el);
    }
    update(s) {
        const el = this.el, L = this.last;
        if (!s.visible) { if (el.style.display !== "none") el.style.display = "none"; return; }
        if (el.style.display === "none") el.style.display = "";
        const tf = `translate(${s.x.toFixed(1)}px,${s.y.toFixed(1)}px)`;
        if (L.tf !== tf) { el.style.transform = tf; L.tf = tf; }
        const z = s.scale.toFixed(3);
        if (L.z !== z) {
            if (SUPPORTS_ZOOM) this.inner.style.zoom = z;
            else this.inner.style.transform = `scale(${z})`;
            L.z = z;
        }
        const w = s.w + "px", h = (s.collapsed ? s.titleH : s.h) + "px", th = s.titleH + "px";
        if (L.w !== w) { this.inner.style.width = w; L.w = w; }
        if (L.h !== h) { this.inner.style.height = h; L.h = h; }
        if (L.th !== th) { this.inner.style.setProperty("--ogl-title", th); L.th = th; }
        const flags = { "is-selected": s.selected, "is-error": !!s.error, "is-collapsed": s.collapsed, "is-running": s.running };
        for (const k in flags) if (L[k] !== flags[k]) { el.classList.toggle(k, flags[k]); L[k] = flags[k]; }
        if (L.lic !== s.licence) {
            this.led.className = "ogl__led " + ({ ok: "ok", warn: "warn", err: "err" }[s.licence] || "");
            this.led.title = { ok: "Onyx session: active", warn: "Onyx session: stale", err: "Onyx session: invalid" }[s.licence] || "Onyx session: not synced";
            L.lic = s.licence;
        }
        const t = s.error ? (s.error.code ? s.error.code + " · " : "") + s.error.message : "";
        if (L.err !== t) { this.err.textContent = t; L.err = t; }
    }
    dispose() { this.el.remove(); }
}

// ── State: licence LED, running node, node errors ───────────────────────────
let licence = "off";
async function pollLicence() {
    try {
        const r = await api.fetchApi("/onyx_session/profile_status");
        if (r.ok) { const d = await r.json(); if (d?.state) licence = d.state; }
    } catch (_) { /* endpoint absent without the real pack: LED stays off */ }
}
pollLicence();
setInterval(pollLicence, 30000);

let runningId = null;
api.addEventListener("executing", (ev) => {
    const d = ev.detail;
    runningId = d == null ? null : Number(typeof d === "object" ? d.node : d);
});
api.addEventListener("execution_success", () => { runningId = null; });
api.addEventListener("execution_interrupted", () => { runningId = null; });

const nodeErrors = new Map();
api.addEventListener("execution_start", () => { nodeErrors.clear(); runningId = null; });
api.addEventListener("execution_error", (ev) => {
    runningId = null;
    const d = ev.detail || {};
    const id = d.node_id ?? d.node;
    if (id == null) return;
    const msg = String(d.exception_message || d.exception_type || "");
    const m = /\bSS-20\d\b/.exec(msg);
    nodeErrors.set(Number(id), { code: m ? m[0] : "", message: msg });
});

// ── Loop ────────────────────────────────────────────────────────────────────
const overlays = new Map();
let last = 0, started = false;
function startLoop() {
    if (started) return;
    started = true;
    const tick = (t) => {
        requestAnimationFrame(tick);
        if (t - last < FRAME_MS) return;
        last = t;
        try { update(); } catch (e) { console.warn("[GlassLab]", e); }
    };
    requestAnimationFrame(tick);
}

function update() {
    const canvas = app?.canvas, graph = canvas?.graph || app?.graph, el = canvas?.canvas;
    if (!canvas || !graph || !el) return;
    const scale = canvas.ds?.scale ?? 1, off = canvas.ds?.offset ?? [0, 0];
    const rect = el.getBoundingClientRect();
    const titleH = (typeof LiteGraph !== "undefined" && LiteGraph?.NODE_TITLE_HEIGHT) || 20;
    const selected = canvas.selected_nodes || {};
    const seen = new Set();
    for (const node of graph._nodes || []) {
        if (node?.comfyClass !== TARGET) continue;
        seen.add(node);
        let ov = overlays.get(node);
        if (!ov) { ov = new Overlay(); overlays.set(node, ov); }
        const w = node.size[0], h = node.size[1] + titleH;
        const x = rect.left + (off[0] + node.pos[0]) * scale;
        const y = rect.top + (off[1] + node.pos[1] - titleH) * scale;
        const visible = x + w * scale > rect.left && x < rect.right && y + h * scale > rect.top && y < rect.bottom;
        ov.update({
            x, y, w, h, titleH, scale, visible,
            selected: !!selected[node.id],
            running: runningId === Number(node.id),
            error: nodeErrors.get(node.id) || null,
            licence,
            collapsed: !!node.flags?.collapsed,
        });
    }
    for (const [node, ov] of overlays) if (!seen.has(node)) { ov.dispose(); overlays.delete(node); }
}

// ── Demo DOM widget panel ───────────────────────────────────────────────────
function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

function buildWidgetPanel(node) {
    const W = Object.fromEntries((node.widgets || []).map((w) => [w.name, w]));
    const setW = (name, v) => { const w = W[name]; if (!w) return; w.value = v; w.callback?.(v); node.graph?.setDirtyCanvas(true, true); };
    const panel = el("div", "oglw");
    for (const ev of ["pointerdown", "mousedown", "wheel", "keydown"]) panel.addEventListener(ev, (e) => e.stopPropagation());

    const r1 = el("div", "oglw__row");
    r1.append(el("span", "oglw__label", "mode"));
    const seg = el("div", "oglw-seg");
    const segBtns = {};
    for (const m of ["generate", "edit", "faceswap"]) {
        const b = el("button", "", m);
        b.type = "button";
        b.addEventListener("click", () => { setW("mode", m); paintSeg(); });
        segBtns[m] = b; seg.append(b);
    }
    const paintSeg = () => { for (const m in segBtns) segBtns[m].classList.toggle("on", W.mode?.value === m); };
    paintSeg();
    r1.append(seg);

    const r2 = el("div", "oglw__row");
    r2.append(el("span", "oglw__label", "preset"));
    const cw = el("div", "oglw-combo-wrap");
    const combo = el("select", "oglw-combo");
    for (const o of ["Casual phone", "Fashion editorial", "Luxury lifestyle", "Playful social"]) { const op = el("option", "", o); op.value = o; combo.append(op); }
    cw.append(combo); r2.append(cw);

    const r3 = el("div", "oglw__row");
    r3.append(el("span", "oglw__label", "strength"));
    const sl = el("div", "oglw-slider");
    const range = el("input"); range.type = "range"; range.min = "0"; range.max = "1"; range.step = "0.01"; range.value = String(W.strength?.value ?? 0.65);
    const out = el("output", "", Number(range.value).toFixed(2));
    range.addEventListener("input", () => { out.textContent = Number(range.value).toFixed(2); setW("strength", Number(range.value)); });
    sl.append(range, out); r3.append(sl);

    const r4 = el("div", "oglw__row");
    r4.append(el("span", "oglw__label", "trigger"));
    const inp = el("input", "oglw-input"); inp.type = "text"; inp.placeholder = "ohwx woman";
    r4.append(inp);

    const r5 = el("div", "oglw__row");
    r5.append(el("span", "oglw__label", "metadata"));
    const sw = el("label", "oglw-switch");
    const chk = el("input"); chk.type = "checkbox"; chk.checked = !!W.keep_metadata?.value;
    chk.addEventListener("change", () => setW("keep_metadata", chk.checked));
    sw.append(chk, el("i"));
    r5.append(sw, el("span", "", "keep EXIF"));

    const r6 = el("div", "oglw__row");
    const gen = el("button", "oglw-btn", "Generate"); gen.type = "button";
    gen.addEventListener("click", () => app.queuePrompt?.(0, 1));
    const reset = el("button", "oglw-btn ghost", "Reset"); reset.type = "button";
    reset.addEventListener("click", () => {
        setW("mode", "generate"); setW("strength", 0.65); setW("keep_metadata", false); setW("simulate_sync_error", false);
        range.value = "0.65"; out.textContent = "0.65"; chk.checked = false; paintSeg();
    });
    r6.append(gen, reset);

    const foot = el("div", "oglw__foot");
    const fl = el("span", "", "Files "); fl.append(el("b", "", "60 / 60"));
    const fr = el("span", "", "Run · "); fr.append(el("b", "", "21:20:00"));
    foot.append(fl, fr);

    panel.append(r1, r2, r3, r4, r5, r6, foot);
    return { panel, sync: () => { paintSeg(); range.value = String(W.strength?.value ?? 0.65); out.textContent = Number(range.value).toFixed(2); chk.checked = !!W.keep_metadata?.value; } };
}

function applySlotColors(node) {
    const paint = (s) => { if (!s) return; const c = SLOT_COLORS[String(s.type || "").toUpperCase()] || SLOT_FALLBACK; s.color_on = c; s.color_off = c; };
    (node.inputs || []).forEach(paint);
    (node.outputs || []).forEach(paint);
}

window.OnyxGlassLab = { palette: PALETTE, slotColors: SLOT_COLORS };

app.registerExtension({
    name: "onyx.GlassLab",
    async init() { injectStyles(); startLoop(); },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET) return;
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = orig?.apply(this, arguments);
            const { panel, sync } = buildWidgetPanel(this);
            const panelHeight = () => Math.max(270, (panel.scrollHeight || 0) + 16);
            const widget = this.addDOMWidget("glass_widgets", "div", panel, {
                serialize: false,
                getMinHeight: panelHeight,
                hideOnZoom: false,
            });
            widget.computeSize = (w) => [w, panelHeight()];
            requestAnimationFrame(() => { this.setSize(this.computeSize()); this.graph?.setDirtyCanvas(true, true); });
            const origConfigure = this.onConfigure;
            this.onConfigure = function () { const r2 = origConfigure?.apply(this, arguments); sync(); return r2; };
            return r;
        };
    },
    nodeCreated(node) {
        if (node?.comfyClass !== TARGET) return;
        node.color = PALETTE.title;
        node.bgcolor = PALETTE.body;
        node.boxcolor = PALETTE.accent;
        node.title_color = PALETTE.text;
        applySlotColors(node);
    },
});
