// Onyx Glass Lab — sober glass theme preview.
// Targets ONLY the OnyxGlassLabDemo node. Everything is namespaced (ogl-*)
// so it can run next to the real Onyx pack without touching it.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "OnyxGlassLabDemo";

// ── Palette — the only block to edit ────────────────────────────────────────
// Neutral graphite body, one accent. Accent is provisional until the
// onyx-ai.eu hex codes are confirmed.
const PALETTE = {
    accent:      "#2fd8f0",
    accentRGB:   "47, 216, 240",
    body:        "#0e1012",
    title:       "#141719",
    titleText:   "#e8ebee",
    chipBg:      "rgba(8, 10, 12, 0.72)",
    chipText:    "#c9d1d6",
    line:        "rgba(255, 255, 255, 0.06)",
    red:         "#ff5a6a",
    redRGB:      "255, 90, 106",
    amber:       "#ffc24b",
    green:       "#5be39a",
    ledOff:      "#3c4247",
};

// Ambient blobs are off for the sober variant. Flip to true to get one slow
// drifting highlight in the top-left corner.
const BLOBS = false;

const SLOT_COLORS = {
    IMAGE: "#4fd6ea", MASK: "#a68cff", LATENT: "#f28bd1", STRING: "#f2cf6b",
    MODEL: "#8be58b", INT: "#ffb469", FLOAT: "#ffb469", CLIP: "#9ad9ff",
    VAE: "#b4a0ff", CONDITIONING: "#ff9a7a", AUDIO: "#f28bd1", VIDEO: "#8bf2d1",
};
const SLOT_FALLBACK = "#8a949c";

const REDUCED_MOTION = typeof matchMedia === "function"
    && matchMedia("(prefers-reduced-motion: reduce)").matches;
const FRAME_MS = 1000 / 30;
const SUPPORTS_ZOOM = typeof CSS !== "undefined" && CSS.supports?.("zoom", "1");

// ── Styles ──────────────────────────────────────────────────────────────────
function injectStyles() {
    if (document.getElementById("ogl-style")) return;
    const a = PALETTE.accentRGB, r = PALETTE.redRGB;
    const css = `
.ogl-root{position:fixed;inset:0;pointer-events:none;z-index:2;overflow:hidden}
.ogl{position:absolute;left:0;top:0;pointer-events:none;will-change:transform}
/* Inner box is scaled with CSS zoom (not transform) so text stays crisp at any canvas zoom. */
.ogl__in{position:relative;transform-origin:0 0;border-radius:9px;
  box-shadow:0 0 0 1px ${PALETTE.line} inset,0 0 0 1px rgba(${a},.18);transition:box-shadow 160ms ease}
.ogl.is-selected .ogl__in{box-shadow:0 0 0 1px rgba(${a},.9) inset,0 0 0 2px rgba(${a},.22),0 0 14px rgba(${a},.22)}
.ogl.is-error .ogl__in{box-shadow:0 0 0 1px rgba(${r},.7) inset,0 0 0 2px rgba(${r},.16),0 0 14px rgba(${r},.22)}
.ogl__frame{position:absolute;inset:0;border-radius:inherit;overflow:hidden;pointer-events:none}
.ogl__accent{position:absolute;left:12%;right:12%;top:0;height:1px;background:linear-gradient(90deg,transparent,rgba(${a},.7),transparent)}
.ogl.is-error .ogl__accent{background:linear-gradient(90deg,transparent,rgba(${r},.8),transparent)}
.ogl__blob{position:absolute;width:46%;height:70%;left:-10%;top:-30%;border-radius:50%;filter:blur(22px);opacity:.5;
  background:radial-gradient(circle,rgba(${a},.22) 0%,transparent 70%);display:${BLOBS ? "block" : "none"}}
.ogl.is-anim .ogl__blob{animation:ogl-drift 40s ease-in-out infinite alternate}
@keyframes ogl-drift{to{transform:translate(14%,10%)}}
.ogl__chip{position:absolute;right:6px;top:3px;display:inline-flex;align-items:center;gap:5px;padding:2px 7px;border-radius:999px;
  background:rgba(8,10,12,.92);border:1px solid rgba(255,255,255,.10);
  font:700 8.5px/1 Inter,"Segoe UI",system-ui,sans-serif;letter-spacing:.12em;text-transform:uppercase;color:${PALETTE.chipText};white-space:nowrap}
.ogl.is-error .ogl__chip{border-color:rgba(${r},.5)}
.ogl__led{width:6px;height:6px;border-radius:50%;background:${PALETTE.ledOff};transition:background 200ms,box-shadow 200ms}
.ogl__led.ok{background:${PALETTE.green};box-shadow:0 0 6px rgba(91,227,154,.6)}
.ogl__led.warn{background:${PALETTE.amber};box-shadow:0 0 6px rgba(255,194,75,.6)}
.ogl__led.err{background:${PALETTE.red};box-shadow:0 0 6px rgba(${r},.6)}
.ogl__lock{opacity:.8;font-size:9px;padding-left:5px;border-left:1px solid rgba(255,255,255,.1)}
.ogl__err{position:absolute;left:6px;right:6px;bottom:6px;display:none;padding:5px 8px;border-radius:6px;background:rgba(40,8,14,.86);
  border:1px solid rgba(${r},.5);color:#ffd7dc;font:600 10px/1.3 ui-monospace,Consolas,monospace;overflow:hidden;max-height:46%}
.ogl.is-error .ogl__err{display:block}
.ogl.is-collapsed .ogl__err,.ogl.is-collapsed .ogl__blob{display:none}

/* ── Glass widget library (DOM widgets) ── */
.oglw{box-sizing:border-box;width:100%;padding:8px 8px 6px;display:grid;gap:8px;color:${PALETTE.titleText};
  font:500 11px/1.3 Inter,"Segoe UI",system-ui,sans-serif;background:rgba(255,255,255,.02);border:1px solid ${PALETTE.line};border-radius:8px}
.oglw *{box-sizing:border-box}
.oglw__row{display:flex;align-items:center;gap:8px;min-width:0}
.oglw__row>.grow{flex:1;min-width:0}
.oglw__label{font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:#8a949c;font-weight:700;min-width:64px}
.oglw-seg{display:inline-flex;padding:2px;gap:2px;border-radius:8px;background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.08)}
.oglw-seg button{border:0;background:transparent;color:#9aa5ad;padding:5px 10px;border-radius:6px;font:600 10px/1 inherit;letter-spacing:.03em;cursor:pointer}
.oglw-seg button:hover{color:${PALETTE.titleText}}
.oglw-seg button.on{background:rgba(${a},.14);color:${PALETTE.accent};box-shadow:0 0 0 1px rgba(${a},.45) inset}
.oglw-combo{position:relative;width:100%;appearance:none;-webkit-appearance:none;padding:6px 26px 6px 9px;border-radius:7px;color:${PALETTE.titleText};
  background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.08);font:inherit;outline:none;cursor:pointer}
.oglw-combo:focus{border-color:rgba(${a},.6);box-shadow:0 0 0 2px rgba(${a},.18)}
.oglw-combo-wrap{position:relative;flex:1;min-width:0}
.oglw-combo-wrap::after{content:"";position:absolute;right:10px;top:50%;width:6px;height:6px;border-right:1.5px solid ${PALETTE.accent};border-bottom:1.5px solid ${PALETTE.accent};transform:translateY(-70%) rotate(45deg);pointer-events:none}
.oglw-slider{flex:1;min-width:0;display:flex;align-items:center;gap:8px}
.oglw-slider input{-webkit-appearance:none;appearance:none;flex:1;height:3px;border-radius:999px;background:rgba(255,255,255,.12);outline:none}
.oglw-slider input::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:${PALETTE.accent};box-shadow:0 0 8px rgba(${a},.6);cursor:pointer}
.oglw-slider input::-moz-range-thumb{width:12px;height:12px;border:0;border-radius:50%;background:${PALETTE.accent};box-shadow:0 0 8px rgba(${a},.6);cursor:pointer}
.oglw-slider output{min-width:36px;text-align:right;font-variant-numeric:tabular-nums;color:${PALETTE.accent};font-size:10px}
.oglw-input{flex:1;min-width:0;padding:6px 9px;border-radius:7px;color:${PALETTE.titleText};background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.08);font:inherit;outline:none}
.oglw-input:focus{border-color:rgba(${a},.6);box-shadow:0 0 0 2px rgba(${a},.18)}
.oglw-switch{position:relative;width:32px;height:18px;flex:none;cursor:pointer}
.oglw-switch input{opacity:0;width:0;height:0;position:absolute}
.oglw-switch i{position:absolute;inset:0;border-radius:999px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.08);transition:background 160ms}
.oglw-switch i::after{content:"";position:absolute;left:2px;top:2px;width:12px;height:12px;border-radius:50%;background:#9aa5ad;transition:transform 160ms,background 160ms}
.oglw-switch input:checked+i{background:rgba(${a},.35);border-color:rgba(${a},.6)}
.oglw-switch input:checked+i::after{transform:translateX(14px);background:#fff;box-shadow:0 0 6px rgba(${a},.7)}
.oglw-btn{border:0;border-radius:7px;padding:7px 12px;font:700 10px/1 inherit;letter-spacing:.06em;text-transform:uppercase;cursor:pointer;
  background:${PALETTE.accent};color:#06181c;box-shadow:0 0 0 1px rgba(${a},.4),0 4px 12px rgba(${a},.18)}
.oglw-btn:hover{filter:brightness(1.08)}
.oglw-btn.ghost{background:rgba(0,0,0,.35);color:${PALETTE.titleText};border:1px solid rgba(255,255,255,.1);box-shadow:none}
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
        el.className = "ogl" + (BLOBS && !REDUCED_MOTION ? " is-anim" : "");
        el.innerHTML = `<div class="ogl__in">
            <div class="ogl__frame"><div class="ogl__blob"></div><div class="ogl__accent"></div></div>
            <div class="ogl__chip"><span class="ogl__led" title="Onyx session: not synced"></span><span>ONYX</span><span class="ogl__lock">🔒</span></div>
            <div class="ogl__err"></div></div>`;
        this.el = el;
        this.inner = el.firstElementChild;
        this.led = el.querySelector(".ogl__led");
        this.err = el.querySelector(".ogl__err");
        this.last = {};
        ensureRoot().appendChild(el);
    }
    update(s, blur) {
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
        const w = s.w + "px", h = (s.collapsed ? s.titleH : s.h) + "px";
        if (L.w !== w) { this.inner.style.width = w; L.w = w; }
        if (L.h !== h) { this.inner.style.height = h; L.h = h; }
        const flags = { "is-selected": s.selected, "is-error": !!s.error, "is-collapsed": s.collapsed, "is-noblur": !blur };
        for (const k in flags) if (L[k] !== flags[k]) { el.classList.toggle(k, flags[k]); L[k] = flags[k]; }
        if (L.lic !== s.licence) {
            this.led.className = "ogl__led " + ({ ok: "ok", warn: "warn", err: "err" }[s.licence] || "");
            this.led.title = { ok: "Onyx session: active", warn: "Onyx session: stale", err: "Onyx session: invalid" }[s.licence] || "Onyx session: not synced";
            L.lic = s.licence;
        }
        const t = s.error ? (s.error.code ? s.error.code + " — " : "") + s.error.message : "";
        if (L.err !== t) { this.err.textContent = t; L.err = t; }
    }
    dispose() { this.el.remove(); }
}

// ── State: licence LED + node errors ────────────────────────────────────────
let licence = "off";
async function pollLicence() {
    try {
        const r = await api.fetchApi("/onyx_session/profile_status");
        if (r.ok) { const d = await r.json(); if (d?.state) licence = d.state; }
    } catch (_) { /* endpoint absent without the real pack: LED stays off */ }
}
pollLicence();
setInterval(pollLicence, 30000);

const nodeErrors = new Map();
api.addEventListener("execution_start", () => nodeErrors.clear());
api.addEventListener("execution_error", (ev) => {
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
    let blurBudget = 40;
    for (const node of graph._nodes || []) {
        if (node?.comfyClass !== TARGET) continue;
        seen.add(node);
        let ov = overlays.get(node);
        if (!ov) { ov = new Overlay(); overlays.set(node, ov); }
        const w = node.size[0], h = node.size[1] + titleH;
        const x = rect.left + (off[0] + node.pos[0]) * scale;
        const y = rect.top + (off[1] + node.pos[1] - titleH) * scale;
        const visible = x + w * scale > rect.left && x < rect.right && y + h * scale > rect.top && y < rect.bottom;
        const blur = visible && blurBudget-- > 0;
        ov.update({
            x, y, w, h, titleH, scale, visible,
            selected: !!selected[node.id],
            error: nodeErrors.get(node.id) || null,
            licence,
            collapsed: !!node.flags?.collapsed,
        }, blur);
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

    // Segmented toggle bound to `mode`
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

    // Combo with chevron
    const r2 = el("div", "oglw__row");
    r2.append(el("span", "oglw__label", "preset"));
    const cw = el("div", "oglw-combo-wrap");
    const combo = el("select", "oglw-combo");
    for (const o of ["Casual phone", "Fashion editorial", "Luxury lifestyle", "Playful social"]) { const op = el("option", "", o); op.value = o; combo.append(op); }
    cw.append(combo); r2.append(cw);

    // Slider bound to `strength`
    const r3 = el("div", "oglw__row");
    r3.append(el("span", "oglw__label", "strength"));
    const sl = el("div", "oglw-slider");
    const range = el("input"); range.type = "range"; range.min = "0"; range.max = "1"; range.step = "0.01"; range.value = String(W.strength?.value ?? 0.65);
    const out = el("output", "", Number(range.value).toFixed(2));
    range.addEventListener("input", () => { out.textContent = Number(range.value).toFixed(2); setW("strength", Number(range.value)); });
    sl.append(range, out); r3.append(sl);

    // Text field
    const r4 = el("div", "oglw__row");
    r4.append(el("span", "oglw__label", "trigger"));
    const inp = el("input", "oglw-input"); inp.type = "text"; inp.placeholder = "ohwx woman";
    r4.append(inp);

    // Switch bound to `keep_metadata`
    const r5 = el("div", "oglw__row");
    r5.append(el("span", "oglw__label", "metadata"));
    const sw = el("label", "oglw-switch");
    const chk = el("input"); chk.type = "checkbox"; chk.checked = !!W.keep_metadata?.value;
    chk.addEventListener("change", () => setW("keep_metadata", chk.checked));
    sw.append(chk, el("i"));
    r5.append(sw, el("span", "", "keep EXIF"));

    // Buttons
    const r6 = el("div", "oglw__row");
    const gen = el("button", "oglw-btn", "Generate"); gen.type = "button";
    gen.addEventListener("click", () => app.queuePrompt?.(0, 1));
    const reset = el("button", "oglw-btn ghost", "Reset"); reset.type = "button";
    reset.addEventListener("click", () => {
        setW("mode", "generate"); setW("strength", 0.65); setW("keep_metadata", false); setW("simulate_sync_error", false);
        range.value = "0.65"; out.textContent = "0.65"; chk.checked = false; paintSeg();
    });
    r6.append(gen, reset);

    panel.append(r1, r2, r3, r4, r5, r6);
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
            // Height follows the panel's real content so nothing spills past the node.
            const panelHeight = () => Math.max(250, (panel.scrollHeight || 0) + 14);
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
        node.title_color = PALETTE.titleText;
        applySlotColors(node);
    },
});
