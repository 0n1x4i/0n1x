/**
 * Onyx Image and Video Batch Loader — frontend (v2)
 *
 * Layout, top to bottom:
 *   toolbar   — add media, thumbnail size, MiniMax 15 s switch, clear
 *   drop area — big when the list is empty, a thin strip once it is not,
 *               so there is always somewhere to drop more files
 *   grid      — draggable cards (drag to reorder), duration badges, limit flags
 *   settings  — collapsible panel for every video setting of the node
 *   footer    — status + Queue All
 *
 * The node's native widgets are all hidden and driven from this panel: the
 * values still live in those widgets, so saving, loading and the Python side
 * see exactly what they saw before.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ── Queue All en paquets (inchange) ─────────────────────────────────────────
// Le frontend plafonne le nombre de taches ajoutees en un clic : un
// queuePrompt(0, 420) est silencieusement tronque. On envoie donc par paquets
// en attendant que la file redescende entre chacun.
const QUEUE_CHUNK = 50;
const QUEUE_LOW_WATER = 8;
const QUEUE_POLL_MS = 1500;

async function pendingCount() {
    try {
        const q = await (await api.fetchApi("/queue")).json();
        return (q.queue_running?.length || 0) + (q.queue_pending?.length || 0);
    } catch (e) {
        return QUEUE_CHUNK;
    }
}
async function waitUntil(predicate) {
    while (!(await predicate())) await new Promise(r => setTimeout(r, QUEUE_POLL_MS));
}
async function queueInChunks(total, chunkSize, onProgress) {
    let sent = 0;
    while (sent < total) {
        const batch = Math.min(chunkSize, total - sent);
        await app.queuePrompt(0, batch);
        sent += batch;
        onProgress?.(sent, total);
        if (sent >= total) break;
        await waitUntil(async () => (await pendingCount()) <= QUEUE_LOW_WATER);
    }
    return sent;
}
// Mode consume : chaque run retire un item de la liste, donc on attend que la
// file soit VIDE avant de reconstruire le paquet suivant a partir de la liste.
async function queueConsuming(getRemaining, chunkSize, onProgress) {
    let sent = 0, guard = 0;
    while (getRemaining() > 0 && guard < 10000) {
        const batch = Math.min(chunkSize, getRemaining());
        const before = getRemaining();
        await app.queuePrompt(0, batch);
        sent += batch;
        onProgress?.(sent, before);
        await waitUntil(async () => (await pendingCount()) === 0);
        if (getRemaining() >= before) {
            console.warn("[Onyx Batch] nothing was consumed — stopping to avoid a loop.");
            break;
        }
        guard++;
    }
    return sent;
}

// ── Constantes ──────────────────────────────────────────────────────────────
const LIMIT_S = 15.0;       // doit correspondre a _MINIMAX_LIMIT_S (Python)
const HARD_S = 16.5;        // doit correspondre a _MINIMAX_HARD_S (Python)
const VIDEO_RE = /\.(mp4|mov|webm|mkv|avi|m4v|mpe?g|wmv|flv)$/i;
const THUMB_SIZES = { S: 92, M: 128, L: 172 };
const SETTING_WIDGETS = [
    "video_max_side", "trim_mode", "trim_start", "trim_end", "resize_method",
    "custom_width", "custom_height", "force_fps", "consume_on_load",
    "queue_batch_size", "minimax_15s_limit", "video_first_frame", "max_frames",
];

// ── Styles ──────────────────────────────────────────────────────────────────
const STYLE_ID = "onyx-batch-v2-style";
function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.obl{--bg:#06161c;--panel:#0a1f27;--panel2:#0d2730;--line:#15363f;--line2:#1f4a55;--acc:#22b8d4;--acc2:#5fd8ef;
  --dim:#6f9aa5;--txt:#cfe9ef;--ok:#3ecf6a;--warn:#f0b43c;--bad:#ef5b5b;
  display:flex;flex-direction:column;width:100%;height:100%;box-sizing:border-box;background:var(--bg);
  border:1px solid var(--line);border-radius:8px;overflow:hidden;color:var(--txt);
  font:12px/1.35 'Segoe UI',system-ui,sans-serif;user-select:none}
.obl *{box-sizing:border-box}
.obl-bar{display:flex;align-items:center;gap:6px;padding:8px 10px;background:var(--panel);border-bottom:1px solid var(--line);flex-shrink:0}
.obl-head{display:flex;flex-direction:column;flex:1;min-width:0;margin-right:4px}
.obl-title{font-weight:700;letter-spacing:.3px;color:var(--acc2);white-space:nowrap}
.obl-stats{color:var(--dim);font-size:10.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.obl-btn{background:var(--panel2);color:var(--txt);border:1px solid var(--line2);border-radius:6px;padding:5px 10px;
  font:600 11.5px 'Segoe UI',system-ui,sans-serif;cursor:pointer;white-space:nowrap;transition:background .12s,border-color .12s,color .12s}
.obl-btn:hover{background:#12333d;border-color:var(--acc)}
.obl-btn:disabled{opacity:.45;cursor:default}
.obl-btn.primary{background:var(--acc);border-color:var(--acc);color:#04161b}
.obl-btn.primary:hover{background:var(--acc2)}
.obl-btn.danger{color:#ef8a8a}
.obl-btn.danger.armed{background:var(--bad);border-color:var(--bad);color:#fff}
.obl-seg{display:flex;border:1px solid var(--line2);border-radius:6px;overflow:hidden}
.obl-seg button{background:transparent;border:0;color:var(--dim);padding:4px 8px;font:600 11px 'Segoe UI',sans-serif;cursor:pointer}
.obl-seg button.on{background:var(--line2);color:var(--txt)}
.obl-pill{display:flex;align-items:center;gap:6px;padding:4px 9px;border:1px solid var(--line2);border-radius:999px;cursor:pointer;
  color:var(--dim);font-weight:600;font-size:11px;white-space:nowrap}
.obl-pill i{width:26px;height:14px;border-radius:7px;background:#23404a;position:relative;transition:background .15s}
.obl-pill i:after{content:"";position:absolute;top:2px;left:2px;width:10px;height:10px;border-radius:50%;background:#8fb3bc;transition:left .15s,background .15s}
.obl-pill.on{color:var(--txt);border-color:var(--acc)}
.obl-pill.on i{background:var(--acc)}
.obl-pill.on i:after{left:14px;background:#fff}
.obl-drop{margin:10px;border:2px dashed var(--line2);border-radius:10px;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:6px;cursor:pointer;color:var(--dim);transition:border-color .15s,background .15s;flex-shrink:0}
.obl-drop.big{flex:1;min-height:170px}
.obl-drop.big .obl-drop-ico{font-size:34px;line-height:1}
.obl-drop.big b{color:var(--acc2);font-size:14px}
.obl-drop.strip{flex-direction:row;padding:7px;margin:8px 10px 0;border-width:1px;font-size:11.5px}
.obl-drop.strip .obl-drop-ico{font-size:14px}
.obl-drop:hover,.obl.dragging .obl-drop{border-color:var(--acc);background:rgba(34,184,212,.08);color:var(--txt)}
.obl-grid-wrap{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;padding:10px;scrollbar-width:thin;scrollbar-color:var(--line2) transparent}
.obl-grid{display:grid;gap:8px}
.obl-card{position:relative;border:2px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel);cursor:grab;
  transition:border-color .12s,box-shadow .12s,opacity .12s}
.obl-card:hover{border-color:var(--line2)}
.obl-card.active{border-color:var(--acc);box-shadow:0 0 0 2px rgba(34,184,212,.25)}
.obl-card.drag-src{opacity:.35}
.obl-card.ins-before{box-shadow:-4px 0 0 0 var(--acc)}
.obl-card.ins-after{box-shadow:4px 0 0 0 var(--acc)}
.obl-card.lim-trim{border-color:rgba(240,180,60,.55)}
.obl-card.lim-skip{border-color:rgba(239,91,91,.6)}
.obl-card.lim-skip .obl-thumb{filter:grayscale(.85) brightness(.55)}
.obl-thumb{display:block;width:100%;aspect-ratio:1;object-fit:cover;background:#0b1a20;pointer-events:none}
.obl-name{padding:4px 6px;font-size:10.5px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.obl-card.active .obl-name{color:var(--txt);font-weight:700}
.obl-badge{position:absolute;padding:1px 5px;border-radius:4px;font-size:10px;font-weight:700;background:rgba(0,0,0,.66);color:#dfeff3;pointer-events:none}
.obl-idx{top:5px;left:5px}
.obl-dur{top:5px;right:5px}
.obl-dur.trim{background:var(--warn);color:#241700}
.obl-dur.skip{background:var(--bad);color:#fff}
.obl-flag{position:absolute;left:0;right:0;bottom:22px;text-align:center;font-size:10px;font-weight:700;padding:2px 0;pointer-events:none}
.obl-flag.trim{background:rgba(240,180,60,.9);color:#241700}
.obl-flag.skip{background:rgba(239,91,91,.92);color:#fff}
.obl-del{position:absolute;bottom:26px;right:5px;width:22px;height:22px;border-radius:50%;border:0;background:rgba(0,0,0,.72);
  color:#ff9b9b;font-weight:700;cursor:pointer;display:none;align-items:center;justify-content:center;padding:0}
.obl-card:hover .obl-del{display:flex}
.obl-del:hover{background:var(--bad);color:#fff}
.obl-settings{border-top:1px solid var(--line);background:var(--panel);flex-shrink:0}
.obl-set-head{display:flex;align-items:center;gap:8px;padding:7px 10px;cursor:pointer}
.obl-set-head b{color:var(--txt);font-size:11.5px}
.obl-set-head span{color:var(--dim);font-size:11px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.obl-chev{color:var(--dim);transition:transform .15s}
.obl-settings.open .obl-chev{transform:rotate(90deg)}
.obl-set-body{display:none;padding:2px 10px 10px;gap:10px;flex-direction:column}
.obl-settings.open .obl-set-body{display:flex}
.obl-group{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px 8px;align-items:end}
.obl-group-t{grid-column:1/-1;color:var(--acc2);font-size:10px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;margin-top:2px}
.obl-f{display:flex;flex-direction:column;gap:3px;min-width:0}
.obl-f label{color:var(--dim);font-size:10.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.obl-f input,.obl-f select{width:100%;background:#071419;border:1px solid var(--line2);border-radius:5px;color:var(--txt);
  padding:4px 6px;font:12px 'Segoe UI',system-ui,sans-serif;outline:none;user-select:text}
.obl-f input:focus,.obl-f select:focus{border-color:var(--acc)}
.obl-hint{grid-column:1/-1;color:var(--dim);font-size:10.5px}
.obl-foot{display:flex;align-items:center;gap:8px;padding:8px 10px;background:var(--panel2);border-top:1px solid var(--line);flex-shrink:0}
.obl-status{flex:1;min-width:0;color:var(--dim);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.obl-status.warn{color:var(--warn)}
.obl-queue{padding:7px 16px;font-size:12.5px}
`;
    document.head.appendChild(s);
}

// ── Helpers ─────────────────────────────────────────────────────────────────
function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}
function fmtDur(s) {
    if (!(s > 0)) return "";
    if (s < 60) return `${s.toFixed(s < 10 ? 1 : 1)}s`;
    const m = Math.floor(s / 60), r = Math.round(s - m * 60);
    return `${m}:${String(r).padStart(2, "0")}`;
}
function isVideoMeta(meta) {
    return !!meta?.is_video || VIDEO_RE.test(meta?.filename || "");
}
function hideNativeWidget(w) {
    if (!w) return;
    w.hidden = true;
    w.computeSize = () => [0, -4];
    w.draw = () => {};
    if (w.element) w.element.style.cssText = "display:none!important";
}

// ── Main setup ──────────────────────────────────────────────────────────────
function setupBatchLoader(node) {
    injectStyle();
    node.properties ||= {};

    let data = { images: [], order: [] };
    let currentId = null;
    let uploading = false;
    let fitPending = true;

    const W = name => node.widgets?.find(w => w.name === name);
    const getVal = (name, fb) => { const w = W(name); return w === undefined ? fb : w.value; };
    const setVal = (name, v) => {
        const w = W(name); if (!w) return;
        w.value = v; w.callback?.(v);
        app.graph?.setDirtyCanvas(true, true);
    };

    const hideAll = () => {
        hideNativeWidget(W("batch_data"));
        SETTING_WIDGETS.forEach(n => hideNativeWidget(W(n)));
    };
    hideAll(); setTimeout(hideAll, 50); setTimeout(hideAll, 300);

    const syncBatch = () => {
        const w = W("batch_data");
        if (w) w.value = JSON.stringify(data);
        app.graph?.setDirtyCanvas(true, true);
    };
    const metaOf = id => data.images.find(m => m.id === id);

    // ── Limit / eligibility (mirrors _effective_duration in Python) ────────
    function effectiveDuration(meta) {
        const dur = Number(meta.duration) || 0, fps = Number(meta.fps) || 0;
        if (dur <= 0) return 0;
        const mode = getVal("trim_mode", "seconds");
        const ts = Number(getVal("trim_start", 0)) || 0, te = Number(getVal("trim_end", 0)) || 0;
        let start, end;
        if (mode === "frames") {
            if (fps <= 0) return dur;
            start = ts / fps; end = te > 0 ? te / fps : dur;
        } else {
            start = ts; end = te > 0 ? te : dur;
        }
        return Math.max(0, Math.min(end, dur) - Math.min(start, dur));
    }
    function limitState(meta) {
        if (!getVal("minimax_15s_limit", false) || !isVideoMeta(meta)) return "";
        const d = effectiveDuration(meta);
        if (!(d > 0)) return "";
        if (d > HARD_S + 1e-6) return "skip";
        if (d > LIMIT_S + 1e-6) return "trim";
        return "";
    }
    const eligibleIds = () => data.order.filter(id => { const m = metaOf(id); return m && limitState(m) !== "skip"; });

    // ── DOM ────────────────────────────────────────────────────────────────
    const root = el("div", "obl");

    const fileInput = el("input");
    fileInput.type = "file"; fileInput.multiple = true; fileInput.accept = "image/*,video/*";
    fileInput.style.display = "none";
    root.appendChild(fileInput);

    // toolbar
    const bar = el("div", "obl-bar");
    const title = el("div", "obl-title", "Media batch");
    const stats = el("div", "obl-stats");
    const addBtn = el("button", "obl-btn", "+ Add");
    addBtn.title = "Add images or videos (you can also drop files anywhere on the node)";
    const seg = el("div", "obl-seg");
    const sizeBtns = {};
    for (const k of Object.keys(THUMB_SIZES)) {
        const b = el("button", "", k); b.title = `Thumbnail size ${k}`;
        b.addEventListener("click", () => { node.properties.obl_thumb = k; fitPending = true; render(); });
        sizeBtns[k] = b; seg.appendChild(b);
    }
    const limitPill = el("div", "obl-pill");
    limitPill.append(el("i"), document.createTextNode("MiniMax 15s"));
    limitPill.title = "MiniMax 15 s limit\nOn: clips between 15 s and 16.5 s (after trim) are cut to 15 s; " +
                      "clips over 16.5 s are skipped. Images are not affected.";
    limitPill.addEventListener("click", () => { if (!W("minimax_15s_limit")) return; setVal("minimax_15s_limit", !getVal("minimax_15s_limit", false)); render(); });
    const clearBtn = el("button", "obl-btn danger", "Clear");
    const head = el("div", "obl-head");
    head.append(title, stats);
    bar.append(head, addBtn, seg, limitPill, clearBtn);

    // drop area
    const drop = el("div", "obl-drop");
    drop.addEventListener("click", () => fileInput.click());

    // grid
    const gridWrap = el("div", "obl-grid-wrap");
    const grid = el("div", "obl-grid");
    gridWrap.appendChild(grid);

    // settings
    const settings = el("div", "obl-settings");
    const setHead = el("div", "obl-set-head");
    const chev = el("div", "obl-chev", "▸");
    const setTitle = el("b", "", "Video settings");
    const setSummary = el("span");
    setHead.append(chev, setTitle, setSummary);
    const setBody = el("div", "obl-set-body");
    settings.append(setHead, setBody);
    setHead.addEventListener("click", () => {
        node.properties.obl_settings_open = !settings.classList.contains("open");
        settings.classList.toggle("open", node.properties.obl_settings_open);
        fitPending = true; fitNode();
    });

    const controls = {};
    function numField(name, label, { step = 1, min = 0, max = 100000, float = false } = {}) {
        const f = el("div", "obl-f"); const l = el("label", "", label);
        const i = el("input"); i.type = "number"; i.step = String(step); i.min = String(min); i.max = String(max);
        i.addEventListener("change", () => {
            let v = float ? parseFloat(i.value) : parseInt(i.value, 10);
            if (!isFinite(v)) v = Number(getVal(name, 0)) || 0;
            v = Math.min(max, Math.max(min, v));
            i.value = String(v); setVal(name, v); render();
        });
        i.addEventListener("keydown", e => e.stopPropagation());
        f.append(l, i); controls[name] = { get: () => i, set: v => { i.value = String(v); } };
        return f;
    }
    function selField(name, label, options) {
        const f = el("div", "obl-f"); const l = el("label", "", label);
        const s = el("select");
        for (const o of options) { const op = el("option", "", o); op.value = o; s.appendChild(op); }
        s.addEventListener("change", () => { setVal(name, s.value); render(); });
        f.append(l, s); controls[name] = { get: () => s, set: v => { s.value = String(v); } };
        return f;
    }
    function boolField(name, label, onTxt, offTxt) {
        const f = el("div", "obl-f"); const l = el("label", "", label);
        const s = el("select");
        for (const [v, t] of [["1", onTxt], ["0", offTxt]]) { const op = el("option", "", t); op.value = v; s.appendChild(op); }
        s.addEventListener("change", () => { setVal(name, s.value === "1"); render(); });
        f.append(l, s); controls[name] = { get: () => s, set: v => { s.value = v ? "1" : "0"; } };
        return f;
    }
    function group(titleTxt, ...fields) {
        const g = el("div", "obl-group"); g.appendChild(el("div", "obl-group-t", titleTxt));
        fields.forEach(x => g.appendChild(x)); return g;
    }
    const hint = t => el("div", "obl-hint", t);
    setBody.append(
        group("Decode",
            numField("video_max_side", "Max side (px, 0 = native)", { step: 64, max: 4096 }),
            numField("force_fps", "Force fps (0 = native)", { step: 0.01, max: 240, float: true }),
            numField("max_frames", "Frame load cap (0 = all)", { step: 1, max: 100000 }),
            boolField("minimax_15s_limit", "MiniMax 15 s limit", "on", "off")),
        group("Resize",
            selField("resize_method", "Method", ["none", "maintain aspect ratio", "stretch to fit", "pad", "crop"]),
            numField("custom_width", "Width (0 = auto)", { step: 8, max: 8192 }),
            numField("custom_height", "Height (0 = auto)", { step: 8, max: 8192 })),
        group("Trim — applied to every clip",
            selField("trim_mode", "Unit", ["seconds", "frames"]),
            numField("trim_start", "Start", { step: 0.01, float: true }),
            numField("trim_end", "End (0 = clip end)", { step: 0.01, float: true })),
        group("Outputs",
            boolField("video_first_frame", "Video first frame → image output", "on", "off"),
            hint("On: each video outputs ONLY its first frame (at trim start, after resize) through `image` — the video itself is not loaded.")),
        group("Queue",
            boolField("consume_on_load", "After loading an item", "remove it from the list", "keep the list"),
            numField("queue_batch_size", "Runs sent per batch", { min: 1, max: 50 }),
            hint("Remove + batch 1 = a crashed batch resumes where it stopped.")),
    );

    // footer
    const foot = el("div", "obl-foot");
    const status = el("div", "obl-status");
    const queueBtn = el("button", "obl-btn primary obl-queue", "▶ Queue all");
    foot.append(status, queueBtn);

    root.append(bar, drop, gridWrap, settings, foot);

    // ── Rendering ──────────────────────────────────────────────────────────
    // Un reglage dont le widget n'existe pas = le serveur ComfyUI tourne encore
    // sur une ancienne version du node (pas redemarre apres une mise a jour).
    // Le controle est alors desactive et signale, au lieu de basculer dans le
    // vide sans que rien ne change a l'execution.
    let missingSettings = [];
    function syncControls() {
        missingSettings = [];
        for (const [name, c] of Object.entries(controls)) {
            const el = c.get();
            if (!W(name)) {
                missingSettings.push(name);
                el.disabled = true;
                el.title = "Not available: restart ComfyUI to load the updated node.";
                continue;
            }
            el.disabled = false; el.title = "";
            const v = getVal(name, undefined);
            if (v !== undefined && document.activeElement !== el) c.set(v);
        }
        limitPill.style.opacity = W("minimax_15s_limit") ? "" : ".4";
        limitPill.classList.toggle("on", !!getVal("minimax_15s_limit", false));
        const k = node.properties.obl_thumb || "M";
        for (const [key, b] of Object.entries(sizeBtns)) b.classList.toggle("on", key === k);
        settings.classList.toggle("open", !!node.properties.obl_settings_open);

        const parts = [];
        const ms = Number(getVal("video_max_side", 1024));
        parts.push(ms ? `≤${ms}px` : "native size");
        const ff = Number(getVal("force_fps", 0)); parts.push(ff ? `${ff} fps` : "native fps");
        const mf = Number(getVal("max_frames", 0)); if (mf) parts.push(`≤${mf} frames`);
        const rm = getVal("resize_method", "none");
        if (rm !== "none") parts.push(`${rm} ${getVal("custom_width", 0) || "auto"}×${getVal("custom_height", 0) || "auto"}`);
        const ts = Number(getVal("trim_start", 0)), te = Number(getVal("trim_end", 0));
        const u = getVal("trim_mode", "seconds") === "frames" ? "f" : "s";
        if (ts || te) parts.push(`trim ${ts}${u}–${te ? te + u : "end"}`);
        if (getVal("minimax_15s_limit", false)) parts.push("MiniMax 15s");
        if (getVal("video_first_frame", false)) parts.push("1st frame → image");
        if (getVal("consume_on_load", false)) parts.push("consume");
        setSummary.textContent = parts.join(" · ");
    }

    function renderStats() {
        const n = data.order.length;
        let vids = 0, total = 0;
        for (const id of data.order) {
            const m = metaOf(id); if (!m) continue;
            if (isVideoMeta(m)) { vids++; total += effectiveDuration(m) || 0; }
        }
        const imgs = n - vids;
        stats.textContent = n
            ? [imgs && `${imgs} image${imgs > 1 ? "s" : ""}`, vids && `${vids} video${vids > 1 ? "s" : ""}`,
               total > 0 && fmtDur(total)].filter(Boolean).join(" · ")
            : "empty";
    }

    function renderFooter() {
        const n = data.order.length;
        const elig = eligibleIds().length;
        const skipped = n - elig;
        const trims = data.order.filter(id => { const m = metaOf(id); return m && limitState(m) === "trim"; }).length;
        if (queueBtn.dataset.busy !== "1") {
            queueBtn.textContent = `▶ Queue all (${elig})`;
            queueBtn.disabled = elig === 0;
        }
        const msgs = [];
        if (!n) msgs.push("Drop images or videos to start.");
        else {
            if (trims) msgs.push(`${trims} clip${trims > 1 ? "s" : ""} cut to 15 s`);
            if (skipped) msgs.push(`${skipped} over 16.5 s will be skipped`);
            if (!msgs.length) msgs.push(getVal("consume_on_load", false)
                ? "Items leave the list once loaded." : "Each run loads the next item, in order.");
        }
        if (missingSettings.length) msgs.unshift("Restart ComfyUI: the server runs an older version of this node");
        status.textContent = msgs.join(" · ");
        status.classList.toggle("warn", !!skipped || missingSettings.length > 0);
    }

    function cardFor(id, i) {
        const meta = metaOf(id);
        const card = el("div", "obl-card");
        card.dataset.id = id;
        card.draggable = true;
        const lim = limitState(meta);
        if (lim) card.classList.add(`lim-${lim}`);
        if (id === currentId) card.classList.add("active");

        const img = el("img", "obl-thumb");
        img.loading = "lazy"; img.draggable = false;
        img.src = `/onyx/view/${encodeURIComponent(meta.thumbnail || meta.filename)}`;
        img.onerror = () => { img.style.visibility = "hidden"; };

        const name = (meta.original_name || meta.filename || "").replace(/\.[^.]+$/, "");
        const nameEl = el("div", "obl-name", name);
        nameEl.title = meta.original_name || meta.filename;

        card.append(img, el("div", "obl-badge obl-idx", String(i + 1)));
        if (isVideoMeta(meta)) {
            const eff = effectiveDuration(meta);
            const d = el("div", "obl-badge obl-dur", eff > 0 ? `🎬 ${fmtDur(eff)}` : "🎬");
            if (lim) d.classList.add(lim);
            card.appendChild(d);
            if (lim === "trim") card.appendChild(el("div", "obl-flag trim", "→ cut to 15s"));
            if (lim === "skip") card.appendChild(el("div", "obl-flag skip", "> 16.5s · skipped"));
        }
        const del = el("button", "obl-del", "✕");
        del.title = "Remove from the batch";
        del.addEventListener("click", e => { e.stopPropagation(); removeItem(id); });
        card.append(nameEl, del);

        // drag to reorder
        card.addEventListener("dragstart", e => {
            e.stopPropagation();
            e.dataTransfer.setData("application/x-onyx-batch", id);
            e.dataTransfer.effectAllowed = "move";
            card.classList.add("drag-src");
        });
        card.addEventListener("dragend", () => {
            card.classList.remove("drag-src");
            grid.querySelectorAll(".ins-before,.ins-after").forEach(c => c.classList.remove("ins-before", "ins-after"));
        });
        card.addEventListener("dragover", e => {
            if (!e.dataTransfer.types.includes("application/x-onyx-batch")) return;
            e.preventDefault(); e.stopPropagation();
            const r = card.getBoundingClientRect();
            const after = e.clientX > r.left + r.width / 2;
            card.classList.toggle("ins-after", after); card.classList.toggle("ins-before", !after);
        });
        card.addEventListener("dragleave", () => card.classList.remove("ins-before", "ins-after"));
        card.addEventListener("drop", e => {
            const src = e.dataTransfer.getData("application/x-onyx-batch");
            if (!src) return;
            e.preventDefault(); e.stopPropagation();
            const after = card.classList.contains("ins-after");
            card.classList.remove("ins-before", "ins-after");
            moveItem(src, id, after);
        });
        return card;
    }

    function render() {
        const n = data.order.length;
        const k = node.properties.obl_thumb || "M";
        grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${THUMB_SIZES[k]}px, 1fr))`;

        drop.className = `obl-drop ${n ? "strip" : "big"}`;
        drop.innerHTML = n
            ? `<span class="obl-drop-ico">＋</span><span>Drop more files here or click</span>`
            : `<div class="obl-drop-ico">🖼️</div><b>Drop images or videos</b><span>or click to browse</span>`;
        gridWrap.style.display = n ? "" : "none";

        grid.innerHTML = "";
        data.order.forEach((id, i) => { if (metaOf(id)) grid.appendChild(cardFor(id, i)); });

        syncControls(); renderStats(); renderFooter();
        fitNode();
    }

    function setActive(id) {
        currentId = id;
        for (const c of grid.children) c.classList.toggle("active", c.dataset.id === id);
        const a = [...grid.children].find(c => c.dataset.id === id);
        if (a) a.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }

    // ── Sizing ─────────────────────────────────────────────────────────────
    // La hauteur voulue est calculee a partir du contenu puis exposee au
    // frontend via getMinHeight : les widgets natifs etant caches, le node ne
    // contient plus que ce panneau, et il ne peut pas etre ecrase.
    function desiredHeight() {
        const n = data.order.length;
        const fixed = (bar.offsetHeight || 44) + (foot.offsetHeight || 46)
                    + (settings.offsetHeight || 36) + (n ? (drop.offsetHeight || 32) + 8 : 0);
        if (!n) return fixed + 200;
        const size = THUMB_SIZES[node.properties.obl_thumb || "M"];
        const cols = Math.max(1, Math.floor((node.size[0] - 22) / (size + 8)));
        const rows = Math.ceil(n / cols);
        const gridH = Math.min(Math.max(rows * (size + 30) + 20, size + 50), 3 * (size + 30) + 20);
        return fixed + gridH;
    }
    let minH = 420;
    function fitNode() {
        requestAnimationFrame(() => {
            minH = Math.max(260, desiredHeight());
            if (fitPending || node.size[1] < minH + 30) {
                const h = minH + 30;
                const w = Math.max(node.size[0], 420);
                if (node.setSize) node.setSize([w, h]); else node.size = [w, h];
                fitPending = false;
            }
            app.graph?.setDirtyCanvas(true, true);
        });
    }

    // ── Actions ────────────────────────────────────────────────────────────
    function moveItem(srcId, targetId, after) {
        if (srcId === targetId) return;
        const o = data.order.filter(x => x !== srcId);
        let at = o.indexOf(targetId);
        if (at < 0) return;
        if (after) at++;
        o.splice(at, 0, srcId);
        data.order = o;
        syncBatch(); render();
    }

    async function removeItem(id) {
        fetch(`/onyx/batch_delete/${id}`, { method: "DELETE" }).catch(() => {});
        data.images = data.images.filter(m => m.id !== id);
        data.order = data.order.filter(x => x !== id);
        if (currentId === id) currentId = null;
        fitPending = true; syncBatch(); render();
    }

    let clearTimer = null;
    clearBtn.addEventListener("click", () => {
        if (!data.order.length) return;
        if (!clearBtn.classList.contains("armed")) {
            clearBtn.classList.add("armed");
            clearBtn.textContent = `Clear ${data.order.length}?`;
            clearTimer = setTimeout(() => { clearBtn.classList.remove("armed"); clearBtn.textContent = "Clear"; }, 2500);
            return;
        }
        clearTimeout(clearTimer);
        clearBtn.classList.remove("armed"); clearBtn.textContent = "Clear";
        for (const id of [...data.order]) fetch(`/onyx/batch_delete/${id}`, { method: "DELETE" }).catch(() => {});
        data = { images: [], order: [] }; currentId = null;
        fitPending = true; syncBatch(); render();
    });

    addBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", e => { handleFiles([...e.target.files]); fileInput.value = ""; });

    // file drop anywhere on the node (internal card drags are ignored here)
    const isFileDrag = e => [...(e.dataTransfer?.types || [])].includes("Files");
    root.addEventListener("dragover", e => {
        if (!isFileDrag(e)) return;
        e.preventDefault(); e.stopPropagation(); root.classList.add("dragging");
    });
    root.addEventListener("dragleave", e => {
        if (!root.contains(e.relatedTarget)) root.classList.remove("dragging");
    });
    root.addEventListener("drop", e => {
        root.classList.remove("dragging");
        if (!isFileDrag(e)) return;
        e.preventDefault(); e.stopPropagation();
        // f.type est parfois vide depuis l'explorateur Windows : l'extension sert de repli.
        const files = [...e.dataTransfer.files].filter(
            f => f.type.startsWith("image/") || f.type.startsWith("video/") || VIDEO_RE.test(f.name));
        if (files.length) handleFiles(files);
    });

    async function handleFiles(files) {
        if (uploading) return;
        uploading = true;
        addBtn.disabled = true;
        const prev = status.textContent;
        status.textContent = `Uploading ${files.length} file${files.length > 1 ? "s" : ""}…`;
        const form = new FormData();
        for (const f of files) form.append("files", f);
        try {
            const resp = await fetch("/onyx/batch_upload", { method: "POST", body: form });
            const json = await resp.json();
            if (json.success) {
                for (const m of json.images) {
                    if (!metaOf(m.id)) { data.images.push(m); data.order.push(m.id); }
                }
                fitPending = true; syncBatch(); render();
            } else {
                console.error("[Onyx Batch] Upload failed:", json.error);
                status.textContent = `Upload failed: ${json.error || "unknown error"}`;
            }
        } catch (e) {
            console.error("[Onyx Batch] Upload error:", e);
            status.textContent = `Upload error: ${e.message || e}`;
        } finally {
            uploading = false;
            addBtn.disabled = false;
            if (status.textContent.startsWith("Uploading")) status.textContent = prev;
        }
    }

    // Items saved before durations were recorded at upload: ask the server once.
    async function probeMissing() {
        let changed = false;
        for (const m of data.images) {
            if (!isVideoMeta(m) || Number(m.duration) > 0 || m._probed) continue;
            m._probed = true;
            try {
                const r = await (await fetch(`/onyx/batch_probe/${encodeURIComponent(m.filename)}`)).json();
                if (r.success) { m.duration = r.duration; m.fps = r.fps; m.frame_count = r.frame_count; changed = true; }
            } catch (_) {}
        }
        if (changed) { syncBatch(); render(); }
    }

    queueBtn.addEventListener("click", async () => {
        const elig = eligibleIds().length;
        if (!elig || queueBtn.dataset.busy === "1") return;
        const consume = !!getVal("consume_on_load", false);
        const chunk = Math.max(1, Math.min(QUEUE_CHUNK, parseInt(getVal("queue_batch_size", QUEUE_CHUNK), 10) || QUEUE_CHUNK));
        queueBtn.dataset.busy = "1";
        queueBtn.disabled = true;
        const progress = (sent, total) => {
            queueBtn.textContent = sent >= total ? `⏳ ${sent}/${total}` : `⏳ ${sent}/${total} · waiting`;
        };
        try {
            if (consume) {
                queueBtn.textContent = `⏳ 0/${elig}`;
                await queueConsuming(() => eligibleIds().length, chunk, progress);
            } else if (elig <= chunk) {
                await app.queuePrompt(0, elig);
            } else {
                queueBtn.textContent = `⏳ 0/${elig}`;
                await queueInChunks(elig, chunk, progress);
            }
        } catch (e) {
            console.error("[Onyx Batch] queuePrompt error:", e);
        } finally {
            queueBtn.dataset.busy = "0";
            render();
        }
    });

    // ── Server events ──────────────────────────────────────────────────────
    const wsHandler = ({ detail }) => {
        if (String(detail?.node_id) !== String(node.id)) return;
        const i = detail.current_index ?? -1;
        // index dans la liste envoyee au run : on le retraduit en id, la liste
        // locale ayant pu etre reordonnee depuis.
        setActive(i >= 0 ? data.order[i] ?? null : null);
    };
    const consumeHandler = ({ detail }) => {
        if (String(detail?.node_id) !== String(node.id)) return;
        const id = String(detail.image_id || "");
        if (!id || !data.order.includes(id)) return;
        data.order = data.order.filter(x => x !== id);
        data.images = data.images.filter(m => String(m.id) !== id);
        currentId = null;
        syncBatch(); render();
    };
    api.addEventListener("onyx_batch_loader_update", wsHandler);
    api.addEventListener("onyx_batch_loader_consume", consumeHandler);

    // ── Widget ─────────────────────────────────────────────────────────────
    node.addDOMWidget("images_display", "ONYX_BATCH_DISPLAY", root, {
        serialize: false,
        getValue: () => undefined,
        setValue: () => {},
        getMinHeight: () => minH,
    });

    const origSerialize = node.serialize?.bind(node);
    node.serialize = function () {
        const out = origSerialize ? origSerialize() : {};
        if (out.widgets_values) out.widgets_values[0] = JSON.stringify(data);
        return out;
    };

    const origConfigure = node.onConfigure?.bind(node);
    node.onConfigure = function (info) {
        origConfigure?.(info);
        const saved = info?.widgets_values?.[0];
        if (saved) {
            try { data = JSON.parse(saved) || { images: [], order: [] }; }
            catch { data = { images: [], order: [] }; }
            data.images ||= []; data.order ||= [];
        }
        hideAll();
        fitPending = false;
        syncBatch(); render();
        probeMissing();
    };

    // Les reglages peuvent aussi changer par ailleurs (undo, paste) : on se
    // resynchronise quand le node est redessine plutot que de le rater.
    const origWidgetChanged = node.onWidgetChanged?.bind(node);
    node.onWidgetChanged = function () {
        const r = origWidgetChanged?.(...arguments);
        syncControls(); renderStats(); renderFooter();
        return r;
    };

    const origRemoved = node.onRemoved?.bind(node);
    node.onRemoved = function () {
        api.removeEventListener("onyx_batch_loader_update", wsHandler);
        api.removeEventListener("onyx_batch_loader_consume", consumeHandler);
        origRemoved?.();
    };

    node.size = [520, 460];
    render();
}

app.registerExtension({
    name: "Onyx.ImageBatchLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "OnyxImageBatchLoader") return;
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            setupBatchLoader(this);
        };
    },
});
