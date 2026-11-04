// Onyx — curves editor (Onyx Curves Adjust node + Post Studio "Curves" stage).
// Interaction and curve maths follow "Curves Adjust Pro" from comfyui-machinepainting-nodes
// (MIT, © 2025 MachinePainting), of which Onyx Curves Adjust is a fork.
// Click: add a point · drag: move · shift+click or double-click: remove.
import { app } from "../../scripts/app.js";

const CH = ["rgb", "red", "green", "blue"];
const COLORS = { rgb: "#ffffff", red: "#ff6b6b", green: "#6bff8a", blue: "#6b9bff" };
const IDENTITY = () => ({ rgb: [[0, 0], [1, 1]], red: [[0, 0], [1, 1]], green: [[0, 0], [1, 1]], blue: [[0, 0], [1, 1]] });

function parse(v) {
  try { const o = JSON.parse(v); const out = IDENTITY(); for (const c of CH) if (Array.isArray(o?.[c]) && o[c].length >= 2) out[c] = o[c].map((p) => [+p[0], +p[1]]); return out; }
  catch { return IDENTITY(); }
}

// Same spline as the Python side (weighted tangents, no overshoot).
function lut(points, n = 128) {
  const p = [...points].sort((a, b) => a[0] - b[0]);
  if (p[0][0] > 0.001) p.unshift([0, p[0][1]]);
  if (p[p.length - 1][0] < 0.999) p.push([1, p[p.length - 1][1]]);
  const k = p.length, t = [];
  for (let i = 0; i < k; i++) {
    if (i === 0) t.push((p[1][1] - p[0][1]) / (p[1][0] - p[0][0] || 1e-6));
    else if (i === k - 1) t.push((p[k - 1][1] - p[k - 2][1]) / (p[k - 1][0] - p[k - 2][0] || 1e-6));
    else {
      const dx0 = p[i][0] - p[i - 1][0], dx1 = p[i + 1][0] - p[i][0];
      const s0 = (p[i][1] - p[i - 1][1]) / (dx0 || 1e-6), s1 = (p[i + 1][1] - p[i][1]) / (dx1 || 1e-6);
      t.push((s0 * dx1 + s1 * dx0) / ((dx0 + dx1) || 1e-6));
    }
  }
  const out = [];
  for (let j = 0; j < n; j++) {
    const x = j / (n - 1);
    let s = 0;
    for (let q = 0; q < k - 1; q++) { if (p[q][0] <= x && x <= p[q + 1][0]) { s = q; break; } s = q; }
    const a = p[s], b = p[s + 1], h = b[0] - a[0], u = h > 1e-4 ? (x - a[0]) / h : 0;
    const u2 = u * u, u3 = u2 * u;
    const y = (2 * u3 - 3 * u2 + 1) * a[1] + (u3 - 2 * u2 + u) * t[s] * h + (-2 * u3 + 3 * u2) * b[1] + (u3 - u2) * t[s + 1] * h;
    out.push(Math.min(1, Math.max(0, y)));
  }
  return out;
}

export function curveEditor(value, onChange, { size = 260 } = {}) {
  const curves = parse(value);
  let ch = "rgb", drag = -1;
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:6px;align-items:center;width:100%";
  const cv = document.createElement("canvas");
  cv.width = size * 2; cv.height = size * 2;
  cv.style.cssText = `width:${size}px;height:${size}px;border:1px solid rgba(255,255,255,.14);border-radius:6px;background:#141418;cursor:crosshair;touch-action:none`;
  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;justify-content:center";
  const mk = (label, fn, color) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = `padding:3px 9px;border-radius:6px;border:1px solid rgba(255,255,255,.16);background:#1b1b22;color:${color || "#ddd"};font:600 11px system-ui;cursor:pointer`;
    b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); fn(); };
    bar.append(b); return b;
  };
  const chBtns = {};
  for (const c of CH) chBtns[c] = mk(c === "rgb" ? "RGB" : c.toUpperCase(), () => { ch = c; draw(); }, COLORS[c]);
  mk("Reset channel", () => { curves[ch] = [[0, 0], [1, 1]]; emit(); });
  mk("Reset all", () => { Object.assign(curves, IDENTITY()); emit(); });
  wrap.append(cv, bar);

  const emit = () => { draw(); onChange?.(JSON.stringify(curves)); };
  const toXY = (e) => { const r = cv.getBoundingClientRect(); return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, 1 - (e.clientY - r.top) / r.height))]; };
  const hit = (xy) => curves[ch].findIndex((p) => Math.hypot(p[0] - xy[0], p[1] - xy[1]) < 0.045);

  function draw() {
    const g = cv.getContext("2d"), W = cv.width, H = cv.height;
    g.clearRect(0, 0, W, H);
    g.strokeStyle = "rgba(255,255,255,.07)"; g.lineWidth = 1;
    for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(i * W / 4, 0); g.lineTo(i * W / 4, H); g.moveTo(0, i * H / 4); g.lineTo(W, i * H / 4); g.stroke(); }
    g.strokeStyle = "rgba(255,255,255,.18)"; g.beginPath(); g.moveTo(0, H); g.lineTo(W, 0); g.stroke();
    const line = (c, width, alpha) => {
      const v = lut(curves[c]); g.strokeStyle = COLORS[c]; g.globalAlpha = alpha; g.lineWidth = width; g.beginPath();
      v.forEach((y, i) => { const X = i / (v.length - 1) * W, Y = (1 - y) * H; i ? g.lineTo(X, Y) : g.moveTo(X, Y); }); g.stroke(); g.globalAlpha = 1;
    };
    for (const c of CH) if (c !== ch) line(c, 2, 0.35);
    line(ch, 4, 1);
    for (const p of curves[ch]) { g.fillStyle = "#111"; g.strokeStyle = COLORS[ch]; g.lineWidth = 3; g.beginPath(); g.arc(p[0] * W, (1 - p[1]) * H, 9, 0, 7); g.fill(); g.stroke(); }
    for (const c of CH) chBtns[c].style.background = c === ch ? "#33334a" : "#1b1b22";
  }

  cv.addEventListener("pointerdown", (e) => {
    e.preventDefault(); e.stopPropagation();
    const xy = toXY(e), i = hit(xy);
    if (i >= 0 && (e.shiftKey || e.button === 2)) { if (curves[ch].length > 2) { curves[ch].splice(i, 1); emit(); } return; }
    if (i >= 0) drag = i;
    else { curves[ch].push(xy); curves[ch].sort((a, b) => a[0] - b[0]); drag = curves[ch].indexOf(xy); draw(); }
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", (e) => {
    if (drag < 0) return;
    e.preventDefault(); e.stopPropagation();
    const xy = toXY(e), pts = curves[ch];
    const lo = drag > 0 ? pts[drag - 1][0] + 0.01 : 0, hi = drag < pts.length - 1 ? pts[drag + 1][0] - 0.01 : 1;
    pts[drag] = [Math.min(hi, Math.max(lo, xy[0])), xy[1]];
    draw();
  });
  cv.addEventListener("pointerup", (e) => { if (drag >= 0) { drag = -1; cv.releasePointerCapture?.(e.pointerId); emit(); } });
  cv.addEventListener("dblclick", (e) => { e.preventDefault(); const i = hit(toXY(e)); if (i >= 0 && curves[ch].length > 2) { curves[ch].splice(i, 1); emit(); } });
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  draw();
  wrap.set = (v) => { Object.assign(curves, parse(v)); draw(); };
  return wrap;
}

// The standalone Onyx Curves Adjust node: the editor replaces the curve_data text field.
app.registerExtension({
  name: "Onyx.CurvesEditor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "OnyxCurvesAdjust") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const w = this.widgets?.find((x) => x.name === "curve_data");
      if (w) {
        const ed = curveEditor(w.value, (v) => { w.value = v; this.setDirtyCanvas?.(true, true); });
        w.computeSize = () => [0, -4]; w.draw = () => {}; w.hidden = true;
        if (w.element) w.element.style.display = "none";
        this.addDOMWidget("curve_editor", "ONYX_CURVES", ed, { serialize: false, getValue: () => undefined, setValue: () => {}, getMinHeight: () => 320 });
        const conf = this.onConfigure?.bind(this);
        this.onConfigure = function (info) { conf?.(info); ed.set(w.value); };
        this.size = [Math.max(this.size[0], 320), Math.max(this.size[1], 430)];
      }
      return r;
    };
  },
});
