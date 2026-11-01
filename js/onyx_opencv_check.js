// Onyx — OpenCV duplicate-install dialog.
// The Python side only DETECTS the problem (utils/onyx_opencv.py). Nothing is
// uninstalled unless the user clicks "Fix at next start" here.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function el(tag, style, text) {
  const e = document.createElement(tag);
  if (style) e.style.cssText = style;
  if (text !== undefined) e.textContent = text;
  return e;
}

function button(label, primary, onClick) {
  const b = el("button",
    "padding:7px 14px;border-radius:6px;cursor:pointer;font-size:13px;margin-left:8px;" +
    (primary ? "background:#7c5cff;color:#fff;border:1px solid #7c5cff;"
             : "background:transparent;color:inherit;border:1px solid #666;"), label);
  b.onclick = onClick;
  return b;
}

function showDialog(st) {
  const overlay = el("div",
    "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;" +
    "align-items:center;justify-content:center;");
  const box = el("div",
    "background:#1e1e24;color:#e8e8ee;max-width:560px;width:calc(100% - 32px);" +
    "border-radius:10px;padding:20px 22px;font:13px/1.5 system-ui,sans-serif;" +
    "box-shadow:0 10px 40px rgba(0,0,0,.5);");
  box.append(el("div", "font-size:16px;font-weight:600;margin-bottom:8px;",
                "Onyx — several OpenCV builds installed"));
  box.append(el("div", "margin-bottom:8px;",
    "These packages all install the same cv2 folder and break each other: " +
    Object.entries(st.installed).map(([n, v]) => `${n} ${v}`).join(", ") + "."));
  box.append(el("div", "margin-bottom:8px;",
    `The fix keeps a single build (${st.plan.install}) and runs at the next ComfyUI start, ` +
    "before anything loads cv2. Nothing is changed until you click Fix."));
  const pre = el("pre",
    "background:#111;padding:8px 10px;border-radius:6px;white-space:pre-wrap;" +
    "font-size:11px;margin:0 0 14px;user-select:text;", st.command);
  box.append(el("div", "opacity:.7;margin-bottom:4px;", "Or run it yourself with ComfyUI closed:"));
  box.append(pre);
  const row = el("div", "display:flex;justify-content:flex-end;flex-wrap:wrap;gap:6px 0;");
  const msg = el("div", "margin-top:10px;opacity:.85;");
  const close = () => overlay.remove();
  row.append(
    button("Don't show again", false, async () => {
      try { await api.fetchApi("/onyx/opencv/dismiss", { method: "POST" }); } catch {}
      close();
    }),
    button("Later", false, close),
    button("Fix at next start", true, async () => {
      try {
        const r = await api.fetchApi("/onyx/opencv/fix", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: "next_start" }),
        });
        const j = await r.json();
        msg.textContent = j.message || "Scheduled. Restart ComfyUI to apply.";
        row.replaceChildren(button("OK", true, close));
      } catch (e) {
        msg.textContent = "Could not schedule the fix: " + e;
      }
    }),
  );
  box.append(row, msg);
  overlay.append(box);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  document.body.append(overlay);
}

app.registerExtension({
  name: "Onyx.OpenCVCheck",
  async setup() {
    try {
      const st = await (await api.fetchApi("/onyx/opencv/status")).json();
      if (st.conflict && !st.dismissed && !st.fix_scheduled) showDialog(st);
    } catch { /* route missing: older pack or check disabled */ }
  },
});
