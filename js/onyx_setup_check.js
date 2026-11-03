// Onyx Setup Check — lists what the open workflow needs and this ComfyUI lacks:
// node packs (with the pack name and link) and model files (with the models/
// folder and the download link), BEFORE you queue.
//
// Runs by itself when a workflow containing Onyx nodes is loaded (silent when
// everything is there), and on demand: right-click the canvas ->
// "Onyx: Check workflow setup". Backend: utils/onyx_setup_check.py.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MODEL_RE = /\.(safetensors|gguf|ckpt|pt|pth|bin|onnx|sft)$/i;

function allNodes(graph, out = []) {
  for (const n of graph?._nodes || graph?.nodes || []) {
    out.push(n);
    if (n.subgraph) allNodes(n.subgraph, out);
  }
  return out;
}

function collect() {
  const nodeTypes = new Set();
  const models = [];
  for (const n of allNodes(app.graph)) {
    if (n.mode === 2 || n.mode === 4) continue;          // muted / bypassed: will not run
    if (n.type) nodeTypes.add(n.type);
    for (const w of n.widgets || []) {
      if (typeof w.value === "string" && MODEL_RE.test(w.value)) {
        models.push({ node_type: n.type, node_title: n.title || n.type, widget: w.name, value: w.value });
      }
    }
  }
  return { node_types: [...nodeTypes], models };
}

function el(tag, css, text) {
  const e = document.createElement(tag);
  if (css) e.style.cssText = css;
  if (text !== undefined) e.textContent = text;
  return e;
}

function link(url, label) {
  if (!url) return el("span", "opacity:.6", " — link unknown");
  const a = el("a", "color:#9fb4ff;margin-left:6px", label || "download");
  a.href = url; a.target = "_blank"; a.rel = "noopener";
  return a;
}

function showReport(r, { manual }) {
  if (r.ok) {
    if (manual) {
      const msg = `Onyx Setup Check: everything is installed (${r.checked_models} model file(s) checked).`;
      if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({ severity: "success", summary: "Onyx Setup Check", detail: msg, life: 4000 });
      } else alert(msg);
    }
    return;
  }
  const overlay = el("div", "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;align-items:center;justify-content:center");
  const box = el("div", "background:#1e1e24;color:#e8e8ee;max-width:680px;width:calc(100% - 32px);max-height:80vh;overflow:auto;" +
    "border-radius:10px;padding:18px 20px;font:13px/1.5 system-ui,sans-serif;box-shadow:0 10px 40px rgba(0,0,0,.5)");
  box.append(el("div", "font-size:16px;font-weight:600;margin-bottom:10px", "Onyx Setup Check — missing for this workflow"));
  const lines = [];

  if (r.missing_nodes.length) {
    box.append(el("div", "font-weight:600;margin:6px 0 4px", `Node packs (${r.missing_nodes.length})`));
    const byPack = {};
    for (const m of r.missing_nodes) (byPack[m.pack || "unknown pack"] ||= { url: m.url, types: [] }).types.push(m.type);
    for (const [pack, info] of Object.entries(byPack)) {
      const row = el("div", "margin:2px 0 2px 10px");
      row.append(el("b", "", pack), document.createTextNode(` — ${info.types.join(", ")}`), link(info.url, "get it"));
      box.append(row);
      lines.push(`[pack] ${pack}: ${info.types.join(", ")} ${info.url || ""}`);
    }
    box.append(el("div", "opacity:.7;margin:2px 0 0 10px;font-size:12px",
      "Install into ComfyUI/custom_nodes/, run its requirements.txt with ComfyUI's Python, restart ComfyUI."));
  }
  if (r.missing_models.length) {
    box.append(el("div", "font-weight:600;margin:12px 0 4px", `Model files (${r.missing_models.length})`));
    for (const m of r.missing_models) {
      const row = el("div", "margin:2px 0 2px 10px");
      row.append(el("code", "", m.file),
        document.createTextNode(` → models/${m.folder || "?"}/  (${m.node_title}.${m.widget})`), link(m.url));
      box.append(row);
      lines.push(`[model] models/${m.folder || "?"}/${m.file} ${m.url || ""}`);
    }
  }
  if (r.elsewhere.length) {
    box.append(el("div", "font-weight:600;margin:12px 0 4px", "Present under another name / sub-folder"));
    for (const m of r.elsewhere) {
      box.append(el("div", "margin:2px 0 2px 10px",
        `${m.node_title}.${m.widget}: the workflow asks "${m.file}", you have "${m.found_as}" — re-select it in the widget.`));
      lines.push(`[rename] ${m.file} -> ${m.found_as}`);
    }
  }
  const row = el("div", "display:flex;justify-content:flex-end;gap:8px;margin-top:16px");
  const btn = (label, primary, fn) => {
    const b = el("button", "padding:6px 14px;border-radius:6px;cursor:pointer;" +
      (primary ? "background:#7c5cff;color:#fff;border:1px solid #7c5cff" : "background:transparent;color:inherit;border:1px solid #666"), label);
    b.onclick = fn; return b;
  };
  row.append(
    btn("Copy list", false, () => navigator.clipboard?.writeText(lines.join("\n"))),
    btn("Close", true, () => overlay.remove()));
  box.append(row);
  overlay.append(box);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
  document.body.append(overlay);
}

async function runCheck({ manual = false } = {}) {
  try {
    const r = await (await api.fetchApi("/onyx/setup_check", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(collect()),
    })).json();
    showReport(r, { manual });
  } catch (e) {
    if (manual) alert("Onyx Setup Check unavailable: " + e);
  }
}

app.registerExtension({
  name: "Onyx.SetupCheck",

  async afterConfigureGraph() {
    // Only for workflows that use the pack, and once the graph is fully built.
    const hasOnyx = allNodes(app.graph).some((n) => /^Onyx|^Onyx_/.test(n.type || ""));
    if (hasOnyx) setTimeout(() => runCheck(), 800);
  },

  setup() {
    const proto = globalThis.LGraphCanvas?.prototype;
    if (!proto?.getCanvasMenuOptions) return;
    const orig = proto.getCanvasMenuOptions;
    proto.getCanvasMenuOptions = function () {
      const options = orig.apply(this, arguments) || [];
      options.push(null, { content: "Onyx: Check workflow setup", callback: () => runCheck({ manual: true }) });
      return options;
    };
  },
});
