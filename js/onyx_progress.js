// Onyx — whole-workflow progress, time left, and readable error hints.
//
// ComfyUI shows the progress of the CURRENT node only. This badge (bottom
// right, while a job runs) shows the whole job: every node of the running
// prompt counts for 1, and each sampler counts for its number of steps, since
// that is where the time goes. Nodes answered from ComfyUI's cache count as
// done instantly. Time left = remaining weight / speed measured on THIS job
// (no fixed per-model guess, so it adapts to the GPU and the settings).
//
// Errors: when a run fails with a known cause (GPU memory, missing model,
// Onyx session, API key, quota, mismatched LoRA/VAE), a toast adds the one
// line that says what to do. ComfyUI's own error dialog is unchanged.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STATE = {
  promptId: null, weights: {}, total: 0, done: new Set(), current: null, frac: 0,
  startedAt: 0, workStartedAt: 0, queueRemaining: 0, hideTimer: null,
};

// ── weights ──────────────────────────────────────────────────────────────────
function nodeWeight(node) {
  const i = node?.inputs || {};
  if (typeof i.steps === "number" && i.steps > 0) {
    const start = typeof i.start_at_step === "number" ? i.start_at_step : 0;
    const end = typeof i.end_at_step === "number" ? Math.min(i.end_at_step, i.steps) : i.steps;
    return Math.max(1, end - start);
  }
  return 1;
}

async function loadWeights(promptId) {
  try {
    const q = await (await api.fetchApi("/queue")).json();
    const entry = (q.queue_running || []).find((e) => e[1] === promptId);
    const prompt = entry?.[2];
    if (!prompt) return;
    STATE.weights = Object.fromEntries(Object.entries(prompt).map(([id, n]) => [id, nodeWeight(n)]));
    STATE.total = Object.values(STATE.weights).reduce((a, b) => a + b, 0);
  } catch { /* badge falls back to the node count */ }
}

function doneWeight() {
  let w = 0;
  for (const id of STATE.done) w += STATE.weights[id] ?? 1;
  if (STATE.current && !STATE.done.has(STATE.current)) w += (STATE.weights[STATE.current] ?? 1) * STATE.frac;
  return w;
}

// ── badge ────────────────────────────────────────────────────────────────────
let badge = null;
function ensureBadge() {
  if (badge) return badge;
  badge = document.createElement("div");
  badge.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:9000;background:rgba(20,20,26,.92);" +
    "color:#ececf2;border:1px solid #34343e;border-radius:8px;padding:7px 11px;font:12px/1.35 system-ui,sans-serif;" +
    "min-width:190px;pointer-events:none;display:none;box-shadow:0 4px 18px rgba(0,0,0,.35)";
  badge.innerHTML = '<div class="t"></div><div style="height:4px;background:#2b2b33;border-radius:2px;margin-top:5px">' +
    '<div class="b" style="height:4px;width:0;background:#9fb4ff;border-radius:2px;transition:width .3s"></div></div>';
  document.body.append(badge);
  return badge;
}

function fmt(sec) {
  if (!isFinite(sec) || sec < 0) return "";
  sec = Math.round(sec);
  const m = Math.floor(sec / 60), s = sec % 60;
  return m >= 60 ? `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

function nodeTitle(id) {
  const n = app.graph?.getNodeById?.(Number(id));
  return n?.title || n?.type || "";
}

function render() {
  if (!STATE.promptId) return;
  const b = ensureBadge();
  const total = STATE.total || Math.max(1, Object.keys(STATE.weights).length);
  const done = doneWeight();
  const pct = Math.min(99, Math.floor((done / total) * 100));
  let eta = "";
  const workElapsed = STATE.workStartedAt ? (performance.now() - STATE.workStartedAt) / 1000 : 0;
  if (workElapsed > 4 && done > 0) eta = fmt(((total - done) / done) * workElapsed);
  const parts = [`Onyx ▸ ${pct}%`];
  if (eta) parts.push(`~${eta} left`);
  if (STATE.queueRemaining > 1) parts.push(`${STATE.queueRemaining - 1} more in queue`);
  const cur = STATE.current ? nodeTitle(STATE.current) : "";
  b.querySelector(".t").textContent = parts.join(" · ") + (cur ? `\n${cur}` : "");
  b.querySelector(".t").style.whiteSpace = "pre";
  b.querySelector(".b").style.width = `${pct}%`;
  b.style.display = "block";
}

function finish(text) {
  if (!badge || !STATE.promptId) return;
  badge.querySelector(".t").textContent = text;
  badge.querySelector(".b").style.width = "100%";
  STATE.promptId = null;
  clearTimeout(STATE.hideTimer);
  STATE.hideTimer = setTimeout(() => { if (!STATE.promptId && badge) badge.style.display = "none"; }, 3500);
}

// ── error hints ──────────────────────────────────────────────────────────────
const HINTS = [
  [/out of memory|OutOfMemoryError|CUDA error: out of memory|not enough memory|allocat(e|ion) .*GiB/i,
    "GPU memory full: lower the resolution, the frame count or the batch size (or use a lighter quantisation), then re-queue."],
  [/Sync error SS-205/i, "Place the Onyx Session node anywhere in this workflow, run it once, then re-queue."],
  [/HTTP 401|HTTP 403|API key (missing|invalid|is required)|PERMISSION_DENIED|UNAUTHENTICATED/i,
    "API key refused: check the key (and that billing / model access is enabled on that provider)."],
  [/HTTP 429|quota|rate.?limit|RESOURCE_EXHAUSTED/i, "Provider quota or rate limit reached: wait a minute or lower the batch size."],
  [/\.(safetensors|gguf|ckpt|pth?|onnx)\b.*(not found|No such file|missing)|(not found|No such file|missing).*\.(safetensors|gguf|ckpt|pth?|onnx)\b|FileNotFoundError/i,
    "A model file is missing: right-click the canvas → \"Onyx: Check workflow setup\" for the folder and the link."],
  [/size mismatch|shape .* (is invalid|mismatch)|Error\(s\) in loading state_dict|expected .* channels/i,
    "A model, LoRA or VAE does not match the others (wrong family or version): check the loaders."],
];

function hintFor(text) {
  for (const [re, hint] of HINTS) if (re.test(text)) return hint;
  return null;
}

app.registerExtension({
  name: "Onyx.Progress",
  setup() {
    api.addEventListener("status", ({ detail }) => {
      STATE.queueRemaining = detail?.exec_info?.queue_remaining ?? 0;
    });
    api.addEventListener("execution_start", ({ detail }) => {
      clearTimeout(STATE.hideTimer);
      Object.assign(STATE, { promptId: detail.prompt_id, weights: {}, total: 0, done: new Set(),
        current: null, frac: 0, startedAt: performance.now(), workStartedAt: 0 });
      loadWeights(detail.prompt_id).then(render);
    });
    api.addEventListener("execution_cached", ({ detail }) => {
      if (detail.prompt_id !== STATE.promptId) return;
      for (const id of detail.nodes || []) STATE.done.add(String(id));
      render();
    });
    api.addEventListener("executing", ({ detail }) => {
      const d = typeof detail === "object" && detail !== null ? detail : { node: detail };
      if (d.prompt_id && d.prompt_id !== STATE.promptId) return;
      if (STATE.current) STATE.done.add(STATE.current);
      if (d.node == null) { finish("Onyx ▸ done"); return; }
      STATE.current = String(d.node);
      STATE.frac = 0;
      if (!STATE.workStartedAt) STATE.workStartedAt = performance.now();
      render();
    });
    api.addEventListener("progress", ({ detail }) => {
      if (detail.prompt_id && detail.prompt_id !== STATE.promptId) return;
      if (detail.max > 0) STATE.frac = Math.min(1, detail.value / detail.max);
      if (detail.node != null) STATE.current = String(detail.node);
      render();
    });
    api.addEventListener("execution_interrupted", () => finish("Onyx ▸ cancelled"));
    api.addEventListener("execution_error", ({ detail }) => {
      finish("Onyx ▸ failed");
      const msg = String(detail?.exception_message || "");
      const hint = hintFor(`${detail?.exception_type || ""} ${msg}`);
      if (!hint) return;
      const where = nodeTitle(detail?.node_id) || detail?.node_type || "a node";
      const first = msg.split("\n").find((l) => l.trim()) || msg;
      const text = `${where}: ${first.slice(0, 220)}\n→ ${hint}`;
      if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({ severity: "warn", summary: "What to do", detail: text, life: 12000 });
      } else {
        console.warn("[Onyx] " + text);
      }
    });
  },
});
