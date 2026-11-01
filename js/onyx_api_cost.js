// Onyx — API cost badge.
// Shows, on the API nodes, what the next run should cost (updated live as the
// widgets change) and, after a run, what was actually counted in the ledger.
// Prices: utils/registry_defaults.json ("pricing"), refreshed from the Worker.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// node type -> how to turn its widgets into an estimate request
const NODES = {
  OnyxNanoBananaAIO: (w) => ({ node_type: "OnyxNanoBananaAIO", params: w }),
  NanoBananaProEditAPINode: (w) => ({ node_type: "simple",
    params: { model: "Nano Banana Pro", provider: "GOOGLE", image_size: w.image_size || "2K", images: 1 } }),
  NanoBanana2EditAPINode: (w) => ({ node_type: "simple",
    params: { model: "Nano Banana 2", provider: "GOOGLE", image_size: w.image_size || "1K", images: 1 } }),
  SeedreamEditAPINode: (w) => ({ node_type: "simple",
    params: { model: "Seedream 4.5", provider: "WAVESPEED", image_size: "*", images: w.num_images || 1 } }),
  SeedreamEditFalAPINode: (w) => ({ node_type: "simple",
    params: { model: "Seedream 4.5", provider: "FAL", image_size: "*", images: w.num_images || 1 } }),
};

// Never send credentials to the estimate route.
const SECRET = /api_?key|apikey|json_folder/i;

function widgetValues(node) {
  const out = {};
  for (const w of node.widgets || []) {
    if (!w.name || SECRET.test(w.name)) continue;
    const v = w.value;
    if (v === null || ["string", "number", "boolean"].includes(typeof v)) out[w.name] = v;
  }
  return out;
}

const PRICE_WIDGETS = ["provider", "model", "image_size", "batch_size", "video_duration",
  "video_mode_enabled", "gpt2_image_quality", "max_black_retries", "face_swap_enabled", "num_images"];

function priceSignature(node) {
  const v = widgetValues(node);
  return PRICE_WIDGETS.map((k) => String(v[k])).join("|");
}

async function refresh(node) {
  const build = NODES[node.comfyClass];
  if (!build) return;
  try {
    const r = await api.fetchApi("/onyx/costs/estimate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(build(widgetValues(node))),
    });
    if (!r.ok) return;
    const est = await r.json();
    node._onyxCost = { text: est.label, notes: est.notes || [], state: est.confidence };
    node.setDirtyCanvas?.(true, false);
  } catch { /* server without the route */ }
}

function drawBadge(node, ctx) {
  const c = node._onyxCost;
  if (!c || node.flags?.collapsed) return;
  ctx.save();
  ctx.font = "11px system-ui, sans-serif";
  const text = c.text;
  const w = ctx.measureText(text).width + 12;
  const x = node.size[0] - w - 6;
  const y = -LiteGraph.NODE_TITLE_HEIGHT - 22;
  ctx.fillStyle = c.state === "cache" ? "#1f6f43"
                : c.state === "unknown" ? "#5a4a1a"
                : c.state === "done" ? "#2d3b6b" : "#3a3550";
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, 18, 5); else ctx.rect(x, y, w, 18);
  ctx.fill();
  ctx.fillStyle = "#f0eefc";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 6, y + 9);
  ctx.restore();
}

app.registerExtension({
  name: "Onyx.ApiCost",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES[nodeData.name]) return;

    // The AIO's own JS flips widgets (video mode, model list) without firing
    // their callbacks, so instead of hooking callbacks we compare a signature of
    // the price-relevant widgets on each redraw and re-estimate when it moves.
    const onDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDraw?.apply(this, arguments);
      const sig = priceSignature(this);
      if (sig !== this._onyxCostSig) {
        this._onyxCostSig = sig;
        clearTimeout(this._onyxCostTimer);
        this._onyxCostTimer = setTimeout(() => refresh(this), 300);
      }
      drawBadge(this, ctx);
      return r;
    };

    const onTitle = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
      const r = onTitle?.apply(this, arguments);
      options.push({
        content: "Onyx: API spend (today / month)",
        callback: async () => {
          try {
            const s = await (await api.fetchApi("/onyx/costs/summary")).json();
            const top = Object.entries(s.by_model_month || {}).slice(0, 6)
              .map(([k, v]) => `  ${k}: $${v.toFixed(2)}`).join("\n");
            alert(`Estimated API spend (list prices, ${s.currency})\n\n` +
                  `This session: $${s.session.total.toFixed(2)} (${s.session.runs} runs)\n` +
                  `Today: $${s.today.toFixed(2)}\nThis month: $${s.month.toFixed(2)}\n` +
                  `Saved by cache this month: $${s.saved_by_cache_month.toFixed(2)}\n` +
                  (s.runs_without_price_month ? `Runs without a known price: ${s.runs_without_price_month}\n` : "") +
                  (top ? `\nBy model this month:\n${top}\n` : "") +
                  `\nPrices updated: ${s.prices_updated || "?"}. Check your provider dashboard for the real bill.`);
          } catch (e) { alert("Cost summary unavailable: " + e); }
        },
      });
      return r;
    };
  },

  setup() {
    // Pushed by the Python side when a run starts / ends (utils/onyx_cost.notify_node).
    api.addEventListener("onyx.api_cost", ({ detail }) => {
      const node = app.graph?.getNodeById?.(Number(detail.node));
      if (!node) return;
      node._onyxCost = detail.final
        ? { text: detail.cache_hit ? "cache hit · $0 (not billed)" : `last run: ${detail.label}`,
            state: detail.cache_hit ? "cache" : "done" }
        : { text: detail.label, state: detail.confidence };
      node.setDirtyCanvas?.(true, false);
    });
  },
});
