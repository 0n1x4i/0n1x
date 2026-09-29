// Onyx Session: the key is typed once, sent to /onyx_session/key (saved in a
// local file by the Python side) and removed from the node, so it is never
// written in a saved / shared workflow nor in the queued prompt.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function status() {
  try { return await (await api.fetchApi("/onyx_session/key")).json(); } catch { return {}; }
}

app.registerExtension({
  name: "Onyx.SessionKey",
  async nodeCreated(node) {
    if (node.comfyClass !== "OnyxSessionNode") return;
    const w = (node.widgets || []).find(x => x.name === "key");
    if (!w) return;
    w.serializeValue = () => "";              // never in the workflow / prompt
    const show = s => {
      const el = w.inputEl;
      if (el) el.placeholder = s.saved ? `Key saved on this computer (${s.masked}) — paste a new one to change it`
                                       : "Paste your Onyx key (saved on this computer only)";
    };
    const save = async value => {
      const key = String(value || "").trim();
      if (!key) return;
      try {
        const r = await api.fetchApi("/onyx_session/key", { method: "POST", body: JSON.stringify({ key }),
                                                           headers: { "Content-Type": "application/json" } });
        const d = await r.json();
        if (d.ok) { w.value = ""; show(d); app.graph.setDirtyCanvas(true); }
      } catch (e) { console.warn("[Onyx Session] could not save the key", e); }
    };
    const cb = w.callback;
    w.callback = function (v, ...rest) { save(v); return cb ? cb.call(this, v, ...rest) : undefined; };
    if (w.inputEl) w.inputEl.addEventListener("change", () => save(w.inputEl.value));
    // an old workflow still carrying a key: move it to the local file, then clear it
    setTimeout(async () => {
      const s = await status();
      if (w.value && String(w.value).trim()) {
        if (!s.saved) await save(w.value); else { w.value = ""; show(s); }
      } else show(s);
    }, 0);
  },
});
