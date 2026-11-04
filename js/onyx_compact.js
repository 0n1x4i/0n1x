// Onyx — Copyright (c) 2026 ONYX-AI. All rights reserved.
// Proprietary software, licensed and not sold. See LICENSE.txt at the root of the pack.
//
// Compact layout for the Onyx nodes that draw their own panel (Studio, Post Studio):
// no ComfyUI title bar, no slot list. The ports become small coloured dots packed on
// the left / right edges (name shown by ComfyUI on hover), the panel starts at the
// very top of the node, and the panel's header is the drag handle.
import { app } from "../../scripts/app.js";

const TOP = 16;       // first dot, px from the top of the node
const GAP = 13;       // between two dots

export function makeCompact(node, dragHandle) {
  try { return _makeCompact(node, dragHandle); }
  catch (e) { console.warn("[Onyx] compact layout skipped:", e); return () => {}; }
}

function _makeCompact(node, dragHandle) {
  const NO_TITLE = (typeof LiteGraph !== "undefined" && LiteGraph.NO_TITLE) || 1;
  // Recent frontends expose some of these as getters only (read from the class): set what
  // can be set, on the class first, and never let a refused assignment break the load.
  const trySet = (obj, key, val) => {
    try {
      const d = Object.getOwnPropertyDescriptor(obj, key) || Object.getOwnPropertyDescriptor(Object.getPrototypeOf(obj) || {}, key);
      if (d && d.get && !d.set) return false;
      obj[key] = val;
      return true;
    } catch { return false; }
  };
  const titleHidden = trySet(node.constructor, "title_mode", NO_TITLE) || trySet(node, "title_mode", NO_TITLE);
  node.__onyxCompact = titleHidden;          // the theme draws no title band only when there is none
  trySet(node, "widgets_up", true);
  trySet(node, "widgets_start_y", 4);

  const place = () => {
    const w = node.size?.[0] || 0;
    (node.inputs || []).forEach((s, i) => { trySet(s, "label", " "); trySet(s, "pos", [0, TOP + i * GAP]); });
    (node.outputs || []).forEach((s, i) => { trySet(s, "label", " "); trySet(s, "pos", [w, TOP + i * GAP]); });
    node.setDirtyCanvas?.(true, true);
  };
  place();
  for (const hook of ["onResize", "onConfigure", "onConnectionsChange"]) {
    const orig = node[hook];
    node[hook] = function (...a) { const r = orig?.apply(this, a); place(); return r; };
  }
  setTimeout(place, 0);
  setTimeout(place, 400);

  if (dragHandle) enableDrag(node, dragHandle);
  return place;
}

// The panel covers the whole node, so LiteGraph never sees a click to drag it: the
// header does it (buttons, inputs and selects inside the header keep working).
function enableDrag(node, el) {
  el.style.cursor = "grab";
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest("button, input, select, textarea, a, [data-nodrag]")) return;
    const canvas = app.canvas;
    if (!canvas) return;
    e.preventDefault(); e.stopPropagation();
    try { canvas.selectNode(node, e.shiftKey); } catch { /* older frontend */ }
    const scale = canvas.ds?.scale || 1;
    const start = [e.clientX, e.clientY], from = [...node.pos];
    el.setPointerCapture?.(e.pointerId);
    el.style.cursor = "grabbing";
    const move = (ev) => {
      node.pos[0] = from[0] + (ev.clientX - start[0]) / scale;
      node.pos[1] = from[1] + (ev.clientY - start[1]) / scale;
      canvas.setDirty(true, true);
    };
    const up = (ev) => {
      el.releasePointerCapture?.(ev.pointerId);
      el.style.cursor = "grab";
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      try { app.graph?.change?.(); } catch { /* ignore */ }
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
}
