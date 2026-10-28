// Onyx — widget value sanitizer.
//
// When a node gains or loses widgets between versions, an older workflow's
// widgets_values can land on the wrong widgets: a button's `null` ends up in a
// text widget, an "" in a boolean, "1" in a number. Those values are then saved
// again and survive forever. A null text widget makes other extensions crash
// while the prompt is built (comfyui-custom-scripts calls value.replace), which
// surfaces as "Cannot read properties of undefined (reading 'output')" and
// blocks every run.
//
// This repairs such values right after a node is configured from a workflow,
// for every Onyx node, using the defaults declared by the Python INPUT_TYPES.

import { app } from "../../scripts/app.js";

function inputSpecs(nodeData) {
    const specs = {};
    for (const group of ["required", "optional"]) {
        for (const [name, spec] of Object.entries(nodeData?.input?.[group] || {})) {
            specs[name] = spec;
        }
    }
    return specs;
}

function defaultFor(spec) {
    if (!Array.isArray(spec)) return undefined;
    const [type, opts] = spec;
    if (opts && Object.prototype.hasOwnProperty.call(opts, "default")) return opts.default;
    if (Array.isArray(type)) return type[0];                 // combo: first option
    if (type === "STRING") return "";
    if (type === "BOOLEAN") return false;
    if (type === "INT" || type === "FLOAT") return opts?.min ?? 0;
    return undefined;
}

function sanitizeNode(node, specs) {
    const fixed = [];
    for (const w of node.widgets || []) {
        if (w.type === "button" || w.options?.serialize === false) continue;
        const spec = specs[w.name];
        if (!spec) continue;
        const type = spec[0];
        const def = defaultFor(spec);
        let v = w.value;
        let nv = v;

        if (v === null || v === undefined) {
            nv = def;
        } else if (type === "BOOLEAN" && typeof v !== "boolean") {
            nv = v === "true" ? true : v === "false" ? false : (typeof def === "boolean" ? def : Boolean(v));
        } else if ((type === "INT" || type === "FLOAT") && typeof v !== "number") {
            const n = Number(v);
            nv = Number.isFinite(n) ? (type === "INT" ? Math.round(n) : n) : def;
        } else if (type === "STRING" && typeof v !== "string") {
            nv = String(v);
        } else if (Array.isArray(type) && type.length && !type.includes(v)) {
            // stale combo value from another widget: only fix obvious garbage
            if (typeof v !== "string") nv = def;
        }

        if (nv !== v && nv !== undefined) {
            w.value = nv;
            if (w.inputEl) w.inputEl.value = nv;
            fixed.push(`${w.name}: ${JSON.stringify(v)} -> ${JSON.stringify(nv)}`);
        }
    }
    if (fixed.length) {
        console.warn(`[Onyx] repaired widget values on #${node.id} ${node.type}:`, fixed);
        node.setDirtyCanvas?.(true, true);
    }
}

app.registerExtension({
    name: "Onyx.WidgetSanitizer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const isOnyx = /^Onyx/i.test(nodeData?.name || "") || /^Onyx/i.test(nodeData?.category || "");
        if (!isOnyx) return;
        const specs = inputSpecs(nodeData);
        const orig = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = orig?.apply(this, arguments);
            try { sanitizeNode(this, specs); } catch (e) { console.warn("[Onyx] sanitizer:", e); }
            return r;
        };
    },
});
