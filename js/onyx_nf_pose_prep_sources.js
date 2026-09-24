// Pure graph helpers shared by the preparation UI and regression checks.
export function connectedNode(node, inputName) {
  const slot = node.inputs?.find((i) => i.name === inputName);
  const link = slot?.link != null ? node.graph?.links?.[slot.link] : null;
  const id = Array.isArray(link) ? link[1] : link?.origin_id;
  return id == null ? null : node.graph?.getNodeById?.(id);
}

export function masterSource(node) {
  let upstream = connectedNode(node, "master_image");
  const visited = new Set();
  while (upstream && !visited.has(upstream.id)) {
    visited.add(upstream.id);
    if (upstream.type === "LoadImage") {
      const filename = String(upstream.widgets?.find((w) => w.name === "image")?.value || "");
      return filename ? { node_id: upstream.id, filename } : null;
    }
    if (["ImageScaleToTotalPixels", "ImageScale"].includes(upstream.type)) {
      upstream = connectedNode(upstream, "image");
    } else if (upstream.type === "Reroute") {
      upstream = connectedNode(upstream, upstream.inputs?.[0]?.name);
    } else {
      return null;
    }
  }
  return null;
}

export function imageViewQuery(source) {
  let filename = source.filename.replaceAll("\\", "/");
  const annotation = filename.match(/ \[(input|output|temp)\]$/);
  const type = annotation?.[1] || "input";
  if (annotation) filename = filename.slice(0, -annotation[0].length);
  const parts = filename.split("/");
  const query = new URLSearchParams({ filename: parts.pop(), type });
  if (parts.length) query.set("subfolder", parts.join("/"));
  return query.toString();
}

export function poseLines(text) {
  return [...new Set(String(text || "").split("\n")
    .map((line) => line.replace(/^\s*(?:[-*•]|\d+[.)])\s*/, "").trim()).filter(Boolean))];
}

export function preparationSignature(state, master, references) {
  const copy = state.mode === "Reference Copy";
  return JSON.stringify({
    version: 3, mode: state.mode, backend: state.backend, model_id: state.model_id,
    endpoint: state.endpoint, detail: state.detail,
    vibe: copy ? null : state.vibe, count: copy ? null : state.count,
    selfie: copy ? null : state.include_selfie, master,
    refs: copy ? references.map((x) => [x.id, x.filename]) : [],
  });
}
