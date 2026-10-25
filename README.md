# Onyx Glass Lab

Standalone preview of the "Glassmorphism" node theme for the Onyx pack.
One demo node, zero overlap with the real Onyx pack: install both side by side.

## Install

1. Download the ZIP of this branch and extract it.
2. Rename the extracted folder to `OnyxGlassLab` (no spaces, no dashes — ComfyUI
   cannot serve JS from folders with spaces in the name).
3. Move it to `ComfyUI/custom_nodes/OnyxGlassLab`.
4. Restart ComfyUI, then hard-refresh the browser (Ctrl+Shift+R).
5. Add the node **Onyx / Glass Lab / Onyx Glass Lab — Demo**.

## What to look at

- Normal state: dark neutral body, thin accent rim, "ONYX" chip with LED and lock.
- Click the node: stronger accent ring (selected state).
- Toggle `simulate_sync_error` and queue: red glass error banner with the
  SS-204 text inside the node.
- Collapse the node (click the dot in the title): chrome shrinks to the title.
- The "Glass widgets" panel at the bottom of the node shows the DOM widget
  library (segmented toggle, combo, slider, switch, buttons).

## Tuning

Every colour lives in the `PALETTE` block at the top of `js/glass_lab_theme.js`.

## Remove

Delete `ComfyUI/custom_nodes/OnyxGlassLab`.
