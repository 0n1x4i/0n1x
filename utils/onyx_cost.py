# -*- coding: utf-8 -*-
"""
Onyx API cost — estimate what a run will cost BEFORE it is queued, and keep a
running total of what was spent.

Prices live in the registry (utils/registry_defaults.json, "pricing" block,
refreshable from the Worker, overridable in _cache/pricing_override.json).

What the estimate is and is not:
  * It is the provider's list price x what this run asks for: images x
    batch_size, or seconds of video x price per second at that resolution.
  * "exact" = the price of that model on that very provider is in the table.
    "indicative" (shown with ≈) = only the model owner's list price is known;
    resellers (Kie, Atlas Cloud, WaveSpeed...) are often cheaper.
    "unknown" = no price at all for this model: the badge says so instead of
    inventing a number. Add it to the table.
  * Not included: replays from max_black_retries (each one is billed), the extra
    calls of the face-swap automation, LLM prompt calls, taxes, free credits.

Ledger: every finished generation appends one line to _cache/api_costs.jsonl
(timestamp, node, provider, model, units, estimated cost, cache hit or not).
GET /onyx/costs/summary returns today / this month / this session totals.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field

from . import onyx_registry as _reg

_LEDGER = os.path.join(_reg.CACHE_DIR, "api_costs.jsonl")
_lock = threading.Lock()
_session = {"started": time.time(), "total": 0.0, "runs": 0, "saved_by_cache": 0.0}


@dataclass
class Estimate:
    kind: str = "image"            # image | video
    provider: str = ""
    model: str = ""
    unit_price: float | None = None
    units: float = 0.0             # images, or seconds
    unit_label: str = "image"
    total: float | None = None
    confidence: str = "unknown"    # exact | indicative | unknown
    currency: str = "USD"
    notes: list = field(default_factory=list)

    def label(self) -> str:
        """Short text for the node badge."""
        if self.total is None:
            return f"cost: unknown for {self.model} on {self.provider}"
        approx = "≈ " if self.confidence != "exact" else ""
        n = int(self.units) if float(self.units).is_integer() else self.units
        unit = self.unit_label + ("s" if n != 1 and self.unit_label == "image" else "")
        s = f"{approx}${self.total:.3f}  ({n} {unit} × ${self.unit_price:.3f})"
        if self.confidence == "indicative":
            s += " · indicative"
        return s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label()
        return d


def _pricing() -> dict:
    return _reg.registry().get("pricing") or {}


def _lookup(section: str, model: str, provider: str, size: str):
    table = (_pricing().get(section) or {}).get(model) or {}
    for prov, confidence in ((provider, "exact"), ("_reference", "indicative")):
        rates = table.get(prov)
        if not isinstance(rates, dict):
            continue
        if size in rates:
            return float(rates[size]), confidence
        if "*" in rates:
            return float(rates["*"]), confidence
    return None, "unknown"


def _is_video(model: str, video_mode: bool) -> bool:
    if video_mode:
        return True
    return bool((_pricing().get("video") or {}).get(model)) and not (_pricing().get("image") or {}).get(model)


def estimate_aio(params: dict) -> Estimate:
    """Estimate for one OnyxNanoBananaAIO run, from its widget values."""
    provider = str(params.get("provider") or "GOOGLE")
    model = str(params.get("model") or "")
    size = str(params.get("image_size") or "2K")
    video = _is_video(model, bool(params.get("video_mode_enabled")))
    est = Estimate(provider=provider, model=model,
                   currency=_pricing().get("currency", "USD"))
    if video:
        est.kind, est.unit_label = "video", "s"
        seconds = float(params.get("video_duration") or 8)
        est.units = seconds
        res = size if size.endswith("p") else "720p"
        est.unit_price, est.confidence = _lookup("video", model, provider, res)
        if model == "Kling 3.0 Motion Control":
            est.notes.append("Duration follows the reference video, not the widget.")
    else:
        batch = max(1, int(params.get("batch_size") or 1))
        est.units = batch
        est.unit_price, est.confidence = _lookup("image", model, provider, size)
        if str(model).startswith("GPT Image"):
            est.notes.append(f"Quality '{params.get('gpt2_image_quality', 'high')}' changes the price.")
    if est.unit_price is not None:
        est.total = round(est.unit_price * est.units, 4)
    retries = int(params.get("max_black_retries") or 0)
    if retries:
        est.notes.append(f"Up to {retries} replay(s) on a black image, each billed.")
    if params.get("face_swap_enabled"):
        est.notes.append("Face-swap automation makes extra calls not counted here.")
    return est


def estimate_simple(model: str, provider: str, size: str = "*", images: int = 1) -> Estimate:
    """Estimate for the single-model nodes of Node.py."""
    est = Estimate(provider=provider, model=model, units=images,
                   currency=_pricing().get("currency", "USD"))
    est.unit_price, est.confidence = _lookup("image", model, provider, size)
    if est.unit_price is not None:
        est.total = round(est.unit_price * images, 4)
    return est


# ─────────────────────────────────────────────────────────────────────────────
# Ledger
# ─────────────────────────────────────────────────────────────────────────────

def record(node: str, est: Estimate, *, cache_hit: bool = False) -> None:
    line = {
        "ts": time.time(), "node": node, "provider": est.provider, "model": est.model,
        "kind": est.kind, "units": est.units, "unit_price": est.unit_price,
        "cost": 0.0 if cache_hit else est.total, "confidence": est.confidence,
        "cache_hit": cache_hit, "saved": est.total if cache_hit else 0.0,
    }
    with _lock:
        _session["runs"] += 1
        if cache_hit:
            _session["saved_by_cache"] += est.total or 0.0
        else:
            _session["total"] += est.total or 0.0
        try:
            os.makedirs(os.path.dirname(_LEDGER), exist_ok=True)
            with open(_LEDGER, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        except OSError:
            pass
    tag = "cache hit, $0" if cache_hit else est.label()
    print(f"💰 [Onyx Cost] {node} · {est.provider} · {est.model}: {tag}")


def summary() -> dict:
    now = time.localtime()
    day_start = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))
    month_start = time.mktime((now.tm_year, now.tm_mon, 1, 0, 0, 0, 0, 0, -1))
    today = month = saved_month = 0.0
    unknown_month = 0
    by_model: dict = {}
    try:
        with open(_LEDGER, encoding="utf-8") as fh:
            for raw in fh:
                try:
                    e = json.loads(raw)
                except ValueError:
                    continue
                if e.get("ts", 0) < month_start:
                    continue
                cost = e.get("cost")
                if cost is None:
                    unknown_month += 1
                    continue
                month += cost
                saved_month += e.get("saved") or 0.0
                if e["ts"] >= day_start:
                    today += cost
                key = f"{e.get('provider')} · {e.get('model')}"
                by_model[key] = round(by_model.get(key, 0.0) + cost, 4)
    except OSError:
        pass
    return {
        "currency": _pricing().get("currency", "USD"),
        "session": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in _session.items()},
        "today": round(today, 4), "month": round(month, 4),
        "saved_by_cache_month": round(saved_month, 4),
        "runs_without_price_month": unknown_month,
        "by_model_month": dict(sorted(by_model.items(), key=lambda kv: -kv[1])),
        "prices_updated": _pricing().get("updated"),
    }


def notify_node(unique_id, est: Estimate, *, final: bool = False, cache_hit: bool = False) -> None:
    """Push the estimate to the node badge (js/onyx_api_cost.js)."""
    if unique_id is None:
        return
    try:
        from server import PromptServer
        payload = {"node": str(unique_id), "final": final, "cache_hit": cache_hit, **est.to_dict()}
        PromptServer.instance.send_sync("onyx.api_cost", payload)
    except Exception:
        pass


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    inst = PromptServer.instance
    if getattr(inst, "_onyx_cost_routes", False):
        return
    inst._onyx_cost_routes = True

    @inst.routes.post("/onyx/costs/estimate")
    async def _estimate(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        node_type = body.get("node_type", "OnyxNanoBananaAIO")
        params = body.get("params") or {}
        if node_type == "OnyxNanoBananaAIO":
            est = estimate_aio(params)
        else:
            est = estimate_simple(params.get("model", ""), params.get("provider", ""),
                                  params.get("image_size", "*"), int(params.get("images", 1) or 1))
        return web.json_response(est.to_dict())

    @inst.routes.get("/onyx/costs/summary")
    async def _summary(_request):
        return web.json_response(summary())
