# -*- coding: utf-8 -*-
"""
Onyx parallel — run several provider calls at once, safely.

API calls spend 95 % of their time waiting on the provider, so a carousel of 8
slides done one after the other waits 8 times. Threads are enough (the GIL is
released while waiting on the network); no asyncio rewrite needed.

    from ..utils.onyx_parallel import map_parallel
    images = map_parallel(generate_one, slides, max_workers=4, label="Re-pose")

Guarantees:
  * Results come back in the SAME order as the inputs.
  * The first failure cancels the calls not started yet and is re-raised, so a
    bad API key stops the batch at once instead of failing 8 times (and a
    paid call is never started after the batch is already doomed).
    Pass `return_exceptions=True` to get the exception objects in the result
    list instead (keep the 7 good slides when 1 fails).
  * ComfyUI's Cancel button is honoured between calls.
  * A ComfyUI progress bar advances as calls finish, when running inside ComfyUI.
  * `max_workers` is capped (default 4): providers rate-limit per key, and 10
    parallel calls on one Google key mostly buy 429s. onyx_http retries those.

Used locally by Node.py (result downloads). The Worker modules (carousel,
Re-pose, Dataset Creator) run inside the pack's package and can import it with
`from ..utils.onyx_parallel import map_parallel`.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, FIRST_EXCEPTION, ThreadPoolExecutor, wait

MAX_WORKERS_CAP = int(os.environ.get("ONYX_MAX_PARALLEL", "6") or 6)


def _interrupted() -> bool:
    try:
        import comfy.model_management as mm
        return bool(mm.processing_interrupted())
    except Exception:
        return False


def _progress(total: int):
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(total)
    except Exception:
        return None


def map_parallel(fn, items, *, max_workers: int = 4, label: str = "",
                 return_exceptions: bool = False, progress: bool = True) -> list:
    items = list(items)
    if not items:
        return []
    workers = max(1, min(int(max_workers), MAX_WORKERS_CAP, len(items)))
    if workers == 1:
        out = []
        for it in items:
            if _interrupted():
                raise RuntimeError(f"{label or 'Batch'}: cancelled.")
            try:
                out.append(fn(it))
            except Exception as e:
                if not return_exceptions:
                    raise
                out.append(e)
        return out

    pbar = _progress(len(items)) if progress else None
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="onyx-par") as pool:
        futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.5,
                                 return_when=FIRST_EXCEPTION if not return_exceptions else FIRST_COMPLETED)
            for f in done:
                i = futures[f]
                exc = f.exception()
                if exc is not None and not return_exceptions:
                    for p in pending:
                        p.cancel()
                    raise exc
                results[i] = exc if exc is not None else f.result()
                if pbar is not None:
                    pbar.update(1)
            if _interrupted():
                for p in pending:
                    p.cancel()
                raise RuntimeError(f"{label or 'Batch'}: cancelled.")
    return results
