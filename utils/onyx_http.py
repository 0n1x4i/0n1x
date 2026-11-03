# -*- coding: utf-8 -*-
"""
Onyx HTTP — one client for every provider call in the pack.

Each request is sent ONCE: no automatic retry, no replay. A failed call fails
the node right away, exactly as before; re-queueing is the user's decision.

What it adds over a bare `requests.post`:
  * One readable error (`OnyxAPIError`) out of the very different error
    bodies of Google, xAI, fal, WaveSpeed, Kie and Atlas Cloud, with a hint
    per status code (bad key, quota, model retired, image too large...).
  * API keys redacted from every message it produces. `requests` puts the full
    URL in its exceptions, and Google's `?key=` style used to land the key in
    the ComfyUI console and in screenshots.
  * A clear message on a read timeout: the provider may still finish (and
    bill) the job, so check its dashboard before re-queueing.

Usage:

    from ..utils.onyx_http import post_json, OnyxAPIError

    data = post_json(url, service="Nano Banana Pro", payload=payload,
                     headers={"x-goog-api-key": key}, timeout=(15, 180))
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

import requests

__all__ = [
    "OnyxAPIError", "request", "post_json", "get_json", "download", "redact",
]

log = logging.getLogger("onyx.http")

DEFAULT_TIMEOUT = (15, 180)        # (connect, read) seconds

_SECRET_HEADERS = ("authorization", "x-goog-api-key", "x-api-key", "api-key", "x-key")

_HINTS = {
    400: "The provider rejected the request (invalid parameter, unsupported option "
         "for this model, or a prompt it refuses).",
    401: "API key missing, invalid or expired.",
    402: "Out of credits on this provider account — top up, then retry.",
    403: "This API key has no access to that model or endpoint (billing not enabled, "
         "region, or model not granted to the project).",
    404: "Model or endpoint not found — it may have been renamed or retired. "
         "Pick another model in the list.",
    408: "The provider timed out waiting for the request.",
    413: "Request too large — send fewer or smaller input images.",
    422: "The provider could not validate the request (a parameter is out of range "
         "or not allowed together with another one).",
    429: "Rate limit or quota reached. Wait a minute before re-queueing, lower the batch size, or check "
         "the quota/billing page of this provider.",
    500: "Internal error on the provider side.",
    502: "Provider gateway error — usually transient.",
    503: "Provider overloaded or in maintenance — usually transient.",
    504: "Provider gateway timeout — usually transient.",
    529: "Provider overloaded — usually transient.",
}


class OnyxAPIError(RuntimeError):
    """Readable provider error. `str(err)` is safe to show: keys are redacted."""

    def __init__(self, service: str, message: str, status: Optional[int] = None,
                 provider_message: str = ""):
        super().__init__(message)
        self.service = service
        self.status = status
        self.provider_message = provider_message


# ─────────────────────────────────────────────────────────────────────────────
# Redaction
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_SECRET = re.compile(r"([?&](?:key|api_key|apikey|token|access_token)=)[^&\s'\"]+", re.I)
_BEARER = re.compile(r"((?:Bearer|Key)\s+)[A-Za-z0-9._\-:]{8,}")


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove API keys from `text`: query-string keys, bearer tokens, and any
    explicit secret passed in (header values)."""
    if not text:
        return text
    out = _QUERY_SECRET.sub(r"\1***", str(text))
    out = _BEARER.sub(r"\1***", out)
    for s in secrets:
        s = (s or "").strip()
        if len(s) >= 6:
            out = out.replace(s, "***")
    return out


def _secrets_from(headers: Optional[dict], params: Optional[dict]) -> list:
    found = []
    for k, v in (headers or {}).items():
        if k.lower() in _SECRET_HEADERS and v:
            v = str(v)
            found.append(v)
            # "Bearer xxx" / "Key xxx": also redact the bare token
            if " " in v:
                found.append(v.split(" ", 1)[1])
    for k, v in (params or {}).items():
        if k.lower() in ("key", "api_key", "apikey", "token") and v:
            found.append(str(v))
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Error body parsing
# ─────────────────────────────────────────────────────────────────────────────

def _provider_message(resp: requests.Response) -> str:
    """Best-effort extraction of the human message from any provider's error body."""
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:400]
    return _message_from_body(body)[:600]


def _message_from_body(body) -> str:
    if isinstance(body, str):
        return body
    if not isinstance(body, dict):
        return str(body)
    err = body.get("error")
    if isinstance(err, dict):                       # Google / xAI / OpenAI shape
        msg = err.get("message") or err.get("status") or ""
        code = err.get("status") or err.get("code") or ""
        return f"{msg} ({code})" if code and str(code) not in msg else str(msg)
    if isinstance(err, str) and err:
        return err
    detail = body.get("detail")                     # fal.ai / FastAPI shape
    if isinstance(detail, list):
        parts = []
        for d in detail[:3]:
            if isinstance(d, dict):
                loc = ".".join(str(x) for x in d.get("loc", []) if x != "body")
                parts.append(f"{loc}: {d.get('msg', '')}" if loc else str(d.get("msg", d)))
            else:
                parts.append(str(d))
        return "; ".join(parts)
    if detail:
        return str(detail)
    for key in ("message", "msg", "error_message", "reason"):   # WaveSpeed / Kie / Atlas
        if body.get(key):
            return str(body[key])
    data = body.get("data")
    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])
    return str(body)[:300]


def _build_error(service: str, resp: requests.Response, secrets) -> OnyxAPIError:
    status = resp.status_code
    pmsg = redact(_provider_message(resp), secrets)
    hint = _HINTS.get(status) or ("Provider-side error." if status >= 500 else "Request refused.")
    msg = f"{service}: HTTP {status} — {hint}"
    if pmsg:
        msg += f"\n  Provider says: {pmsg}"
    return OnyxAPIError(service, msg, status=status, provider_message=pmsg)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def request(method: str, url: str, *, service: str = "API",
            headers: Optional[dict] = None, params: Optional[dict] = None,
            json=None, data=None, timeout=DEFAULT_TIMEOUT,
            raise_for_status: bool = True,
            session: Optional[requests.Session] = None,
            **kwargs) -> requests.Response:
    """Send one HTTP request, once (no retry).

    Returns the `requests.Response` (status < 400 unless raise_for_status=False).
    Raises OnyxAPIError on failure, never a raw requests exception.
    """
    secrets = _secrets_from(headers, params)
    sender = session or requests
    try:
        resp = sender.request(method, url, headers=headers, params=params,
                              json=json, data=data, timeout=timeout, **kwargs)
    except requests.exceptions.ReadTimeout:
        read_s = timeout[1] if isinstance(timeout, tuple) else timeout
        raise OnyxAPIError(
            service,
            f"{service}: no answer within {read_s}s. The provider may still finish (and bill) "
            f"this job — check the provider dashboard before re-queueing.",
        ) from None
    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as e:
        raise OnyxAPIError(
            service,
            f"{service}: could not reach the provider (network down, DNS, proxy or firewall). "
            f"Details: {redact(str(e), secrets)[:300]}",
        ) from None
    except requests.exceptions.RequestException as e:
        raise OnyxAPIError(service, f"{service}: request error — {redact(str(e), secrets)[:300]}") from None

    if resp.status_code < 400 or not raise_for_status:
        return resp
    raise _build_error(service, resp, secrets)


def post_json(url: str, *, service: str = "API", payload=None, **kwargs) -> dict:
    """POST `payload` as JSON and return the decoded JSON body."""
    resp = request("POST", url, service=service, json=payload, **kwargs)
    return _decode(resp, service, kwargs)


def get_json(url: str, *, service: str = "API", **kwargs) -> dict:
    """GET and return the decoded JSON body."""
    resp = request("GET", url, service=service, **kwargs)
    return _decode(resp, service, kwargs)


def download(url: str, *, service: str = "API", timeout=(15, 120), **kwargs) -> bytes:
    """Fetch a result file (image/video) and return its bytes."""
    resp = request("GET", url, service=f"{service} download", timeout=timeout, **kwargs)
    return resp.content


def _decode(resp: requests.Response, service: str, kwargs) -> dict:
    try:
        return resp.json()
    except ValueError:
        secrets = _secrets_from(kwargs.get("headers"), kwargs.get("params"))
        snippet = redact((resp.text or "")[:300], secrets)
        raise OnyxAPIError(service, f"{service}: the provider answered HTTP {resp.status_code} "
                                    f"but not with JSON: {snippet!r}",
                           status=resp.status_code) from None
