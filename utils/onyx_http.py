# -*- coding: utf-8 -*-
"""
Onyx HTTP — one client for every provider call in the pack.

Every API node used to call `requests.post` on its own, each with its own
timeout and its own error handling, and none of them retried. One 429 from
Google in the middle of a 40-image batch killed the whole queue.

What this module does, and why each rule is the way it is:

  * Retries 408 / 429 / 500 / 502 / 503 / 504 / 529 with exponential backoff
    and jitter, and honours `Retry-After` when the provider sends it. Those
    statuses mean "the provider did not do the work": nothing was generated,
    nothing was billed.
  * Retries connection errors (DNS, refused, reset before the answer, connect
    timeout): the request never reached the provider.
  * Does NOT retry a READ timeout by default. The provider received the
    request and may still be generating — and billing — the image. Replaying
    it would pay twice for one picture. Pass `retry_read_timeout=True` only
    for idempotent, free calls (polling, listing models).
  * Never retries 400 / 401 / 403 / 404 / 413 / 422: the same request will
    fail the same way, and the user needs the message, not a 30 s wait.
  * Builds one readable error (`OnyxAPIError`) out of the very different
    error bodies of Google, xAI, fal, WaveSpeed, Kie and Atlas Cloud, with a
    hint per status code.
  * Redacts API keys from every message it produces. `requests` puts the full
    URL in its exceptions, and Google's `?key=` style used to land the key in
    the ComfyUI console and in screenshots people post on Discord.
  * Waits in short slices and checks ComfyUI's cancel button between them, so
    "Cancel" works during a backoff instead of after it.

Usage:

    from ..utils.onyx_http import post_json, OnyxAPIError

    data = post_json(url, service="Nano Banana Pro", payload=payload,
                     headers={"x-goog-api-key": key}, timeout=(15, 180))

The protected modules served by the Worker run inside the pack's package (see
nodes/onyx_remote_exec.load_bundle_module), so they can use it the same way:
`from ..utils.onyx_http import post_json`.
"""

from __future__ import annotations

import email.utils
import logging
import random
import re
import time
from typing import Iterable, Optional

import requests

__all__ = [
    "OnyxAPIError", "request", "post_json", "get_json", "download",
    "RETRY_STATUSES", "redact",
]

log = logging.getLogger("onyx.http")

# Statuses where the provider did not run the job, so a replay costs nothing.
# 529 = Anthropic/xAI-style "overloaded".
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})

DEFAULT_TIMEOUT = (15, 180)        # (connect, read) seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0         # 2s, 4s, 8s (+ jitter)
DEFAULT_BACKOFF_MAX = 30.0
MAX_RETRY_AFTER = 120.0            # a provider asking for 10 minutes is a quota, not a blip

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
    429: "Rate limit or quota reached. Wait a minute, lower the batch size, or check "
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
                 provider_message: str = "", retryable: bool = False, attempts: int = 1):
        super().__init__(message)
        self.service = service
        self.status = status
        self.provider_message = provider_message
        self.retryable = retryable
        self.attempts = attempts


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


def _build_error(service: str, resp: requests.Response, secrets, attempts: int) -> OnyxAPIError:
    status = resp.status_code
    pmsg = redact(_provider_message(resp), secrets)
    hint = _HINTS.get(status) or ("Provider-side error." if status >= 500 else "Request refused.")
    msg = f"{service}: HTTP {status} — {hint}"
    if pmsg:
        msg += f"\n  Provider says: {pmsg}"
    if attempts > 1:
        msg += f"\n  (gave up after {attempts} attempts)"
    return OnyxAPIError(service, msg, status=status, provider_message=pmsg,
                        retryable=status in RETRY_STATUSES, attempts=attempts)


# ─────────────────────────────────────────────────────────────────────────────
# Waiting (interruptible)
# ─────────────────────────────────────────────────────────────────────────────

def _check_interrupted():
    try:
        import comfy.model_management as mm  # only inside ComfyUI
    except Exception:
        return
    mm.throw_exception_if_processing_interrupted()


def _sleep(seconds: float):
    end = time.monotonic() + max(0.0, seconds)
    while True:
        _check_interrupted()
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(0.25, left))


def _retry_after(resp: requests.Response) -> Optional[float]:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - time.time())
    except Exception:
        return None


def _backoff(attempt: int, base: float, cap: float) -> float:
    # attempt is 1-based: 2s, 4s, 8s ... with ±25 % jitter so parallel
    # workers that failed together do not retry together.
    delay = min(cap, base * (2 ** (attempt - 1)))
    return delay * random.uniform(0.75, 1.25)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def request(method: str, url: str, *, service: str = "API",
            headers: Optional[dict] = None, params: Optional[dict] = None,
            json=None, data=None, timeout=DEFAULT_TIMEOUT,
            max_retries: int = DEFAULT_MAX_RETRIES,
            backoff_base: float = DEFAULT_BACKOFF_BASE,
            backoff_max: float = DEFAULT_BACKOFF_MAX,
            retry_read_timeout: bool = False,
            raise_for_status: bool = True,
            session: Optional[requests.Session] = None,
            **kwargs) -> requests.Response:
    """Send one HTTP request with the retry rules described in the module docstring.

    Returns the `requests.Response` (status < 400 unless raise_for_status=False).
    Raises OnyxAPIError on failure, never a raw requests exception.
    """
    secrets = _secrets_from(headers, params)
    sender = session or requests
    attempts = 0
    while True:
        attempts += 1
        _check_interrupted()
        try:
            resp = sender.request(method, url, headers=headers, params=params,
                                  json=json, data=data, timeout=timeout, **kwargs)
        except requests.exceptions.ReadTimeout as e:
            if retry_read_timeout and attempts <= max_retries:
                delay = _backoff(attempts, backoff_base, backoff_max)
                print(f"[Onyx HTTP] {service}: read timeout, retry {attempts}/{max_retries} in {delay:.1f}s")
                _sleep(delay)
                continue
            raise OnyxAPIError(
                service,
                f"{service}: no answer within {timeout if not isinstance(timeout, tuple) else timeout[1]}s. "
                f"The provider may still finish (and bill) this job, so it was not replayed "
                f"automatically. Check the provider dashboard before re-queueing.",
                retryable=False, attempts=attempts,
            ) from None
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as e:
            if attempts <= max_retries:
                delay = _backoff(attempts, backoff_base, backoff_max)
                print(f"[Onyx HTTP] {service}: connection error, retry {attempts}/{max_retries} in {delay:.1f}s")
                _sleep(delay)
                continue
            raise OnyxAPIError(
                service,
                f"{service}: could not reach the provider after {attempts} attempts "
                f"(network down, DNS, proxy or firewall). Details: {redact(str(e), secrets)[:300]}",
                retryable=True, attempts=attempts,
            ) from None
        except requests.exceptions.RequestException as e:
            raise OnyxAPIError(service, f"{service}: request error — {redact(str(e), secrets)[:300]}",
                               attempts=attempts) from None

        if resp.status_code < 400 or not raise_for_status:
            return resp

        if resp.status_code in RETRY_STATUSES and attempts <= max_retries:
            wait = _retry_after(resp)
            if wait is not None and wait > MAX_RETRY_AFTER:
                # A long Retry-After is a quota window: waiting it out silently
                # would freeze the queue for minutes. Tell the user instead.
                raise _build_error(service, resp, secrets, attempts)
            delay = wait if wait is not None else _backoff(attempts, backoff_base, backoff_max)
            print(f"[Onyx HTTP] {service}: HTTP {resp.status_code}, retry {attempts}/{max_retries} "
                  f"in {delay:.1f}s")
            _sleep(delay)
            continue

        raise _build_error(service, resp, secrets, attempts)


def post_json(url: str, *, service: str = "API", payload=None, **kwargs) -> dict:
    """POST `payload` as JSON and return the decoded JSON body."""
    resp = request("POST", url, service=service, json=payload, **kwargs)
    return _decode(resp, service, kwargs)


def get_json(url: str, *, service: str = "API", **kwargs) -> dict:
    """GET and return the decoded JSON body. Read timeouts are retried: a GET is free."""
    kwargs.setdefault("retry_read_timeout", True)
    resp = request("GET", url, service=service, **kwargs)
    return _decode(resp, service, kwargs)


def download(url: str, *, service: str = "API", timeout=(15, 120), **kwargs) -> bytes:
    """Fetch a result file (image/video) and return its bytes. Downloads are
    free and idempotent, so read timeouts are retried."""
    kwargs.setdefault("retry_read_timeout", True)
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
