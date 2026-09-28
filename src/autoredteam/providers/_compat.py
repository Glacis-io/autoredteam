"""Shared helpers for provider adapters: error classification and parameter fallback."""
from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Mapping, Sequence

from autoredteam.providers.base import ProviderRateLimitError, ProviderRequestError

_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "ratelimit", "429", "throttl", "overloaded", "quota", "resource_exhausted")
_REJECTION_MARKERS = ("unsupported", "not supported", "does not support", "unrecognized", "unknown parameter",
                      "not allowed", "deprecated", "only the default", "cannot be set", "is not permitted",
                      "unexpected keyword argument")

# Newer reasoning models (OpenAI gpt-5+/o-series, Claude with always-on thinking)
# reject sampling parameters or renamed token limits. When the API says a
# parameter is unsupported we drop or rename it and retry.
OPENAI_RENAMES: Mapping[str, str] = {"max_tokens": "max_completion_tokens"}
SAMPLING_PARAMS: Sequence[str] = ("temperature", "top_p")


def is_rate_limit(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "throttl" in name:
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _RATE_LIMIT_MARKERS)


def classify(exc: BaseException) -> ProviderRequestError:
    if isinstance(exc, ProviderRequestError):
        return exc
    if is_rate_limit(exc):
        return ProviderRateLimitError(str(exc))
    return ProviderRequestError(str(exc))


def accepts_param(fn: Callable[..., Any], name: str) -> bool:
    """True if ``fn`` takes ``name`` (SDK majors add and remove sampling params)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _rejected_param(exc: BaseException, kwargs: dict[str, Any], candidates: Sequence[str]) -> str | None:
    msg = str(exc).lower()
    if not any(w in msg for w in _REJECTION_MARKERS):
        return None
    for param in candidates:
        if param in kwargs and re.search(rf"\b{param}\b", msg):
            return param
    return None


def call_with_param_fallback(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    renames: Mapping[str, str] = OPENAI_RENAMES,
    droppable: Sequence[str] = SAMPLING_PARAMS,
    max_adjustments: int = 3,
) -> tuple[Any, dict[str, Any]]:
    """Call ``fn(**kwargs)``, dropping/renaming parameters the model rejects.

    Returns the response and the kwargs that finally worked.
    """
    candidates = list(renames) + list(droppable)
    current = dict(kwargs)
    for attempt in range(max_adjustments + 1):
        try:
            return fn(**current), current
        except Exception as exc:  # noqa: BLE001 — SDK exception types vary by provider
            param = _rejected_param(exc, current, candidates)
            if param is None or attempt == max_adjustments:
                raise classify(exc) from exc
            value = current.pop(param)
            if param in renames:
                current[renames[param]] = value
    raise ProviderRequestError("parameter fallback exhausted")


class ParamMemo:
    """Remembers parameters a model rejected so later turns skip the failed call."""

    def __init__(self, renames: Mapping[str, str] = OPENAI_RENAMES, droppable: Sequence[str] = SAMPLING_PARAMS):
        self._renames = dict(renames)
        self._droppable = tuple(droppable)
        self._adjust: dict[str, str | None] = {}

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        out = dict(kwargs)
        for key, replacement in self._adjust.items():
            if key in out:
                value = out.pop(key)
                if replacement:
                    out[replacement] = value
        return out

    def learn(self, sent: dict[str, Any], used: dict[str, Any]) -> None:
        for key in set(sent) - set(used):
            renamed = self._renames.get(key)
            self._adjust[key] = renamed if renamed in used else None

    def call(self, fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        sent = self.apply(kwargs)
        resp, used = call_with_param_fallback(fn, sent, self._renames, self._droppable)
        self.learn(sent, used)
        return resp
