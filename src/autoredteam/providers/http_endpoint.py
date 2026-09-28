"""Generic HTTP endpoint provider — red-team your application's own API, not just the raw model.

Configure through ``TargetSpec.metadata`` (the CLI exposes ``--http-*`` flags):

``http_body``           JSON (or text) request template. Placeholders:
                        ``{{prompt}}``, ``{{system_prompt}}``, ``{{messages}}``
                        (OpenAI-style list incl. system), ``{{history}}`` (list
                        without system), ``{{session_id}}``, ``{{turn_index}}``,
                        ``{{model}}``, ``{{env.NAME}}``.
``http_headers``        dict of headers; values may use ``{{env.NAME}}``.
``http_method``         default ``POST``.
``http_response_path``  where the reply text lives, e.g. ``choices[0].message.content``
                        or ``data.answer``. Without it, common shapes are auto-detected.
``http_timeout``        seconds, default 60.

A placeholder that is the entire JSON string value (``"{{messages}}"``) is
replaced by the native value, so lists and objects stay structured.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Optional

from autoredteam.prepare import TargetCapabilities
from autoredteam.providers.base import (
    BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderRateLimitError,
    ProviderRequestError, ProviderResponse, TargetSpec,
)

DEFAULT_BODY = '{"model": "{{model}}", "messages": "{{messages}}"}'
_PLACEHOLDER = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")
_WHOLE_PLACEHOLDER = re.compile(r"^\{\{\s*([\w.]+)\s*\}\}$")
_AUTODETECT_PATHS = (
    "choices[0].message.content",
    "choices[0].text",
    "content[0].text",
    "output_text",
    "candidates[0].content.parts[0].text",
    "message.content",
)
_AUTODETECT_KEYS = ("response", "output", "answer", "reply", "text", "content", "message", "result", "completion")


def _resolve(name: str, variables: dict[str, Any]) -> Any:
    if name.startswith("env."):
        var = name[4:]
        if var not in os.environ:
            raise ProviderConfigurationError(f"environment variable {var} referenced by HTTP template is not set")
        return os.environ[var]
    if name not in variables:
        raise ProviderConfigurationError(f"unknown HTTP template placeholder {{{{{name}}}}}")
    return variables[name]


def render_string(template: str, variables: dict[str, Any]) -> str:
    def _sub(m: re.Match) -> str:
        value = _resolve(m.group(1), variables)
        return value if isinstance(value, str) else json.dumps(value)
    return _PLACEHOLDER.sub(_sub, template)


def _render_node(node: Any, variables: dict[str, Any]) -> Any:
    if isinstance(node, str):
        whole = _WHOLE_PLACEHOLDER.match(node)
        if whole:
            return _resolve(whole.group(1), variables)
        return render_string(node, variables)
    if isinstance(node, list):
        return [_render_node(v, variables) for v in node]
    if isinstance(node, dict):
        return {k: _render_node(v, variables) for k, v in node.items()}
    return node


def render_body(template: str, variables: dict[str, Any]) -> tuple[bytes, str]:
    """Render a body template. Returns (bytes, content_type)."""
    # Allow bare placeholders in JSON position, e.g. {"messages": {{messages}}}.
    quoted = re.sub(r'(?<=[:\[,])(\s*)(\{\{\s*[\w.]+\s*\}\})(?=\s*[,}\]])', r'\1"\2"', template)
    try:
        parsed = json.loads(quoted)
    except json.JSONDecodeError:
        return render_string(template, variables).encode("utf-8"), "text/plain; charset=utf-8"
    return json.dumps(_render_node(parsed, variables)).encode("utf-8"), "application/json"


_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def extract_path(data: Any, path: str) -> Any:
    cur = data
    for key, index in _PATH_TOKEN.findall(path):
        if index:
            cur = cur[int(index)]
        else:
            cur = cur[key]
    return cur


def extract_text(data: Any, path: str = "") -> str:
    if path:
        try:
            value = extract_path(data, path)
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderRequestError(f"response path '{path}' not found in response: {e}") from e
        return value if isinstance(value, str) else json.dumps(value)
    if isinstance(data, str):
        return data
    for candidate in _AUTODETECT_PATHS:
        try:
            value = extract_path(data, candidate)
        except (KeyError, IndexError, TypeError):
            continue
        if isinstance(value, str):
            return value
    if isinstance(data, dict):
        for key in _AUTODETECT_KEYS:
            if isinstance(data.get(key), str):
                return data[key]
    return json.dumps(data)


class HTTPEndpointSession(BaseTargetSession):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        if not spec.endpoint:
            raise ProviderConfigurationError("http provider requires --endpoint (full URL)")
        meta = spec.metadata or {}
        self._method = str(meta.get("http_method", "POST")).upper()
        self._headers: dict[str, str] = dict(meta.get("http_headers") or {})
        self._body = meta.get("http_body") or DEFAULT_BODY
        self._response_path = meta.get("http_response_path", "")
        self._timeout = float(meta.get("http_timeout", 60))
        self._history: list[dict[str, str]] = []
        self._session_id = uuid.uuid4().hex

    def _variables(self, content: str, turn_index: int) -> dict[str, Any]:
        history = self._history + [{"role": "user", "content": content}]
        return {
            "prompt": content,
            "system_prompt": self.spec.system_prompt,
            "messages": [{"role": "system", "content": self.spec.system_prompt}] + history,
            "history": history,
            "session_id": self._session_id,
            "turn_index": turn_index,
            "model": self.spec.model,
        }

    def send_user_turn(self, content: str, turn_index: int = 0,
                       metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        variables = self._variables(content, turn_index)
        url = render_string(self.spec.endpoint, variables)
        headers = {k: render_string(v, variables) for k, v in self._headers.items()}
        data = None
        if self._method not in ("GET", "HEAD"):
            data, content_type = render_body(self._body, variables)
            headers.setdefault("Content-Type", content_type)
        headers.setdefault("User-Agent", "autoredteam")
        req = urllib.request.Request(url, data=data, method=self._method, headers=headers)

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = resp.status
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            if e.code == 429:
                raise ProviderRateLimitError(f"HTTP 429: {detail}") from e
            raise ProviderRequestError(f"HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise ProviderRequestError(str(e)) from e
        latency = (time.monotonic() - t0) * 1000

        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        text = extract_text(parsed, self._response_path)
        self._history.append({"role": "user", "content": content})
        self._history.append({"role": "assistant", "content": text})
        return ProviderResponse(text=text, raw={"status": status}, finish_reason="stop", latency_ms=round(latency, 1))

    def reset(self) -> None:
        self._history = []
        self._session_id = uuid.uuid4().hex

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(multi_turn=True, system_prompt_configurable=False)


def register(registry) -> None:
    registry.register(
        ProviderDescriptor(
            provider_id="http", display_name="HTTP Endpoint (any API)",
            auth_mode="headers", supported_families=[], required_fields=["endpoint"],
        ),
        HTTPEndpointSession,
    )
