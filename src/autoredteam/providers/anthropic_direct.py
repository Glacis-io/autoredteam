"""Anthropic direct API provider."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from autoredteam.prepare import TargetCapabilities
from autoredteam.providers._compat import ParamMemo, accepts_param
from autoredteam.providers.base import BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderResponse, TargetSpec


def text_from_content_blocks(blocks: Any) -> str:
    """Join text blocks, skipping thinking/tool blocks that newer Claude models emit first."""
    parts = []
    for block in blocks or []:
        if getattr(block, "type", "text") == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


class AnthropicDirectSession(BaseTargetSession):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        try:
            import anthropic
        except ImportError:
            raise ProviderConfigurationError("pip install 'glacis-autoredteam[anthropic]'")
        api_key = spec.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderConfigurationError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key, base_url=spec.endpoint or None)
        self._history: list[dict[str, str]] = []
        self._params = ParamMemo(renames={})
        # anthropic>=1.0 removed sampling parameters from Messages.create.
        self._send_temperature = accepts_param(self._client.messages.create, "temperature")

    def send_user_turn(self, content: str, turn_index: int = 0, metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        self._history.append({"role": "user", "content": content})
        kwargs = {
            "model": self.spec.model,
            "system": self.spec.system_prompt,
            "messages": list(self._history),
            "max_tokens": self.spec.max_output_tokens,
        }
        if self._send_temperature:
            kwargs["temperature"] = self.spec.temperature
        t0 = time.monotonic()
        try:
            resp = self._params.call(self._client.messages.create, kwargs)
        except Exception:
            self._history.pop()
            raise
        latency = (time.monotonic() - t0) * 1000
        text = text_from_content_blocks(resp.content)
        self._history.append({"role": "assistant", "content": text})
        return ProviderResponse(text=text, raw={"id": resp.id}, finish_reason=resp.stop_reason or "", provider_request_id=resp.id or "", latency_ms=round(latency, 1))

    def reset(self) -> None: self._history = []
    def history(self) -> list[dict[str, Any]]: return list(self._history)
    def capabilities(self) -> TargetCapabilities: return TargetCapabilities(multi_turn=True, tool_use=True, system_prompt_configurable=True)


def register(registry) -> None:
    registry.register(ProviderDescriptor(provider_id="anthropic", display_name="Anthropic", auth_mode="api_key", supported_families=["anthropic"], required_fields=["model"]), AnthropicDirectSession)
