"""Anthropic direct API provider."""
from __future__ import annotations
import os, time
from typing import Any, Optional
from autoredteam.prepare import TargetCapabilities
from autoredteam.providers.base import BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderRequestError, ProviderRateLimitError, ProviderResponse, TargetSpec

class AnthropicDirectSession(BaseTargetSession):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        try:
            import anthropic
        except ImportError:
            raise ProviderConfigurationError("pip install anthropic")
        api_key = spec.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key: raise ProviderConfigurationError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._history: list[dict[str, str]] = []

    def send_user_turn(self, content: str, turn_index: int = 0, metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        self._history.append({"role": "user", "content": content})
        t0 = time.monotonic()
        try:
            resp = self._client.messages.create(model=self.spec.model, system=self.spec.system_prompt, messages=self._history, temperature=self.spec.temperature, max_tokens=self.spec.max_output_tokens)
        except Exception as e:
            if any(x in str(e).lower() for x in ["rate", "429", "overloaded"]): raise ProviderRateLimitError(str(e)) from e
            raise ProviderRequestError(str(e)) from e
        latency = (time.monotonic() - t0) * 1000
        text = resp.content[0].text if resp.content else ""
        self._history.append({"role": "assistant", "content": text})
        return ProviderResponse(text=text, raw={"id": resp.id}, finish_reason=resp.stop_reason or "", provider_request_id=resp.id or "", latency_ms=round(latency, 1))

    def reset(self) -> None: self._history = []
    def history(self) -> list[dict[str, Any]]: return list(self._history)
    def capabilities(self) -> TargetCapabilities: return TargetCapabilities(multi_turn=True, tool_use=True, system_prompt_configurable=True)

def register(registry) -> None:
    registry.register(ProviderDescriptor(provider_id="anthropic", display_name="Anthropic", auth_mode="api_key", supported_families=["anthropic"], required_fields=["model"]), AnthropicDirectSession)
