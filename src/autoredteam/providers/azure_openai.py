"""Azure OpenAI provider."""
from __future__ import annotations
import os, time
from typing import Any, Optional
from autoredteam.prepare import TargetCapabilities
from autoredteam.providers.base import BaseTargetSession, ProviderConfigurationError, ProviderDescriptor, ProviderRequestError, ProviderRateLimitError, ProviderResponse, TargetSpec

class AzureOpenAISession(BaseTargetSession):
    def __init__(self, spec: TargetSpec):
        super().__init__(spec)
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ProviderConfigurationError("pip install openai")
        endpoint = spec.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_key = spec.api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        api_version = spec.api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not endpoint: raise ProviderConfigurationError("Azure OpenAI endpoint required")
        if not api_key: raise ProviderConfigurationError("AZURE_OPENAI_API_KEY not set")
        self._client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self._deployment = spec.deployment or spec.model
        self._history: list[dict[str, str]] = []

    def send_user_turn(self, content: str, turn_index: int = 0, metadata: Optional[dict[str, Any]] = None) -> ProviderResponse:
        self._history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": self.spec.system_prompt}] + self._history
        t0 = time.monotonic()
        try:
            resp = self._client.chat.completions.create(model=self._deployment, messages=messages, temperature=self.spec.temperature, max_tokens=self.spec.max_output_tokens)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e): raise ProviderRateLimitError(str(e)) from e
            raise ProviderRequestError(str(e)) from e
        latency = (time.monotonic() - t0) * 1000
        text = resp.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": text})
        return ProviderResponse(text=text, raw={"id": resp.id}, finish_reason=resp.choices[0].finish_reason or "", provider_request_id=resp.id or "", latency_ms=round(latency, 1))

    def reset(self) -> None: self._history = []
    def history(self) -> list[dict[str, Any]]: return list(self._history)
    def capabilities(self) -> TargetCapabilities: return TargetCapabilities(multi_turn=True, tool_use=True, system_prompt_configurable=True)

def register(registry) -> None:
    registry.register(ProviderDescriptor(provider_id="azure_openai", display_name="Azure OpenAI", auth_mode="api_key", supported_families=["openai"], required_fields=["model", "endpoint"]), AzureOpenAISession)
